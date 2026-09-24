"""Blocking / candidate generation: exact normalized-name inverted index.

Blocking proposes candidate pairs. It never decides whether a pair is a match -
that is the ranking model's job. So the only thing that matters here is
**recall at a tolerable candidate volume**.

The naive alternative is 2,206,821 x 10,320,219 = 22.8 trillion comparisons,
which is why we index instead: look up each S1 key and read back only the rows
that share it.

Implementation notes
--------------------
The textbook implementation is a ``dict[str, list[int]]``. For S2 that dict
would hold ~4.0M keys with string keys, costing roughly 700MB-1GB of RSS before
postings. Since the challenge dataset is HPC-scale and repeated over two target
sources, this module stores the same information in flat numpy arrays:

============  ======================  ===================  ================
array         dtype                   size (S2, ~5M rows)  purpose
============  ======================  ===================  ================
key_hashes    uint64, sorted          ~32MB (4M uniques)   binary-search lookup
key_offsets   int64                   ~32MB                slice into keys_blob
keys_blob     bytes (concatenated)    ~100MB               exact verification
postings      int64                   ~40MB                entity ids, key-grouped
post_offsets  int64                   ~32MB                slice into postings
============  ======================  ===================  ================

Peak build RSS is roughly 250MB for S2 instead of ~1GB, and lookup is a
``searchsorted`` instead of a dict probe.

**Hashes are verified against the real string on every lookup.** A hash match
that fails string comparison is rejected, so the index can never emit a
spurious pair. The reverse (two distinct keys sharing a 64-bit hash, causing a
missed pair) has probability ~4e-7 at this key count and is accepted: blocking
should fail toward "miss" rather than "wrong candidate".

Union semantics
---------------
Multiple blockers each return postings for the same S1. The candidate set is
their **union**, computed with one ``np.unique`` over packed
``(s1_position, entity_code)`` integers, which also deduplicates for free.
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

from .data_loader import (
    TARGET_SOURCES,
    SOURCE_PREFIX,
    iter_prepared,
    prepared_path,
    require_file,
)
from .utils import (
    ID_NUMERIC_MODULUS,
    ID_SOURCE_NAMES,
    decode_entity_ids,
    encode_entity_ids,
    ensure_dir,
    human_bytes,
    read_json,
    stable_hash64,
    write_json,
)

logger = logging.getLogger(__name__)

# On-disk format version. Bump when the layout changes so stale indexes are
# detected instead of silently misread.
INDEX_VERSION = 1
HASH_NAME = "blake2b-64"
META_FILE = "meta.json"
KEYS_FILE = "keys.bin"
KEY_HASHES_FILE = "key_hashes.npy"
KEY_OFFSETS_FILE = "key_offsets.npy"
POSTINGS_FILE = "postings.npy"
POSTINGS_OFFSETS_FILE = "postings_offsets.npy"

# Packing factor for (s1_position, entity_code) -> single int64.
# Entity codes are <= 3*10**10 + 10**9 < 3.2e10, so 10**11 separates them
# cleanly; s1_position <= 2.2e6 gives a max packed value of ~2.2e17, well inside
# int64 range. Packing lets np.unique do union + dedupe + sort in one call.
PAIR_MULTIPLIER = 10**11

# Blocker registry. Token / n-gram / dense blockers plug in here as they are
# implemented; the CLI and the index loader are already generic.
BLOCKER_EXACT_NAME = "exact_name"
BLOCKER_TOKEN = "token"
BLOCKER_CHAR_NGRAM = "char_ngram"
BLOCKER_DENSE = "dense"
KNOWN_BLOCKERS = (BLOCKER_EXACT_NAME, BLOCKER_TOKEN, BLOCKER_CHAR_NGRAM, BLOCKER_DENSE)


def index_dir_for(config: dict, split: str, source: str, blocker: str) -> Path:
    """Directory holding one persisted index."""
    return Path(config["resolved"]["index_dir"]) / f"{split}_{source}_{blocker}"


def _key_field_for(config: dict, blocker: str) -> str:
    """Which normalized column a blocker keys on, from config."""
    section = config.get("blocking", {}).get(blocker, {}) or {}
    if blocker == BLOCKER_EXACT_NAME:
        return section.get("key", "name_norm")
    return section.get("key", "name_norm")


# ---------------------------------------------------------------------------
# Exact-name index
# ---------------------------------------------------------------------------
class ExactNameIndex:
    """Inverted index from an exact normalized key to entity ids.

    Thread- and process-safe for reads; building is single-process.

    Attributes:
        source: logical source name, e.g. ``"source2"``.
        prefix: id prefix, e.g. ``"S2"``.
        key_field: the normalized column that was indexed.
        key_hashes: sorted uint64 hashes of the unique keys.
        postings: int64 entity id codes, grouped by key.
    """

    __slots__ = (
        "source",
        "prefix",
        "key_field",
        "key_hashes",
        "key_offsets",
        "keys_blob",
        "postings_offsets",
        "postings",
        "n_entities_indexed",
        "n_skipped_empty",
    )

    def __init__(
        self,
        source: str,
        prefix: str,
        key_field: str,
        key_hashes: np.ndarray,
        key_offsets: np.ndarray,
        keys_blob: bytes,
        postings_offsets: np.ndarray,
        postings: np.ndarray,
        n_entities_indexed: int = 0,
        n_skipped_empty: int = 0,
    ) -> None:
        self.source = source
        self.prefix = prefix
        self.key_field = key_field
        self.key_hashes = key_hashes
        self.key_offsets = key_offsets
        self.keys_blob = keys_blob
        self.postings_offsets = postings_offsets
        self.postings = postings
        self.n_entities_indexed = n_entities_indexed
        self.n_skipped_empty = n_skipped_empty

    # -- introspection ------------------------------------------------------
    @property
    def n_unique_keys(self) -> int:
        return len(self.key_hashes)

    @property
    def n_postings(self) -> int:
        return len(self.postings)

    @property
    def average_postings_per_key(self) -> float:
        return self.n_postings / self.n_unique_keys if self.n_unique_keys else 0.0

    def memory_bytes(self) -> int:
        return (
            self.key_hashes.nbytes
            + self.key_offsets.nbytes
            + len(self.keys_blob)
            + self.postings_offsets.nbytes
            + self.postings.nbytes
        )

    def describe(self) -> dict:
        return {
            "source": self.source,
            "key_field": self.key_field,
            "n_entities_indexed": int(self.n_entities_indexed),
            "n_skipped_empty_keys": int(self.n_skipped_empty),
            "n_unique_keys": int(self.n_unique_keys),
            "n_postings": int(self.n_postings),
            "avg_postings_per_key": round(self.average_postings_per_key, 3),
            "max_postings_per_key": int(np.diff(self.postings_offsets).max()) if self.n_unique_keys else 0,
            "index_memory": human_bytes(self.memory_bytes()),
        }

    def key_at(self, position: int) -> str:
        """Decode the unique key stored at ``position``."""
        start = int(self.key_offsets[position])
        end = int(self.key_offsets[position + 1])
        return self.keys_blob[start:end].decode("utf-8")

    def postings_for_position(self, position: int) -> np.ndarray:
        """Entity id codes for the key at ``position`` (a view, no copy)."""
        return self.postings[self.postings_offsets[position] : self.postings_offsets[position + 1]]

    # -- build --------------------------------------------------------------
    @classmethod
    def build(
        cls,
        chunk_iterator: Iterable[pd.DataFrame],
        source: str,
        prefix: str,
        key_field: str = "name_norm",
        entity_column: str = "entity_id",
        log: Optional[logging.Logger] = None,
        total_rows: Optional[int] = None,
    ) -> "ExactNameIndex":
        """Build an index from an iterable of prepared chunks.

        Memory: O(rows) transiently - roughly ``16 * rows`` bytes for hashes and
        ids plus the UTF-8 key blob (~25 bytes/row here). For S2 that peaks near
        250MB and drops back after grouping. Chunking keeps the *reader* flat;
        the accumulators are inherent to a single-pass build.

        Args:
            chunk_iterator: yields prepared chunks containing ``entity_column``
                and ``key_field``.
            source: logical source name.
            prefix: id prefix (``"S2"``/``"S3"``).
            key_field: normalized column to index.
            entity_column: id column.
            log: logger for progress.
            total_rows: expected row count, for progress reporting.

        Returns:
            A populated :class:`ExactNameIndex`.
        """
        log = log or logger
        hash_parts: list[np.ndarray] = []
        id_parts: list[np.ndarray] = []
        blob = bytearray()
        offset_parts: list[np.ndarray] = []

        rows_seen = 0
        rows_kept = 0
        rows_skipped = 0

        for chunk in chunk_iterator:
            if entity_column not in chunk.columns or key_field not in chunk.columns:
                raise KeyError(
                    f"prepared chunk is missing {entity_column!r} or {key_field!r}; "
                    f"found {list(chunk.columns)}"
                )
            rows_seen += len(chunk)

            keys = chunk[key_field].to_numpy(dtype=object)
            ids = chunk[entity_column].to_numpy(dtype=object)

            # Records whose normalized name is empty carry no blocking signal.
            # Indexing them would put every such record in one giant bucket and
            # generate all-pairs noise for zero recall benefit.
            keep_mask = np.fromiter(
                (isinstance(k, str) and len(k) > 0 for k in keys), dtype=bool, count=len(keys)
            )
            skipped = int((~keep_mask).sum())
            if skipped:
                rows_skipped += skipped
                keys = keys[keep_mask]
                ids = ids[keep_mask]
            if len(keys) == 0:
                continue

            rows_kept += len(keys)
            hash_parts.append(stable_hash64(keys))

            # Store globally packed codes (source_code * 10**10 + numeric, see
            # utils.encode_entity_id). Packing the source in - rather than the
            # bare numeric id - is what lets a union across S2 and S3 stay
            # self-describing: every code knows which source it came from, so no
            # separate provenance array is needed to decode it back to "S2-123".
            # Verified against the full dataset: ids have no leading zeros, so
            # this round-trips losslessly.
            id_parts.append(encode_entity_ids(pd.Series(ids)))

            # UTF-8 key blob. Encoded once here and reused for offsets.
            encoded = [k.encode("utf-8") for k in keys]
            lengths = np.fromiter((len(e) for e in encoded), dtype=np.int64, count=len(encoded))
            chunk_offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
            np.cumsum(lengths, out=chunk_offsets[1:])
            chunk_offsets += len(blob)
            offset_parts.append(chunk_offsets[:-1])
            blob.extend(b"".join(encoded))

            if log:
                log.info(
                    "  read %s/%s rows (%s indexable, %s skipped empty)",
                    f"{rows_seen:,}",
                    f"{total_rows:,}" if total_rows else "?",
                    f"{rows_kept:,}",
                    f"{rows_skipped:,}",
                )

            del encoded, lengths, chunk_offsets, keys, ids

        if not hash_parts:
            raise ValueError(f"no indexable rows found for {source} (every {key_field} was empty?)")

        row_offsets = np.concatenate(offset_parts)
        hashes = np.concatenate(hash_parts)
        ids = np.concatenate(id_parts)
        del hash_parts, id_parts, offset_parts

        if log:
            log.info(
                "  grouping %s rows (%s) ...",
                f"{rows_kept:,}",
                human_bytes(hashes.nbytes + ids.nbytes + len(blob)),
            )

        # Sort by hash so identical keys become adjacent.
        order = np.argsort(hashes, kind="stable")
        sorted_hashes = hashes[order]
        sorted_ids = ids[order]

        # Group boundaries: first index of each distinct hash.
        is_new_group = np.empty(len(sorted_hashes), dtype=bool)
        is_new_group[0] = True
        if len(sorted_hashes) > 1:
            np.not_equal(sorted_hashes[1:], sorted_hashes[:-1], out=is_new_group[1:])
        group_starts = np.flatnonzero(is_new_group)

        key_hashes = sorted_hashes[group_starts]
        postings_offsets = np.empty(len(group_starts) + 1, dtype=np.int64)
        postings_offsets[:-1] = group_starts
        postings_offsets[-1] = len(sorted_hashes)
        postings = sorted_ids  # already grouped: rows within a group are contiguous

        # One representative key per group, kept for exact verification at query
        # time. row_offsets holds one entry per row, so a trailing sentinel is
        # needed before it can be used to slice the last row's key.
        representative_rows = order[group_starts]
        row_offsets_full = np.empty(len(row_offsets) + 1, dtype=np.int64)
        row_offsets_full[:-1] = row_offsets
        row_offsets_full[-1] = len(blob)
        rep_starts = row_offsets_full[representative_rows]
        rep_ends = row_offsets_full[representative_rows + 1]

        unique_blob = bytearray()
        unique_offsets = np.empty(len(representative_rows) + 1, dtype=np.int64)
        cursor = 0
        for i in range(len(representative_rows)):
            unique_offsets[i] = cursor
            unique_blob.extend(blob[int(rep_starts[i]) : int(rep_ends[i])])
            cursor += int(rep_ends[i] - rep_starts[i])
        unique_offsets[-1] = cursor

        del blob, row_offsets, row_offsets_full, rep_starts, rep_ends
        del hashes, ids, order, sorted_hashes, sorted_ids, is_new_group, group_starts
        gc.collect()

        index = cls(
            source=source,
            prefix=prefix,
            key_field=key_field,
            key_hashes=key_hashes,
            key_offsets=unique_offsets,
            keys_blob=bytes(unique_blob),
            postings_offsets=postings_offsets,
            postings=postings,
            n_entities_indexed=rows_kept,
            n_skipped_empty=rows_skipped,
        )

        if log:
            log.info(
                "  built %s index: %s rows -> %s unique keys (%s), skipped %s empty",
                source,
                f"{rows_kept:,}",
                f"{index.n_unique_keys:,}",
                human_bytes(index.memory_bytes()),
                f"{rows_skipped:,}",
            )
        return index

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | os.PathLike) -> Path:
        """Persist the index as flat files in ``directory``. Returns the path."""
        directory = ensure_dir(directory)
        np.save(directory / KEY_HASHES_FILE, self.key_hashes)
        np.save(directory / KEY_OFFSETS_FILE, self.key_offsets)
        np.save(directory / POSTINGS_FILE, self.postings)
        np.save(directory / POSTINGS_OFFSETS_FILE, self.postings_offsets)
        with open(directory / KEYS_FILE, "wb") as handle:
            handle.write(self.keys_blob)
        write_json(
            directory / META_FILE,
            {
                "index_version": INDEX_VERSION,
                "hash": HASH_NAME,
                "blocker": BLOCKER_EXACT_NAME,
                "source": self.source,
                "prefix": self.prefix,
                "key_field": self.key_field,
                "n_entities_indexed": int(self.n_entities_indexed),
                "n_skipped_empty_keys": int(self.n_skipped_empty),
                "n_unique_keys": int(self.n_unique_keys),
                "n_postings": int(self.n_postings),
                "files": {
                    "key_hashes": KEY_HASHES_FILE,
                    "key_offsets": KEY_OFFSETS_FILE,
                    "keys_blob": KEYS_FILE,
                    "postings": POSTINGS_FILE,
                    "postings_offsets": POSTINGS_OFFSETS_FILE,
                },
            },
        )
        return directory

    @classmethod
    def load(cls, directory: str | os.PathLike, log: Optional[logging.Logger] = None) -> "ExactNameIndex":
        """Load a persisted index, validating the format version."""
        directory = Path(directory)
        meta_path = directory / META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"no index at {directory}\n  Run: python scripts/build_indexes.py"
            )
        meta = read_json(meta_path)
        version = meta.get("index_version")
        if version != INDEX_VERSION:
            raise ValueError(
                f"index at {directory} has version {version}, expected {INDEX_VERSION}. "
                f"Rebuild it (delete the directory and rerun build_indexes.py)."
            )
        if meta.get("hash") != HASH_NAME:
            raise ValueError(f"index hash scheme {meta.get('hash')!r} != {HASH_NAME!r}; rebuild the index")

        with open(directory / KEYS_FILE, "rb") as handle:
            keys_blob = handle.read()

        index = cls(
            source=meta["source"],
            prefix=meta["prefix"],
            key_field=meta["key_field"],
            key_hashes=np.load(directory / KEY_HASHES_FILE),
            key_offsets=np.load(directory / KEY_OFFSETS_FILE),
            keys_blob=keys_blob,
            postings_offsets=np.load(directory / POSTINGS_OFFSETS_FILE),
            postings=np.load(directory / POSTINGS_FILE),
            n_entities_indexed=int(meta.get("n_entities_indexed", 0)),
            n_skipped_empty=int(meta.get("n_skipped_empty_keys", 0)),
        )
        if log:
            log.info("loaded index %s: %s", directory.name, index.describe())
        return index

    # -- query --------------------------------------------------------------
    _EMPTY = np.empty(0, dtype=np.int64)

    def _positions_for_hashes(self, hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Binary-search hashes. Returns (position, found_mask) with -1 for misses."""
        n_keys = self.n_unique_keys
        if n_keys == 0:
            return np.full(len(hashes), -1, dtype=np.int64), np.zeros(len(hashes), dtype=bool)

        positions = np.searchsorted(self.key_hashes, hashes).astype(np.int64)
        in_range = positions < n_keys
        clipped = np.minimum(positions, n_keys - 1)
        found = in_range & (self.key_hashes[clipped] == hashes)
        positions = np.where(found, clipped, -1)
        return positions, found

    def _verify_strings(self, positions: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Confirm hash matches by comparing the stored key strings.

        Decodes each distinct position once, so the common case (many S1 sharing
        a key) costs a handful of decodes rather than one per query.
        """
        valid = positions >= 0
        if not valid.any():
            return valid
        uniq_positions, inverse = np.unique(positions[valid], return_inverse=True)
        decoded = np.array([self.key_at(int(p)) for p in uniq_positions], dtype=object)
        matched = np.empty(len(positions), dtype=bool)
        matched[:] = False
        matched[valid] = decoded[inverse] == keys[valid]
        return matched

    def lookup_codes(self, key: str) -> np.ndarray:
        """Entity id codes for an exact key match. Empty array when absent."""
        if not key:
            return self._EMPTY
        hashed = np.array([stable_hash64(key)], dtype=np.uint64)
        positions, found = self._positions_for_hashes(hashed)
        if not found[0] or self.key_at(int(positions[0])) != key:
            return self._EMPTY
        return self.postings_for_position(int(positions[0]))

    def lookup_many(self, keys: pd.Series | Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """Batched lookup returning ``(positions, counts)``.

        ``positions[i]`` is the unique-key position for ``keys[i]`` (-1 on miss)
        and ``counts[i]`` the number of postings. Empty keys always miss.

        Memory: O(len(keys)) int64s - 2.2M S1 entities is ~35MB.
        """
        key_array = keys.to_numpy(dtype=object) if isinstance(keys, pd.Series) else np.asarray(keys, dtype=object)
        non_empty = np.fromiter(
            (isinstance(k, str) and len(k) > 0 for k in key_array), dtype=bool, count=len(key_array)
        )
        positions = np.full(len(key_array), -1, dtype=np.int64)
        counts = np.zeros(len(key_array), dtype=np.int64)
        if not non_empty.any():
            return positions, counts

        hashed = stable_hash64(key_array[non_empty])
        found_positions, found = self._positions_for_hashes(hashed)
        verified = self._verify_strings(found_positions, key_array[non_empty])
        final_positions = np.where(found & verified, found_positions, -1)

        subset_counts = np.zeros(len(key_array), dtype=np.int64)
        valid = final_positions >= 0
        if valid.any():
            offset_index = final_positions[valid]
            subset_counts[valid] = (
                self.postings_offsets[offset_index + 1] - self.postings_offsets[offset_index]
            )

        positions[non_empty] = final_positions
        counts[non_empty] = subset_counts
        return positions, counts

    def expand(self, positions: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expand CSR-style ``(positions, counts)`` into flat postings.

        Returns ``(owner_index, entity_codes)`` where ``owner_index[i]`` is the
        query row that produced ``entity_codes[i]``. Uses the standard
        repeat/arange trick, so it is fully vectorized.

        Memory: O(total candidates) - which is exactly why candidate volume is
        tracked and capped rather than assumed small.
        """
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        valid = positions >= 0
        start_offsets = self.postings_offsets[positions[valid]]
        starts_flat = np.repeat(start_offsets, counts[valid])
        group_base = np.repeat(np.cumsum(counts[valid]) - counts[valid], counts[valid])
        within_group = np.arange(total, dtype=np.int64) - group_base
        flat_indices = starts_flat + within_group
        entity_codes = self.postings[flat_indices]
        owner_index = np.repeat(np.flatnonzero(valid), counts[valid])
        return owner_index, entity_codes


# ---------------------------------------------------------------------------
# Not-yet-implemented blockers (roadmap). Registered so the CLI and the config
# already accept them; the error message says what is missing.
# ---------------------------------------------------------------------------
def _not_implemented(blocker: str):
    def _builder(*args: Any, **kwargs: Any):
        raise NotImplementedError(
            f"blocker {blocker!r} is planned but not implemented yet.\n"
            f"  Milestone 1 implements {BLOCKER_EXACT_NAME!r} only.\n"
            f"  See README 'Roadmap' for the incremental plan."
        )

    return _builder


INDEX_BUILDERS = {
    BLOCKER_EXACT_NAME: ExactNameIndex.build,
    BLOCKER_TOKEN: _not_implemented(BLOCKER_TOKEN),
    BLOCKER_CHAR_NGRAM: _not_implemented(BLOCKER_CHAR_NGRAM),
    BLOCKER_DENSE: _not_implemented(BLOCKER_DENSE),
}


def build_index(
    config: dict,
    split: str,
    source: str,
    blocker: str = BLOCKER_EXACT_NAME,
    limit: Optional[int] = None,
    log: Optional[logging.Logger] = None,
    overwrite: bool = False,
    total_rows: Optional[int] = None,
) -> ExactNameIndex:
    """Build (and persist) one blocker index for one target source.

    Args:
        config: loaded config.
        split: ``train`` or ``test``.
        source: ``source2`` or ``source3``.
        blocker: blocker name from :data:`KNOWN_BLOCKERS`.
        limit: read at most this many rows (for smoke tests on a laptop).
        log: logger.
        overwrite: rebuild even if a valid index already exists.

    Returns:
        The built index (also written to ``config['resolved']['index_dir']``).
    """
    if source not in TARGET_SOURCES:
        raise ValueError(f"indexes are for target sources {TARGET_SOURCES}, got {source!r}")
    if blocker not in INDEX_BUILDERS:
        raise ValueError(f"unknown blocker {blocker!r}; expected one of {KNOWN_BLOCKERS}")

    log = log or logger
    directory = index_dir_for(config, split, source, blocker)

    if not overwrite:
        meta_path = directory / META_FILE
        if meta_path.is_file():
            existing = read_json(meta_path).get("index_version")
            if existing == INDEX_VERSION:
                log.info("index already exists at %s - loading instead of rebuilding", directory)
                return ExactNameIndex.load(directory, log=log)
            log.warning("index at %s is version %s, rebuilding", directory, existing)

    key_field = _key_field_for(config, blocker)
    path = prepared_path(config, split, source)
    require_file(path, hint="Run: python scripts/prepare_data.py")

    log.info(
        "building %s index | source=%s split=%s key_field=%s limit=%s",
        blocker,
        source,
        split,
        key_field,
        limit or "none",
    )

    def _chunks() -> Iterator[pd.DataFrame]:
        columns = ["entity_id", key_field]
        rows = 0
        chunksize = config.get("io", {}).get("chunksize", 500_000)
        if limit:
            chunksize = min(chunksize, limit)
        for chunk in iter_prepared(config, split, source, columns=columns, chunksize=chunksize):
            if limit is not None and rows + len(chunk) > limit:
                chunk = chunk.iloc[: limit - rows]
            rows += len(chunk)
            yield chunk
            if limit is not None and rows >= limit:
                break

    builder = INDEX_BUILDERS[blocker]
    index = builder(
        _chunks(),
        source=source,
        prefix=SOURCE_PREFIX[source],
        key_field=key_field,
        log=log,
        total_rows=total_rows,
    )
    index.save(directory)
    log.info("saved index to %s", directory)
    return index


def load_index(
    config: dict,
    split: str,
    source: str,
    blocker: str = BLOCKER_EXACT_NAME,
    log: Optional[logging.Logger] = None,
) -> ExactNameIndex:
    """Load a persisted index, with a clear error if it is missing."""
    return ExactNameIndex.load(index_dir_for(config, split, source, blocker), log=log)


# ---------------------------------------------------------------------------
# Union of blockers
# ---------------------------------------------------------------------------
def pack_pairs(s1_positions: np.ndarray, entity_codes: np.ndarray) -> np.ndarray:
    """Pack ``(s1_position, entity_code)`` pairs into single int64s.

    Packing is what makes the union a one-liner: ``np.unique`` on packed ints
    simultaneously deduplicates, unions and sorts by (S1, entity).
    """
    if len(s1_positions) != len(entity_codes):
        raise ValueError("s1_positions and entity_codes must be the same length")
    return s1_positions.astype(np.int64) * PAIR_MULTIPLIER + entity_codes.astype(np.int64)


def unpack_pairs(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pack_pairs`."""
    s1_positions, entity_codes = np.divmod(packed, PAIR_MULTIPLIER)
    return s1_positions, entity_codes


def union_blockers(
    blocker_pairs: dict[str, np.ndarray],
    log: Optional[logging.Logger] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Union candidate pairs from several blockers, deduplicating per S1.

    Args:
        blocker_pairs: ``{blocker_name: packed_pairs}`` exactly as produced by
            :func:`pack_pairs`. Every array refers to the same S1 positions.
        log: logger.

    Returns:
        ``(s1_positions, entity_codes, blockers_per_pair)``, sorted by
        (s1_position, entity_code). ``blockers_per_pair`` is an object array of
        comma-joined blocker names, so provenance survives the union - which is
        what later lets us audit which blocker is actually earning its keep.
    """
    non_empty = {name: arr for name, arr in blocker_pairs.items() if len(arr)}
    if not non_empty:
        empty_packed = np.empty(0, dtype=np.int64)
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=object)

    all_packed = np.concatenate(list(non_empty.values()))
    unique_packed = np.unique(all_packed)
    s1_positions, entity_codes = unpack_pairs(unique_packed)

    if len(non_empty) == 1:
        only_name = next(iter(non_empty))
        provenance = np.full(len(unique_packed), only_name, dtype=object)
    else:
        provenance = np.full(len(unique_packed), "", dtype=object)
        for name, packed in non_empty.items():
            # searchsorted + equality: a set membership test without building
            # python sets (millions of elements).
            idx = np.searchsorted(unique_packed, packed)
            np.clip(idx, 0, len(unique_packed) - 1, out=idx)
            present = unique_packed[idx] == packed
            mask = np.zeros(len(unique_packed), dtype=bool)
            mask[idx[present]] = True
            provenance[mask] = np.where(
                provenance[mask] == "", name, np.char.add(provenance[mask].astype(str), f",{name}")
            )

    if log:
        log.info(
            "union: %s blocker(s) -> %s unique pairs",
            len(non_empty),
            f"{len(unique_packed):,}",
        )
    return s1_positions, entity_codes, provenance


def truncate_per_group(
    s1_positions_sorted: np.ndarray,
    cap: int,
    log: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Boolean keep-mask limiting each S1 to its first ``cap`` candidates.

    Relies on ``s1_positions_sorted`` being sorted (which the union guarantees).
    Deterministic: "first" is by entity code, so a rerun keeps the same pairs.

    Args:
        s1_positions_sorted: sorted S1 positions.
        cap: maximum candidates per S1. ``<=0`` means no limit.
        log: logger.

    Returns:
        Boolean array aligned to the input.
    """
    if cap is None or cap <= 0 or len(s1_positions_sorted) == 0:
        return np.ones(len(s1_positions_sorted), dtype=bool)

    is_new_group = np.empty(len(s1_positions_sorted), dtype=bool)
    is_new_group[0] = True
    if len(s1_positions_sorted) > 1:
        np.not_equal(s1_positions_sorted[1:], s1_positions_sorted[:-1], out=is_new_group[1:])
    group_starts = np.flatnonzero(is_new_group)
    group_sizes = np.diff(np.append(group_starts, len(s1_positions_sorted)))
    position_in_group = np.arange(len(s1_positions_sorted), dtype=np.int64) - np.repeat(
        group_starts, group_sizes
    )
    keep = position_in_group < cap
    if log:
        dropped = int((~keep).sum())
        if dropped:
            log.warning("cap=%s dropped %s candidate pairs across %s S1 entities", cap, f"{dropped:,}", f"{len(group_starts):,}")
    return keep


def decode_candidates(entity_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn packed entity codes into ``("S2-12345", "S2")`` arrays.

    Works on codes from any mix of sources, because the source is packed into the
    code (see ``utils.encode_entity_id``). Returns both the id strings and the
    source labels, since the candidate file records both.
    """
    codes = np.asarray(entity_codes, dtype=np.int64)
    if len(codes) == 0:
        return np.empty(0, dtype=object), np.empty(0, dtype=object)

    source_codes = codes // ID_NUMERIC_MODULUS
    target_ids = decode_entity_ids(codes)
    source_labels = np.full(len(codes), "", dtype=object)
    known = np.zeros(len(codes), dtype=bool)
    for source_code, prefix in ID_SOURCE_NAMES.items():
        mask = source_codes == source_code
        if mask.any():
            source_labels[mask] = prefix
            known |= mask
    if not known.all():  # pragma: no cover - encode/decode are guarded by the codec
        raise ValueError(
            "packed entity codes contained an unknown source; the index is stale - rebuild it"
        )
    return target_ids, source_labels
