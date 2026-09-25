#!/usr/bin/env python
"""Phase 1 calibration: the char-3-gram blocker's recall x candidate-volume curve.

The question this answers
------------------------
Phase 0.2-0.5 established that the ``char_3gram_name`` signal - the Jaccard
overlap of the character trigrams of ``name_key`` - is the only cheap signal that
reaches most of the interesting entities. Phase 0 also declined to price it: every
threshold row for ``char_3gram_name`` carries
``"candidate_cost": {"kind": "not_estimable_in_phase_0"}``. Phase 0 measured the
signal on ground-truth pairs, which says nothing about what it costs to
*retrieve* those pairs.

That missing number gates the whole design. Macro F0.5 charges a false positive
four times a false negative, so a blocker that hands the matcher a billion
candidates is not "high recall", it is unusable: against ~7.6M true pairs a
candidate set of 2B rows has global precision ~0.4%, and an accept-all matcher on
it scores approximately zero. Recall and volume have to be chosen together, from a
measured curve.

What it measures
----------------
A grid over three axes:

* **trigram DF cap** - which character trigrams may act as keys at all. The head
  of the distribution is noise (``ent``, `` th``, ``ter``); the tail carries the
  identity.
* **rarest-K** - how many of an entity's surviving trigrams act as its keys.
* **Jaccard threshold** - the exact 3-gram Jaccard a retrieved pair must clear.

For every cell it reports the recall metrics, the volume metrics and the per-source
S2/S3 breakdown, for two candidate sets: ``char`` (the blocker alone) and
``char_plus_exact`` (unioned with the shipped exact-name index). Metrics are
reported over all entities and over the S1-level validation split.

Why retrieval is not brute force
--------------------------------
The ~22.8 trillion S1 x target cross product is never formed and never scored.
Retrieval is an inverted index over trigram keys, per target source:

1. count the document frequency of every distinct trigram over that source's
   corpus (corpus-relative by construction - no learned vocabulary, no hardcoded
   business-name word list);
2. retain trigrams with ``df <= max(df_caps)``;
3. per entity, order its surviving trigrams by df ascending, keep the first
   ``max(rarest_ks)`` as keys, and record each key's rank;
4. build one posting list per key. An entity contributes **many** keys - the
   property the shipped ``ExactNameIndex`` does not have, being one-key-per-entity,
   which is why the token and char blockers could not reuse it as-is;
5. query each S1 entity with the same rarest-K rule and union the postings;
6. verify every retrieved pair with the exact 3-gram Jaccard and threshold.

Why one index serves the whole grid
-----------------------------------
A cell ``(df_cap, rarest_k)`` is a pure filter of the loosest cell
``(max(df_caps), max(rarest_ks))``: keep the postings whose key has
``df <= df_cap`` and whose rank is ``< rarest_k``. Because each entity's keys were
assigned ranks in df-ascending order, that filter yields exactly the same key set
as building the cell's index from scratch - the rarest
``min(rarest_k, survivors)`` trigrams among the ``df <= df_cap`` survivors, where
``survivors`` is what the cap leaves. So the corpus is indexed once per source
instead of once per cell, and a cell costs a gather over the posting array.
``tests/test_char_blocker_calibration.py`` pins that equivalence against a
directly-built cell index.

Volume is measured in two kinds, and every row says which it is
---------------------------------------------------------------
A candidate pair is a distinct ``(S1 entity, target entity)`` pair: reaching one
target through two of an S1 entity's keys is still one candidate, and the shipped
generator deduplicates with ``np.unique`` on packed pairs. Two measurements are
therefore reported, and conflating them would misstate the cost by whatever factor
key overlap produces:

* **exact** (``"kind": "exact"``) - the distinct pair count, read from the same
  accumulator the evaluator's ``n_candidate_pairs`` comes from. Available for every
  cell that reaches the evaluation stage, and necessarily equal to
  ``metrics["n_candidate_pairs"]`` for that (cell, set, threshold).
* **upper bound** (``"kind": "upper_bound_from_posting_expansion"``) - the postings a
  cell's queries expand to, counted as prefix lengths rather than materialized:

      bound(cell) = sum over surviving query keys of |{postings of that key with rank < K}|

  The prefix lengths are a cumulative histogram over the posting array, computed once
  per source, so this prices every one of the grid's cells - including the ones far too
  large to retrieve - from one in-memory pass, with no expansion and no prepared table
  re-read. Because key overlap inflates it, it is an upper bound on the distinct count,
  and the evaluated cells report both so the gap is visible.

The bound is also what gates the evaluation stage: a cell whose bound exceeds
``--max-candidate-rows`` would cost more than the budget to retrieve, so it is reported
with its bound, ``"evaluated": false`` and an explicit reason, and its recall is never
presented as measured. ``--max-candidate-rows 0`` removes the guard.

Verification is bit-identical to Phase 0.1
------------------------------------------
The threshold is applied with ``_trigram_jaccard`` imported unchanged from
``scripts/analyze_name_differences.py`` and called per pair on decoded names. It is
not reimplemented, so identity with the Phase 0.1 signal holds by construction
rather than by argument. Verification is per-pair python work, so it is sharded over
worker processes with an in-flight window bounded against a RAM budget.

Retrieval, unlike verification, is unavoidably narrower than the reference signal,
and the gap is counted rather than hidden. ``_trigram_jaccard`` switches to bare
**character** sets when *either* name is shorter than three code points, so it scores
a pair of short names - ``"ab"`` against ``"ab"`` - at 1.0. A trigram-keyed index has
no key for such a name at all, so such a pair can never be retrieved. Any trigram
blocker has this blind spot; the ``structural`` block of the report counts the S1
entities it applies to (``s1_name_too_short_for_any_trigram``) instead of leaving the
reader to infer it from a recall deficit.

Honesty about cost and truncation
---------------------------------
Every cell in the grid gets a volume number, and every number is labelled
``exact`` or ``upper_bound_from_posting_expansion``. A cell above the budget keeps
its bound, reports ``"evaluated": false`` and the reason, and its recall is never
presented as measured. Nothing is ever silently capped, and no truncated cell is
described as full recall.

Outputs (``--output-dir``, default ``<work_dir>/calibration``)
-------------------------------------------------------------
* ``char_blocker_calibration.json`` - meta, grid, every cell's metrics
* ``char_blocker_calibration.csv`` - one row per (cap, K, threshold, set)
* ``char_blocker_volume.csv``   - the volume curve, one row per (cap, K, scope, kind)
* ``char_blocker_calibration.md`` - concise human-readable summary
* ``char_blocker_top_trigrams.csv`` - head of the df table, as evidence
* ``_artifacts/``               - corpus artifacts reused by ``--resume``

Usage::

    # HPC, full grid
    python scripts/calibrate_char_blocker.py --config configs/config.yaml

    # local smoke test on a tiny synthetic fixture (never the real corpus)
    python scripts/calibrate_char_blocker.py --config /tmp/smoke.yaml --limit-s1 200

NOTE: the full run reads every target row of S2 and S3 and is an HPC job. Do not
run it on a laptop.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import logging
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_name_differences import _trigram_jaccard  # noqa: E402
from src.blocking import (  # noqa: E402
    BLOCKER_EXACT_NAME,
    PAIR_MULTIPLIER,
    ExactNameIndex,
    index_dir_for,
    pack_pairs,
    unpack_pairs,
)
from src.data_loader import (  # noqa: E402
    TARGET_SOURCES,
    describe_environment,
    iter_prepared,
    load_config,
    load_ground_truth,
)
from src.evaluation import (  # noqa: E402
    CandidateEvaluation,
    _contains_sorted,
    split_mask_for,
)
from src.normalization import NAME_KEY  # noqa: E402
from src.utils import (  # noqa: E402
    current_rss_bytes,
    detect_hardware,
    encode_entity_ids,
    ensure_dir,
    fmt_int,
    format_hardware_report,
    human_bytes,
    plan_inflight_window,
    read_json,
    resolve_workers,
    set_seed,
    setup_logging,
    write_json,
)

LOG_NAME = "calibrate_char_blocker"

# Artifact format version. Bump when a layout changes so stale artifacts are
# detected rather than silently misread.
ARTIFACT_VERSION = 1
ARTIFACT_DIRNAME = "_artifacts"

# Grid defaults. Wide enough to show the shape of the curve, small enough that the
# whole thing is one HPC job. These are NOT operating points: choosing one is the
# decision this script exists to inform.
DEFAULT_DF_CAPS = (100, 500, 1000, 5000, 10000)
DEFAULT_RAREST_KS = (1, 2, 3, 5, 10)
DEFAULT_JACCARD_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)

# Explicit, reported guard. A cell above this volume is measured but not
# evaluated, and says so in every output. 0 disables the guard.
DEFAULT_MAX_CANDIDATE_ROWS = 250_000_000

DEFAULT_CHUNK_ROWS = 200_000
DEFAULT_VERIFY_CHUNK_PAIRS = 200_000

# Estimated payload bytes for one verification pair: the S1 key string, the S1
# position, and the target store row. Bounds the in-flight window against RAM.
VERIFY_BYTES_PER_PAIR = 96

# 21 bits per Unicode code point, three code points per trigram: 63 bits, which
# fits an int64 with the sign bit to spare. Unicode caps code points at 0x10FFFF so
# the packing is exact - two different trigrams can never collide, which is why
# this index needs no string verification after a key lookup, unlike a hash-keyed
# one.
_CODE_POINT_BITS = 21
_CODE_POINT_MASK = (1 << _CODE_POINT_BITS) - 1

# The two candidate sets every cell reports. ``char`` is the blocker alone;
# ``char_plus_exact`` adds the shipped exact-name index. Exact is not a recall
# contributor once char is on - identical ``name_norm`` implies identical
# ``name_key`` implies Jaccard 1.0 - but whether *df-capped, rarest-K* retrieval
# actually recovers those pairs is empirical, so this measures it rather than
# assuming it.
SET_CHAR = "char"
SET_CHAR_PLUS_EXACT = "char_plus_exact"
CANDIDATE_SETS = (SET_CHAR, SET_CHAR_PLUS_EXACT)

VOLUME_ALL = "all"
SPLIT_ALL = "all"
SPLIT_VAL = "val"

# Metrics carried through from CandidateEvaluation.compute_metrics(), plus p90
# which it omits. Order is the CSV column order.
REPORTED_METRICS = (
    "blocking_recall_pair",
    "macro_recall_entity",
    "f05_ceiling_from_macro_recall",
    "f05_accept_all_macro",
    "s1_full_recall_rate",
    "s1_partial_recall_rate",
    "n_candidate_pairs",
    "avg_candidates_per_s1",
    "median_candidates_per_s1",
    "p90_candidates_per_s1",
    "p99_candidates_per_s1",
    "max_candidates_per_s1",
    "n_s1_with_zero_candidates",
    "fraction_s1_with_zero_candidates",
    "candidate_precision",
    "reduction_ratio",
    "n_possible_pairs",
    "n_true_pairs",
    "true_pairs_retrieved",
    "n_s1_entities",
    "n_s1_with_true_matches",
)

PER_SOURCE_METRICS = (
    "n_true_pairs",
    "n_candidates",
    "true_pairs_retrieved",
    "blocking_recall_pair",
    "macro_recall_entity",
    "s1_full_recall_rate",
    "s1_partial_recall_rate",
    "candidate_precision",
)

# Phase 0.2-0.4's measured rarest-token blocker, carried in the report as a labelled
# ANALYTIC REFERENCE ONLY. The token blocker is not implemented here, is not enabled
# in config, and is not run by this script; these numbers exist so the char curve can
# be read against the alternative Phase 0 already priced.
TOKEN_ANALYTIC_REFERENCE = {
    "measured_by": "scripts/analyze_blocking_statistics.py (Phase 0.2-0.4)",
    "measured_here": False,
    "reference_cap": 1000,
    "pct_of_true_pairs": 42.3905,
    "pct_of_entities_with_pairs": 47.1908,
    "estimated_candidate_rows": 209_762_533,
    "structural_zero_entities": 1_155_348,
    "structural_zero_pct": 52.3535,
    "note": (
        "Analytic reference from Phase 0, NOT measured by this script. The token "
        "blocker is not enabled and not run here."
    ),
}

_EMPTY_INT64 = np.empty(0, dtype=np.int64)
_EMPTY_UINT8 = np.empty(0, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Trigram codec
# ---------------------------------------------------------------------------
# Everything downstream keys on these codes, so the encoding has to be an exact
# identity rather than a hash. It is: a trigram maps to its three code points packed
# into 63 bits, a bijection, so a key match is a trigram match.
#
# The packing is also what makes this fast. Phase 0.1's
# ``{a[i:i+3] for i in range(len(a) - 2)}`` costs ~10us per string in pure python and
# is the most-repeated operation in the pipeline; the code-point form is a
# vectorized shift-and-or over the whole name at once.
def trigram_codes(text: str) -> np.ndarray:
    """Distinct character-trigram codes of ``text``, sorted ascending.

    Mirrors Phase 0.1's ``set(a[i:i+3] for i in range(len(a) - 2))`` exactly: the
    same sliding window over code points and the same deduplication, since the
    signal is a set overlap. A name shorter than three code points yields no
    trigrams, matching the reference having nothing to intersect.

    Sorted output is what makes selection deterministic: python's ``set`` iteration
    order over strings varies with PYTHONHASHSEED, so an unsorted set would make the
    index unreproducible across runs.
    """
    points = np.frombuffer(text.encode("utf-32-le"), dtype="<u4")
    if points.size < 3:
        return _EMPTY_INT64
    wide = points.astype(np.int64)
    return np.unique((wide[:-2] << 42) | (wide[1:-1] << 21) | wide[2:])


def decode_trigram_code(code: int) -> str:
    """Inverse of :func:`trigram_codes` for one code, for reporting."""
    return (
        chr((int(code) >> 42) & _CODE_POINT_MASK)
        + chr((int(code) >> 21) & _CODE_POINT_MASK)
        + chr(int(code) & _CODE_POINT_MASK)
    )


def trigram_codes_for_list(texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Codes for many strings at once, plus a flat owner index.

    Returns ``(codes, owners)`` with ``owners[i]`` the row that produced
    ``codes[i]``. One flat pair of arrays beats a list of per-row arrays when the
    caller is going to concatenate them anyway.
    """
    parts: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for index, text in enumerate(texts):
        codes = trigram_codes(text)
        if codes.size:
            parts.append(codes)
            owners.append(np.full(codes.size, index, dtype=np.int64))
    if not parts:
        return _EMPTY_INT64, _EMPTY_INT64
    return np.concatenate(parts), np.concatenate(owners)


def rank_within_runs(keys: np.ndarray) -> np.ndarray:
    """Position of each element within its run of equal adjacent values.

    ``keys`` must be sorted, so equal values are contiguous. This is what turns an
    entity's df-ordered trigram list into ranks 0, 1, 2, ... - and therefore what
    makes a ``rank < K`` filter a prefix of each key's posting list.
    """
    if keys.size == 0:
        return _EMPTY_INT64
    starts = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    spans = np.diff(np.append(starts, keys.size))
    return np.arange(keys.size, dtype=np.int64) - np.repeat(starts, spans)


def csr_offsets(row_of_entry: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row entry counts and CSR offsets for a row-sorted entry array."""
    counts = np.bincount(row_of_entry, minlength=n_rows) if row_of_entry.size else np.zeros(n_rows, dtype=np.int64)
    if counts.size > n_rows:
        counts = counts[:n_rows]
    return counts, np.concatenate(([0], np.cumsum(counts)))


# ---------------------------------------------------------------------------
# Corpus-relative trigram document frequency
# ---------------------------------------------------------------------------
class _TrigramDf:
    """Trigram code -> document frequency, over one target source's corpus.

    Corpus-relative by construction: counted from the prepared target table of this
    run. ``codes`` is sorted, which turns a lookup into one ``searchsorted`` over the
    whole query array.
    """

    __slots__ = ("codes", "values")

    def __init__(self, codes: np.ndarray, values: np.ndarray) -> None:
        self.codes = codes
        self.values = values

    def __len__(self) -> int:
        return len(self.codes)

    def lookup(self, codes: np.ndarray) -> np.ndarray:
        """Document frequency of each code; ``0`` for a trigram never seen."""
        if codes.size == 0 or self.codes.size == 0:
            return np.zeros(codes.size, dtype=np.int64)
        positions = np.searchsorted(self.codes, codes).astype(np.int64)
        np.clip(positions, 0, len(self.codes) - 1, out=positions)
        return np.where(self.codes[positions] == codes, self.values[positions], 0)

    def cap_survivors(self, cap: int) -> int:
        return int(np.count_nonzero(self.values <= cap))

    def describe(self, caps: Sequence[int] = ()) -> dict:
        if len(self.values) == 0:
            return {"n_distinct_trigrams": 0}
        return {
            "n_distinct_trigrams": int(len(self.codes)),
            "total_occurrences": int(self.values.sum()),
            "max_df": int(self.values.max()),
            "median_df": float(np.median(self.values)),
            "survivors_by_cap": {str(cap): self.cap_survivors(cap) for cap in caps},
        }

    def top(self, limit: int) -> list[tuple[str, int]]:
        """The ``limit`` most frequent trigrams as (text, df). Reporting only."""
        if len(self.values) == 0:
            return []
        order = np.argsort(self.values)[::-1][:limit]
        return [(decode_trigram_code(int(self.codes[i])), int(self.values[i])) for i in order]

    def save(self, directory: Path) -> None:
        ensure_dir(directory)
        np.save(directory / "df_codes.npy", self.codes)
        np.save(directory / "df_values.npy", self.values)

    @classmethod
    def load(cls, directory: Path) -> "_TrigramDf":
        return cls(
            codes=np.load(directory / "df_codes.npy"), values=np.load(directory / "df_values.npy")
        )


def count_trigram_df(
    chunks: Iterable[pd.DataFrame], name_key_field: str, log: logging.Logger, label: str
) -> _TrigramDf:
    """Count trigram document frequency over a streamed table.

    Per-chunk dedup, then one global merge: each chunk contributes its distinct codes
    with counts, and the pieces are concatenated, sorted once and folded with
    ``add.reduceat``. Deduplicating per chunk keeps the merge proportional to the
    distinct trigram count (a few million) rather than to the ~185M total
    occurrences.
    """
    started = time.time()
    rows = 0
    parts_codes: list[np.ndarray] = []
    parts_counts: list[np.ndarray] = []

    for frame in chunks:
        texts = frame[name_key_field].to_numpy(dtype=object)
        rows += len(texts)
        codes, _ = trigram_codes_for_list(texts)
        if codes.size:
            unique, counts = np.unique(codes, return_counts=True)
            parts_codes.append(unique)
            parts_counts.append(counts.astype(np.int64))
        log.info("  %s: %s rows scanned for trigram df", label, fmt_int(rows))

    if not parts_codes:
        log.warning("%s: no trigrams found in %s rows", label, fmt_int(rows))
        return _TrigramDf(_EMPTY_INT64, _EMPTY_INT64)

    all_codes = np.concatenate(parts_codes)
    all_counts = np.concatenate(parts_counts)
    del parts_codes, parts_counts
    gc.collect()

    order = np.argsort(all_codes, kind="stable")
    all_codes, all_counts = all_codes[order], all_counts[order]
    del order
    gc.collect()

    starts = np.flatnonzero(np.concatenate(([True], all_codes[1:] != all_codes[:-1])))
    table = _TrigramDf(all_codes[starts], np.add.reduceat(all_counts, starts))
    log.info(
        "%s: %s distinct trigrams over %s rows in %.1f s",
        label,
        fmt_int(len(table)),
        fmt_int(rows),
        time.time() - started,
    )
    return table


# ---------------------------------------------------------------------------
# Name-key store
# ---------------------------------------------------------------------------
class _NameKeyStore:
    """A table's ``name_key`` values as a compact blob with a lookup structure.

    Verification has to call the Phase 0.1 reference function on real strings, so the
    target side of every retrieved pair needs its ``name_key`` by entity code, and
    the S1 side needs it by ground-truth position. Holding millions of python strings
    costs far more than the index; a lookup array plus one UTF-8 blob costs ~180MB
    per table and decodes lazily.

    ``row_for_code`` returns a **store row**, and ``key_at_row`` slices the blob with
    it. The blob stays in table order, which avoids permuting millions of
    variable-length strings at build time. For the S1 store the store row is the
    prepared-table row while callers hold ground-truth positions, so
    ``position_to_row`` bridges the two - with ``-1`` for a position the ground truth
    does not know. It is ``None`` (identity) for the target stores, where the two
    coincide.
    """

    __slots__ = ("lookup_codes", "lookup_rows", "row_offsets", "blob", "position_to_row")

    def __init__(
        self,
        lookup_codes: np.ndarray,
        lookup_rows: np.ndarray,
        row_offsets: np.ndarray,
        blob: bytes,
        position_to_row: Optional[np.ndarray] = None,
    ) -> None:
        self.lookup_codes = lookup_codes
        self.lookup_rows = lookup_rows
        self.row_offsets = row_offsets
        self.blob = blob
        self.position_to_row = position_to_row

    @property
    def n_rows(self) -> int:
        return len(self.row_offsets) - 1

    @property
    def n_positions(self) -> int:
        return self.n_rows if self.position_to_row is None else len(self.position_to_row)

    def row_for_code(self, codes: np.ndarray) -> np.ndarray:
        """Store row for each entity code; ``-1`` when the code is absent."""
        if codes.size == 0:
            return _EMPTY_INT64
        if self.lookup_codes.size == 0:
            return np.full(codes.size, -1, dtype=np.int64)
        positions = np.searchsorted(self.lookup_codes, codes).astype(np.int64)
        np.clip(positions, 0, len(self.lookup_codes) - 1, out=positions)
        return np.where(
            self.lookup_codes[positions] == codes, self.lookup_rows[positions], -1
        )

    def row_for_position(self, position: int) -> int:
        """Store row for a caller-space position (identity if unmapped)."""
        if self.position_to_row is None:
            return int(position)
        return int(self.position_to_row[position])

    def key_at_row(self, row: int) -> str:
        """Decode the ``name_key`` stored at a store row."""
        return self.blob[int(self.row_offsets[row]) : int(self.row_offsets[row + 1])].decode("utf-8")

    def key_at_position(self, position: int) -> str:
        """Decode the ``name_key`` at a caller-space position; ``""`` if unknown.

        The empty string is the right miss value rather than an error: it has no
        trigrams, so the Phase 0.1 reference scores it 0.0 against anything, which is
        exactly right for a pair that cannot be verified.
        """
        row = self.row_for_position(position)
        return "" if row < 0 else self.key_at_row(row)

    def valid_positions(self) -> np.ndarray:
        """Positions that map to a stored row, in ascending order."""
        if self.position_to_row is None:
            return np.arange(self.n_positions, dtype=np.int64)
        return np.flatnonzero(self.position_to_row >= 0).astype(np.int64)

    def memory_bytes(self) -> int:
        extra = 0 if self.position_to_row is None else self.position_to_row.nbytes
        return self.lookup_codes.nbytes + self.lookup_rows.nbytes + len(self.blob) + extra

    def describe(self) -> dict:
        return {
            "n_rows": int(self.n_rows),
            "n_positions": int(self.n_positions),
            "blob": human_bytes(len(self.blob)),
            "memory": human_bytes(self.memory_bytes()),
        }

    def save(self, directory: Path) -> None:
        ensure_dir(directory)
        np.save(directory / "nk_lookup_codes.npy", self.lookup_codes)
        np.save(directory / "nk_lookup_rows.npy", self.lookup_rows)
        np.save(directory / "nk_row_offsets.npy", self.row_offsets)
        if self.position_to_row is not None:
            np.save(directory / "nk_position_to_row.npy", self.position_to_row)
        with open(directory / "nk_blob.bin", "wb") as handle:
            handle.write(self.blob)

    @classmethod
    def load(cls, directory: Path) -> "_NameKeyStore":
        with open(directory / "nk_blob.bin", "rb") as handle:
            blob = handle.read()
        alias_path = directory / "nk_position_to_row.npy"
        return cls(
            lookup_codes=np.load(directory / "nk_lookup_codes.npy"),
            lookup_rows=np.load(directory / "nk_lookup_rows.npy"),
            row_offsets=np.load(directory / "nk_row_offsets.npy"),
            blob=blob,
            position_to_row=np.load(alias_path) if alias_path.is_file() else None,
        )


def build_name_key_store(
    chunks: Iterable[pd.DataFrame],
    entity_column: str,
    name_key_field: str,
    log: logging.Logger,
    label: str,
    ground_truth=None,
    n_entities: int = 0,
) -> _NameKeyStore:
    """Stream a table once into a compact ``entity_code -> name_key`` store.

    When ``ground_truth`` is given the store is additionally addressable by
    ground-truth position, which is the coordinate system the evaluator's
    accumulators live in. The blob layout does not change: only an int64
    ``position_to_row`` bridge is added, so nothing variable-length is ever
    permuted.
    """
    started = time.time()
    code_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    blob = bytearray()
    offsets = [0]
    rows = 0

    for frame in chunks:
        code_parts.append(encode_entity_ids(frame[entity_column]))
        if ground_truth is not None:
            position_parts.append(ground_truth.positions_of(frame[entity_column]))
        for text in frame[name_key_field].to_numpy(dtype=object):
            blob.extend(text.encode("utf-8"))
            offsets.append(len(blob))
        rows += len(frame)

    if not code_parts:
        return _NameKeyStore(_EMPTY_INT64, _EMPTY_INT64, np.zeros(1, dtype=np.int64), b"")

    all_codes = np.concatenate(code_parts)
    order = np.argsort(all_codes, kind="stable")

    position_to_row: Optional[np.ndarray] = None
    if ground_truth is not None:
        positions = np.concatenate(position_parts)
        position_to_row = np.full(int(n_entities), -1, dtype=np.int64)
        known = positions >= 0
        position_to_row[positions[known]] = np.flatnonzero(known).astype(np.int64)

    store = _NameKeyStore(
        lookup_codes=all_codes[order],
        lookup_rows=order.astype(np.int64),
        row_offsets=np.asarray(offsets, dtype=np.int64),
        blob=bytes(blob),
        position_to_row=position_to_row,
    )
    log.info(
        "%s: name-key store for %s rows in %.1f s (%s)",
        label,
        fmt_int(rows),
        time.time() - started,
        store.describe(),
    )
    return store


# ---------------------------------------------------------------------------
# Trigram inverted index (multi-key per entity)
# ---------------------------------------------------------------------------
class _TrigramIndex:
    """Inverted index from a trigram code to the entities carrying it.

    Same CSR layout as the shipped :class:`~src.blocking.ExactNameIndex` - sorted
    keys, offsets into a flat posting array - with three differences, all required by
    a character-ngram blocker:

    * **multi-key.** One entity contributes many keys. The shipped index is
      one-key-per-entity, which is why the token and char blockers could not reuse it
      as-is.
    * **rank.** Each posting carries its trigram's rarity rank for its entity, so the
      whole rarest-K grid is a filter over one built index instead of one build per K.
    * **exact keys.** Keys are 63-bit trigram codes, not blake2b hashes: the encoding
      is injective, so there is no collision to verify away and a lookup needs no
      string comparison.

    ``key_df`` rides along so the index is self-contained: the df cap is a mask on
    this array, with no second structure to consult.
    """

    __slots__ = ("source", "keys", "key_df", "postings_offsets", "postings", "ranks", "_rank_cum")

    def __init__(
        self,
        source: str,
        keys: np.ndarray,
        key_df: np.ndarray,
        postings_offsets: np.ndarray,
        postings: np.ndarray,
        ranks: np.ndarray,
    ) -> None:
        self.source = source
        self.keys = keys
        self.key_df = key_df
        self.postings_offsets = postings_offsets
        self.postings = postings
        self.ranks = ranks
        self._rank_cum: Optional[np.ndarray] = None

    @property
    def n_keys(self) -> int:
        return len(self.keys)

    @property
    def n_postings(self) -> int:
        return len(self.postings)

    @property
    def max_rank(self) -> int:
        return int(self.ranks.max()) + 1 if self.ranks.size else 0

    def counts_per_key(self) -> np.ndarray:
        return np.diff(self.postings_offsets)

    def memory_bytes(self) -> int:
        return (
            self.keys.nbytes
            + self.key_df.nbytes
            + self.postings_offsets.nbytes
            + self.postings.nbytes
            + self.ranks.nbytes
        )

    def describe(self) -> dict:
        counts = self.counts_per_key()
        return {
            "source": self.source,
            "n_keys": int(self.n_keys),
            "n_postings": int(self.n_postings),
            "max_rank": self.max_rank,
            "avg_postings_per_key": round(float(counts.mean()), 3) if counts.size else 0.0,
            "max_postings_per_key": int(counts.max()) if counts.size else 0,
            "memory": human_bytes(self.memory_bytes()),
        }

    # -- query --------------------------------------------------------------
    def positions_for_codes(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Binary search for trigram codes; returns ``(position, found)``.

        The int64-code twin of ``ExactNameIndex._positions_for_hashes``, minus the
        string re-verification a hash-based key needs and an injective encoding does
        not.
        """
        if codes.size == 0:
            return _EMPTY_INT64, np.zeros(0, dtype=bool)
        if self.n_keys == 0:
            return np.full(codes.size, -1, dtype=np.int64), np.zeros(codes.size, dtype=bool)
        positions = np.searchsorted(self.keys, codes).astype(np.int64)
        clipped = np.minimum(positions, self.n_keys - 1)
        found = (positions < self.n_keys) & (self.keys[clipped] == codes)
        return np.where(found, clipped, -1), found

    def lookup_many(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batched lookup returning ``(positions, counts)``, ``-1`` on a miss."""
        positions, found = self.positions_for_codes(codes)
        counts = np.zeros(codes.size, dtype=np.int64)
        if found.any():
            index = positions[found]
            counts[found] = self.postings_offsets[index + 1] - self.postings_offsets[index]
        return positions, counts

    def expand(self, positions: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """CSR expand into ``(owner_index, entity_codes)``.

        Deliberately identical to ``ExactNameIndex.expand`` - the same repeat/arange
        arithmetic producing the same ordering - so candidate ordering downstream
        matches the shipped generator.
        """
        total = int(counts.sum())
        if total == 0:
            return _EMPTY_INT64, _EMPTY_INT64
        valid = positions >= 0
        starts_flat = np.repeat(self.postings_offsets[positions[valid]], counts[valid])
        group_base = np.repeat(np.cumsum(counts[valid]) - counts[valid], counts[valid])
        within_group = np.arange(total, dtype=np.int64) - group_base
        return (
            np.repeat(np.flatnonzero(valid), counts[valid]),
            self.postings[starts_flat + within_group],
        )

    # -- grid filtering -----------------------------------------------------
    def rank_cumulative(self) -> np.ndarray:
        """``cum[p, r]`` = postings under key ``p`` whose rank is ``<= r``.

        One ``add.reduceat`` per rank value rather than a ``bincount`` over
        ``n_keys * max_rank`` bins: the bincount materializes a key-of-posting array
        as large as the posting array, while this holds one posting-sized temporary at
        a time. Cached, because every cell needs it.
        """
        if self._rank_cum is None:
            starts = self.postings_offsets[:-1]
            hist = np.zeros((self.n_keys, max(self.max_rank, 1)), dtype=np.int64)
            for rank in range(self.max_rank):
                hist[:, rank] = np.add.reduceat((self.ranks == rank).astype(np.int64), starts)
            self._rank_cum = np.cumsum(hist, axis=1)
        return self._rank_cum

    def kept_counts_per_key(self, cap: int, rarest_k: int) -> np.ndarray:
        """Postings a ``(df_cap, rarest_k)`` cell would return, per key.

        Zero for a key above the cap. This is the entire volume curve, per cell: the
        postings a surviving query key retrieves, with nothing expanded.
        """
        cumulative = self.rank_cumulative()
        column = min(max(int(rarest_k), 1), cumulative.shape[1]) - 1
        kept = cumulative[:, column].copy()
        kept[self.key_df > cap] = 0
        return kept

    def filtered(self, cap: int, rarest_k: int) -> "_TrigramIndex":
        """The sub-index for one ``(df_cap, rarest_k)`` cell.

        Because an entity's keys were appended in rank-ascending order, the postings a
        rank filter keeps form a **prefix** of each key's posting list, so the
        survivors are one boolean mask and one cumulative sum - no per-posting sort,
        no second pass over the corpus.

        Keys that lose every posting are dropped rather than kept with an empty list.
        A tighter K can empty a key that the shared index carried (no entity ranked it
        inside the new cutoff), and a CSR index with empty keys is not what building at
        that cell would produce - the equivalence with a direct build is the whole
        justification for sharing one index, so it has to hold exactly.
        """
        kept_counts = self.kept_counts_per_key(cap, rarest_k)
        counts = self.counts_per_key()
        mask = (self.ranks < rarest_k) & np.repeat(self.key_df <= cap, counts)
        assert int(kept_counts.sum()) == int(mask.sum()), (
            "filtered posting count disagrees with the per-key prefix sums; postings are no "
            "longer rank-ordered within a key"
        )
        non_empty = kept_counts > 0
        return _TrigramIndex(
            source=self.source,
            keys=self.keys[non_empty],
            key_df=self.key_df[non_empty],
            postings_offsets=np.concatenate(([0], np.cumsum(kept_counts[non_empty]))),
            postings=self.postings[mask],
            ranks=self.ranks[mask],
        )

    def top_keys(self, limit: int) -> list[tuple[str, int, int]]:
        """Most-populated keys as (text, df, postings). Evidence for the cap."""
        counts = self.counts_per_key()
        if counts.size == 0:
            return []
        order = np.argsort(counts)[::-1][:limit]
        return [
            (decode_trigram_code(int(self.keys[i])), int(self.key_df[i]), int(counts[i]))
            for i in order
        ]

    # -- persistence --------------------------------------------------------
    def save(self, directory: Path) -> None:
        ensure_dir(directory)
        np.save(directory / "idx_keys.npy", self.keys)
        np.save(directory / "idx_key_df.npy", self.key_df)
        np.save(directory / "idx_postings_offsets.npy", self.postings_offsets)
        np.save(directory / "idx_postings.npy", self.postings)
        np.save(directory / "idx_ranks.npy", self.ranks)

    @classmethod
    def load(cls, directory: Path, source: str) -> "_TrigramIndex":
        return cls(
            source=source,
            keys=np.load(directory / "idx_keys.npy"),
            key_df=np.load(directory / "idx_key_df.npy"),
            postings_offsets=np.load(directory / "idx_postings_offsets.npy"),
            postings=np.load(directory / "idx_postings.npy"),
            ranks=np.load(directory / "idx_ranks.npy"),
        )


def empty_index(source: str) -> _TrigramIndex:
    """An index with no keys, for the degenerate all-capped-out case."""
    return _TrigramIndex(
        source=source,
        keys=_EMPTY_INT64,
        key_df=np.empty(0, dtype=np.int32),
        postings_offsets=np.zeros(1, dtype=np.int64),
        postings=_EMPTY_INT64,
        ranks=_EMPTY_UINT8,
    )


def build_trigram_index(
    chunks: Iterable[pd.DataFrame],
    df: _TrigramDf,
    entity_column: str,
    name_key_field: str,
    max_df_cap: int,
    max_rank: int,
    source: str,
    log: logging.Logger,
    label: str,
) -> _TrigramIndex:
    """Build the shared trigram index for one target source.

    Two properties the rest of the pipeline depends on, both established here:

    * every entity's surviving trigrams are ordered by **df ascending** and truncated
      at ``max_rank``, which is what makes a tighter ``(cap, K)`` cell a filter of
      this index rather than a rebuild;
    * postings are grouped by key and, within a key, ordered by rank ascending, which
      is what makes the rank filter a prefix and the filtered offsets a cumulative
      sum.
    """
    started = time.time()
    rows = 0
    key_parts: list[np.ndarray] = []
    entity_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    indexed_entities = 0
    structural_zero = 0

    for frame in chunks:
        texts = frame[name_key_field].to_numpy(dtype=object)
        entity_codes = encode_entity_ids(frame[entity_column])
        all_codes, owners = trigram_codes_for_list(texts)
        rows += len(texts)

        if all_codes.size == 0:
            structural_zero += len(texts)
            continue

        dfs = df.lookup(all_codes)
        keep = (dfs > 0) & (dfs <= max_df_cap)
        if not keep.any():
            structural_zero += len(texts)
            continue
        codes, row_of, row_dfs = all_codes[keep], owners[keep], dfs[keep]

        # Row first, then df ascending, then code ascending. Row-major is what makes
        # rank_within_runs valid; the df/code tail is a total order, so the selected
        # keys and their ranks are reproducible run to run.
        order = np.lexsort((codes, row_dfs, row_of))
        codes, row_of, row_dfs = codes[order], row_of[order], row_dfs[order]

        keep_rank = rank_within_runs(row_of) < max_rank
        if not keep_rank.any():
            structural_zero += len(texts)
            continue
        survivors = np.unique(row_of[keep_rank])
        indexed_entities += int(survivors.size)
        structural_zero += len(texts) - int(survivors.size)

        key_parts.append(codes[keep_rank])
        entity_parts.append(entity_codes[row_of[keep_rank]])
        rank_parts.append(rank_within_runs(row_of)[keep_rank].astype(np.uint8))
        log.info("  %s: %s rows indexed for char keys", label, fmt_int(rows))

    if not key_parts:
        log.warning(
            "%s: no indexable trigrams; the df cap of %s removed every key", label, max_df_cap
        )
        return empty_index(source)

    all_keys = np.concatenate(key_parts)
    all_entities = np.concatenate(entity_parts)
    all_ranks = np.concatenate(rank_parts)
    del key_parts, entity_parts, rank_parts
    gc.collect()

    # Key-major, then rank ascending within a key. lexsort is stable, so equal
    # (key, rank) pairs keep emission order and the build is reproducible.
    order = np.lexsort((all_ranks, all_keys))
    all_keys, all_entities, all_ranks = all_keys[order], all_entities[order], all_ranks[order]
    del order
    gc.collect()

    starts = np.flatnonzero(np.concatenate(([True], all_keys[1:] != all_keys[:-1])))
    index = _TrigramIndex(
        source=source,
        keys=all_keys[starts],
        key_df=df.lookup(all_keys[starts]).astype(np.int32),
        postings_offsets=np.concatenate((starts, [len(all_keys)])),
        postings=all_entities,
        ranks=all_ranks,
    )
    log.info(
        "%s: trigram index built in %.1f s (%s); %s entities indexed, %s rows contributed no "
        "surviving key (%.2f%% of rows)",
        label,
        time.time() - started,
        index.describe(),
        fmt_int(indexed_entities),
        fmt_int(structural_zero),
        100.0 * structural_zero / max(1, rows),
    )
    return index


# ---------------------------------------------------------------------------
# S1-side key selection, in ground-truth position space
# ---------------------------------------------------------------------------
class _S1Selection:
    """Every S1 entity's trigrams, ranked, at the loosest grid setting.

    Indexed by **ground-truth position**, because the evaluator's accumulators are one
    slot per ground-truth entity. Positions absent from the ground truth have no
    codes, so they contribute nothing anywhere.

    Computed once and reused by every cell. The per-row work - code the name, look up
    df, order by df, truncate - is identical for all cells; only the ``df <= cap``
    filter and the ``rank < K`` cut differ, and both are vectorized gathers over this
    structure. Without it the S1 side would be re-coded once per cell, which at 25
    cells would dominate the run.
    """

    __slots__ = ("codes", "ranks", "dfs", "offsets", "n_rows")

    def __init__(
        self,
        codes: np.ndarray,
        ranks: np.ndarray,
        dfs: np.ndarray,
        offsets: np.ndarray,
        n_rows: int,
    ) -> None:
        self.codes = codes
        self.ranks = ranks
        self.dfs = dfs
        self.offsets = offsets
        self.n_rows = n_rows

    def slice_full(
        self, start_row: int, stop_row: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unfiltered ``(codes, positions, ranks, dfs)`` for rows ``[start, stop)``.

        Unfiltered on purpose: one slice serves every cell, each applying its own
        ``(cap, K)`` predicate to the returned arrays.
        """
        begin = int(self.offsets[start_row])
        end = int(self.offsets[stop_row])
        if begin == end:
            return _EMPTY_INT64, _EMPTY_INT64, _EMPTY_UINT8, _EMPTY_INT64
        lengths = np.diff(self.offsets[start_row : stop_row + 1])
        positions = np.repeat(np.arange(start_row, stop_row, dtype=np.int64), lengths)
        return self.codes[begin:end], positions, self.ranks[begin:end], self.dfs[begin:end]

    @staticmethod
    def cell_filter(
        ranks: np.ndarray, dfs: np.ndarray, cap: int, rarest_k: int
    ) -> np.ndarray:
        """Boolean mask selecting the entries one cell's rules keep."""
        return (dfs <= cap) & (ranks < rarest_k)

    def memory_bytes(self) -> int:
        return self.codes.nbytes + self.ranks.nbytes + self.dfs.nbytes + self.offsets.nbytes

    def describe(self) -> dict:
        return {
            "n_rows": int(self.n_rows),
            "n_entries": int(len(self.codes)),
            "entries_per_row": round(float(len(self.codes)) / max(1, self.n_rows), 3),
            "memory": human_bytes(self.memory_bytes()),
        }

    def save(self, directory: Path) -> None:
        ensure_dir(directory)
        np.save(directory / "s1_codes.npy", self.codes)
        np.save(directory / "s1_ranks.npy", self.ranks)
        np.save(directory / "s1_dfs.npy", self.dfs)
        np.save(directory / "s1_offsets.npy", self.offsets)
        write_json(directory / "s1_meta.json", {"n_rows": int(self.n_rows)})

    @classmethod
    def load(cls, directory: Path) -> "_S1Selection":
        meta = read_json(directory / "s1_meta.json")
        return cls(
            codes=np.load(directory / "s1_codes.npy"),
            ranks=np.load(directory / "s1_ranks.npy"),
            dfs=np.load(directory / "s1_dfs.npy"),
            offsets=np.load(directory / "s1_offsets.npy"),
            n_rows=int(meta["n_rows"]),
        )


def build_s1_selection(
    chunks: Iterable[pd.DataFrame],
    df: _TrigramDf,
    ground_truth,
    entity_column: str,
    name_key_field: str,
    max_df_cap: int,
    max_rank: int,
    n_entities: int,
    log: logging.Logger,
    label: str,
) -> tuple[_S1Selection, dict]:
    """Rank every S1 entity's trigrams once, at the loosest grid setting.

    The ordering is the index's ordering read from the other side - position first,
    then df ascending, then code ascending - so a rank means the same thing on both
    sides of a lookup. That agreement is what makes the cell filter exact rather than
    approximate.
    """
    started = time.time()
    code_parts: list[np.ndarray] = []
    df_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    rows = 0
    unknown = 0
    # Trigram-level drop reasons, counted separately because a row that keeps no key
    # can have lost them three different ways and the fixes are different.
    dropped_absent = 0  # df == 0: the trigram never occurs in this target source
    dropped_above_cap = 0  # df > cap: too common here to be a key
    dropped_beyond_k = 0  # df fine, but the entity has rarer trigrams to spend K on
    too_short = 0

    for frame in chunks:
        texts = frame[name_key_field].to_numpy(dtype=object)
        rows += len(frame)

        positions = ground_truth.positions_of(frame[entity_column])
        known = positions >= 0
        unknown += int((~known).sum())
        if not known.any():
            continue
        positions, texts = positions[known], texts[known]

        all_codes, owners = trigram_codes_for_list(texts)
        if all_codes.size == 0:
            too_short += len(texts)
            continue
        short_here = len(texts) - int(np.unique(owners).size)
        too_short += short_here

        entry_positions = positions[owners]
        dfs = df.lookup(all_codes)
        absent = dfs == 0
        above_cap = dfs > max_df_cap
        dropped_absent += int(absent.sum())
        dropped_above_cap += int(above_cap.sum())

        keep = ~absent & ~above_cap
        if not keep.any():
            continue
        codes, entry_positions, row_dfs = all_codes[keep], entry_positions[keep], dfs[keep]

        order = np.lexsort((codes, row_dfs, entry_positions))
        codes, entry_positions, row_dfs = codes[order], entry_positions[order], row_dfs[order]

        keep_rank = rank_within_runs(entry_positions) < max_rank
        dropped_beyond_k += int((~keep_rank).sum())
        if not keep_rank.any():
            continue

        code_parts.append(codes[keep_rank])
        df_parts.append(row_dfs[keep_rank])
        rank_parts.append(rank_within_runs(entry_positions)[keep_rank].astype(np.uint8))
        position_parts.append(entry_positions[keep_rank])
        log.info("  %s: %s S1 rows coded", label, fmt_int(rows))

    if code_parts:
        codes = np.concatenate(code_parts)
        dfs = np.concatenate(df_parts)
        ranks = np.concatenate(rank_parts)
        positions = np.concatenate(position_parts)
        del code_parts, df_parts, rank_parts, position_parts
        gc.collect()

        # Chunks arrive in prepared-table order, so the concatenation is not yet
        # position-sorted and the CSR below would be invalid. A stable sort restores
        # it while preserving each row's df order, hence its ranks.
        order = np.argsort(positions, kind="stable")
        codes, dfs, ranks, positions = codes[order], dfs[order], ranks[order], positions[order]
        del order
        gc.collect()
        _, offsets = csr_offsets(positions, n_entities)
    else:
        codes, dfs, ranks = _EMPTY_INT64, _EMPTY_INT64, _EMPTY_UINT8
        offsets = np.zeros(n_entities + 1, dtype=np.int64)

    selection = _S1Selection(
        codes=codes, ranks=ranks, dfs=dfs, offsets=offsets, n_rows=n_entities
    )
    per_row = np.diff(offsets)
    breakdown = {
        "n_s1_rows_read": int(rows),
        "n_s1_positions": int(n_entities),
        "s1_unknown_to_ground_truth": int(unknown),
        "s1_name_too_short_for_any_trigram": int(too_short),
        "s1_with_no_key_in_this_source": int(n_entities - int(np.count_nonzero(per_row))),
        "trigrams_dropped_absent_from_this_source": int(dropped_absent),
        "trigrams_dropped_above_df_cap": int(dropped_above_cap),
        "trigrams_dropped_beyond_rarest_k": int(dropped_beyond_k),
        "entries_per_row": round(float(codes.size) / max(1, n_entities), 3),
    }
    log.info(
        "%s: S1 selection built in %.1f s (%s); %s",
        label,
        time.time() - started,
        selection.describe(),
        breakdown,
    )
    return selection, breakdown


# ---------------------------------------------------------------------------
# Verification (bit-identical to Phase 0.1)
# ---------------------------------------------------------------------------
_WORKER_STORES: dict[str, _NameKeyStore] = {}
_WORKER_DIRS: dict[str, str] = {}


def _verify_worker_init(store_dirs: dict[str, str]) -> None:
    """Record where the target name-key stores live; load them lazily.

    Passed as paths rather than values: each store is hundreds of MB of blob, and
    pickling them to every worker would cost more than the verification they support.
    Lazy loading also means a worker that only ever sees source2 never pays for
    source3.
    """
    global _WORKER_DIRS, _WORKER_STORES
    _WORKER_DIRS = dict(store_dirs)
    _WORKER_STORES = {}


def _worker_store(source: str) -> _NameKeyStore:
    store = _WORKER_STORES.get(source)
    if store is None:
        store = _NameKeyStore.load(Path(_WORKER_DIRS[source]))
        _WORKER_STORES[source] = store
    return store


def _verify_chunk(payload: tuple[str, list[str], np.ndarray]) -> np.ndarray:
    """Exact 3-gram Jaccard for one payload of pairs.

    Calls the Phase 0.1 reference function unchanged, on decoded names, so the
    threshold this script calibrates is the same signal Phase 0 measured - including
    the branch that falls back to bare character sets when either name is shorter than
    three characters. A vectorized rewrite over the packed codes would be faster and
    would also be a *second* definition of the signal.
    """
    source, s1_keys, target_rows = payload
    store = _worker_store(source)
    out = np.zeros(len(s1_keys), dtype=np.float32)
    for index in range(len(s1_keys)):
        row = int(target_rows[index])
        if row < 0:
            continue
        out[index] = _trigram_jaccard(s1_keys[index], store.key_at_row(row))
    return out


def verify_similarities(
    source: str,
    s1_positions: np.ndarray,
    target_rows: np.ndarray,
    key_of,
    *,
    workers: int,
    window: int,
    chunk_pairs: int,
    store_dirs: dict[str, str],
) -> np.ndarray:
    """Phase 0.1 Jaccard for every pair, sharded, consumed in submission order.

    ``key_of(position)`` decodes one S1 name. It is called **per payload**, not once
    for the whole chunk: a chunk can retrieve millions of pairs, and materializing
    millions of python strings at once is the kind of object-heavy structure that
    turns a bounded run into an OOM. Only the int64 position array is held across
    payloads.

    Chunks are consumed strictly in submission order, so the similarity array is a
    deterministic function of the input regardless of worker count.
    """
    total = len(target_rows)
    out = np.zeros(total, dtype=np.float32)
    if total == 0:
        return out

    spans = [(start, min(start + chunk_pairs, total)) for start in range(0, total, chunk_pairs)]

    def payload_for(start: int, stop: int) -> tuple[str, list[str], np.ndarray]:
        return (
            source,
            [key_of(int(position)) for position in s1_positions[start:stop]],
            target_rows[start:stop],
        )

    if workers <= 1:
        _verify_worker_init(store_dirs)
        for start, stop in spans:
            out[start:stop] = _verify_chunk(payload_for(start, stop))
        return out

    pending: deque = deque()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_verify_worker_init, initargs=(store_dirs,)
    ) as pool:
        for start, stop in spans:
            pending.append((start, stop, pool.submit(_verify_chunk, payload_for(start, stop))))
            if len(pending) >= window:
                begin, end, future = pending.popleft()
                out[begin:end] = future.result()
        while pending:
            begin, end, future = pending.popleft()
            out[begin:end] = future.result()
    return out


# ---------------------------------------------------------------------------
# Threshold accumulators
# ---------------------------------------------------------------------------
class _AccumulatorSet:
    """One (candidate set, threshold) combination's counters.

    The same six arrays ``CandidateEvaluation`` keeps, held separately so that ten
    threshold/set combinations can share one evaluator. Ten evaluators would each
    rebuild a ~61MB ``true_pair_codes`` array in ``__init__`` - an avoidable ~550MB
    for a structure that is identical in all ten.
    """

    __slots__ = ("candidate_counts", "hit_counts", "candidate_by_source", "hit_by_source", "n_rows")

    def __init__(self, n_entities: int, source_codes: Sequence[int]) -> None:
        self.candidate_counts = np.zeros(n_entities, dtype=np.int64)
        self.hit_counts = np.zeros(n_entities, dtype=np.int64)
        self.candidate_by_source = {
            code: np.zeros(n_entities, dtype=np.int64) for code in source_codes
        }
        self.hit_by_source = {code: np.zeros(n_entities, dtype=np.int64) for code in source_codes}
        self.n_rows = 0

    def add(self, owners: np.ndarray, target_codes: np.ndarray, true_pair_codes: np.ndarray) -> None:
        """Fold one candidate chunk in, using the shipped truth test.

        ``np.add.at`` rather than fancy-indexed ``+=`` for the same reason the shipped
        evaluator uses it: the buffered form collapses repeated indices, so an entity
        that retrieved several true matches would be credited with one.
        """
        if owners.size == 0:
            return
        is_true = _contains_sorted(true_pair_codes, owners * PAIR_MULTIPLIER + target_codes)

        np.add.at(self.candidate_counts, owners, 1)
        if is_true.any():
            np.add.at(self.hit_counts, owners[is_true], 1)

        source_codes = target_codes // 10**10
        for code, counts in self.candidate_by_source.items():
            mask = source_codes == code
            if not mask.any():
                continue
            np.add.at(counts, owners[mask], 1)
            true_in_source = mask & is_true
            if true_in_source.any():
                np.add.at(self.hit_by_source[code], owners[true_in_source], 1)

        self.n_rows += int(owners.size)

    def memory_bytes(self) -> int:
        return (
            self.candidate_counts.nbytes
            + self.hit_counts.nbytes
            + sum(array.nbytes for array in self.candidate_by_source.values())
            + sum(array.nbytes for array in self.hit_by_source.values())
        )


def bind_accumulators(evaluation: CandidateEvaluation, accumulators: _AccumulatorSet) -> None:
    """Point one evaluator at a set of counters.

    The metric *definitions* stay entirely the shipped ones - ``compute_metrics`` reads
    these attributes and nothing else - while the counters themselves are pooled so the
    ten grid combinations do not each own a copy of the truth structure. That pooling
    is the only reason the private names appear here.
    """
    evaluation._candidate_counts = accumulators.candidate_counts
    evaluation._hit_counts = accumulators.hit_counts
    evaluation._candidate_counts_by_source = accumulators.candidate_by_source
    evaluation._hit_counts_by_source = accumulators.hit_by_source
    evaluation._n_candidate_rows = accumulators.n_rows


def extract_metrics(
    evaluation: CandidateEvaluation, s1_mask: Optional[np.ndarray] = None, split_label: str = SPLIT_ALL
) -> dict:
    """The requested metric subset, plus p90 which ``compute_metrics`` omits."""
    metrics = evaluation.compute_metrics(s1_mask=s1_mask, split_label=split_label)
    counts = evaluation._candidate_counts if s1_mask is None else evaluation._candidate_counts[s1_mask]
    metrics["p90_candidates_per_s1"] = float(np.percentile(counts, 90)) if counts.size else 0.0
    n_entities = metrics.get("n_s1_entities")
    zero = metrics.get("n_s1_with_zero_candidates")
    out = {key: metrics.get(key) for key in REPORTED_METRICS}
    out["per_source"] = {
        prefix: {key: block.get(key) for key in PER_SOURCE_METRICS}
        for prefix, block in (metrics.get("per_source") or {}).items()
    }
    out["reach"] = {
        "n_s1_entities": n_entities,
        "n_s1_with_true_matches": metrics.get("n_s1_with_true_matches"),
        "n_s1_with_candidates": (int(n_entities) - int(zero)) if n_entities is not None and zero is not None else None,
        "n_s1_with_zero_candidates": zero,
        "fraction_s1_with_zero_candidates": metrics.get("fraction_s1_with_zero_candidates"),
    }
    return out


def volume_statistics(per_s1: np.ndarray) -> dict:
    """Summary of a per-S1 candidate volume array."""
    n = len(per_s1)
    if n == 0:
        return {
            "n_candidate_pairs": 0,
            "avg_candidates_per_s1": 0.0,
            "median_candidates_per_s1": 0.0,
            "p90_candidates_per_s1": 0.0,
            "p99_candidates_per_s1": 0.0,
            "max_candidates_per_s1": 0,
            "n_s1_with_zero_candidates": 0,
            "fraction_s1_with_zero_candidates": 0.0,
        }
    zeros = int((per_s1 == 0).sum())
    return {
        "n_candidate_pairs": int(per_s1.sum()),
        "avg_candidates_per_s1": float(per_s1.mean()),
        "median_candidates_per_s1": float(np.median(per_s1)),
        "p90_candidates_per_s1": float(np.percentile(per_s1, 90)),
        "p99_candidates_per_s1": float(np.percentile(per_s1, 99)),
        "max_candidates_per_s1": int(per_s1.max()),
        "n_s1_with_zero_candidates": zeros,
        "fraction_s1_with_zero_candidates": float(zeros / n),
    }


# ---------------------------------------------------------------------------
# Volume sweep: every cell, no expansion
# ---------------------------------------------------------------------------
BOUND_KIND = "upper_bound_from_posting_expansion"
EXACT_KIND = "exact"


def source_scope(source: str) -> str:
    """Per-source scope label, matching the ``S2``/``S3`` keys of ``per_source``.

    The volume rows and the evaluator's ``per_source`` block have to use one
    convention, or a reader joining them by source name would find nothing.
    """
    return "S" + source[-1] if source.startswith("source") and source[-1].isdigit() else source


def sweep_volume(
    indexes: dict[str, _TrigramIndex],
    selections: dict[str, _S1Selection],
    cells: Sequence[tuple[int, int]],
    n_entities: int,
    chunk_rows: int,
    log: logging.Logger,
) -> tuple[list[dict], dict[tuple[int, int], int]]:
    """Upper-bound the candidate volume of every cell in the grid.

    Counts posting-list prefixes rather than expanding them, so a cell that would
    produce 40B candidates costs about what one producing 40 costs. Runs entirely off
    the in-memory S1 selection: no prepared table is re-read, because the selection
    already holds every query key the grid needs.

    This is why the whole grid can be priced before anything is retrieved, and why a
    cell too large to retrieve still reports a number. The price of that reach is
    looseness: a target reached through two of an S1 entity's keys is counted twice
    here, so this overstates the distinct pair count by the key-overlap factor. Rows
    are tagged :data:`BOUND_KIND`, and the same cells' exact counts are reported
    alongside once they are retrieved.
    """
    rows: list[dict] = []
    totals: dict[tuple[int, int], int] = {}
    started = time.time()

    for cap, rarest_k in cells:
        union_per_s1 = np.zeros(n_entities, dtype=np.int64)
        for source, index in indexes.items():
            kept = index.kept_counts_per_key(cap, rarest_k)
            selection = selections[source]
            per_s1 = np.zeros(n_entities, dtype=np.int64)
            for start in range(0, n_entities, chunk_rows):
                stop = min(start + chunk_rows, n_entities)
                codes, positions, ranks, dfs = selection.slice_full(start, stop)
                if codes.size == 0:
                    continue
                keep = selection.cell_filter(ranks, dfs, cap, rarest_k)
                if not keep.any():
                    continue
                key_positions, found = index.positions_for_codes(codes[keep])
                contribution = np.where(found, kept[np.maximum(key_positions, 0)], 0)
                if contribution.any():
                    np.add.at(per_s1, positions[keep], contribution)
            union_per_s1 += per_s1
            rows.append(
                {
                    "df_cap": cap,
                    "rarest_k": rarest_k,
                    "scope": source_scope(source),
                    "kind": BOUND_KIND,
                    **volume_statistics(per_s1),
                }
            )
            del per_s1

        stats = volume_statistics(union_per_s1)
        totals[(cap, rarest_k)] = stats["n_candidate_pairs"]
        rows.append(
            {
                "df_cap": cap,
                "rarest_k": rarest_k,
                "scope": VOLUME_ALL,
                "kind": BOUND_KIND,
                **stats,
            }
        )
        log.info(
            "volume bound (cap=%s, K=%s): <= %s pairs, %.1f avg/S1, %.2f%% of S1 with none%s",
            cap,
            rarest_k,
            fmt_int(stats["n_candidate_pairs"]),
            stats["avg_candidates_per_s1"],
            stats["fraction_s1_with_zero_candidates"] * 100.0,
            "" if stats["n_candidate_pairs"] else " [cell is empty]",
        )
        del union_per_s1

    log.info("volume sweep: %d cells in %.1f s", len(cells), time.time() - started)
    return rows, totals


def exact_volume_rows(
    cell: dict, entry: dict, accumulators: "_AccumulatorSet", n_entities: int
) -> list[dict]:
    """Exact distinct-pair volume for one (cell, set, threshold), per scope.

    Read straight off the accumulator the evaluator's ``n_candidate_pairs`` comes
    from, so these rows and the grid's metrics are the same measurement rather than
    two that ought to agree.
    """
    rows = [
        {
            "df_cap": cell["df_cap"],
            "rarest_k": cell["rarest_k"],
            "scope": VOLUME_ALL,
            "kind": EXACT_KIND,
            "set": entry["set"],
            "jaccard_threshold": entry["threshold"],
            **volume_statistics(accumulators.candidate_counts),
        }
    ]
    for code, counts in accumulators.candidate_by_source.items():
        rows.append(
            {
                "df_cap": cell["df_cap"],
                "rarest_k": cell["rarest_k"],
                "scope": source_scope(f"source{code}"),
                "kind": EXACT_KIND,
                "set": entry["set"],
                "jaccard_threshold": entry["threshold"],
                **volume_statistics(counts),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Exact-name pairs (grid-independent, so measured once)
# ---------------------------------------------------------------------------
def exact_packed_pairs(
    exact_index: ExactNameIndex,
    key_store: _NameKeyStore,
    log: logging.Logger,
    label: str,
) -> np.ndarray:
    """The shipped exact-name blocker's candidate set, packed, sorted and unique.

    Computed once rather than per cell because it depends on neither the cap nor K.

    ``key_store`` must hold the **same column the index was keyed on**, which is
    ``exact_index.key_field`` - ``name_norm`` in the shipped config, while the
    verification store holds ``name_key``. ``lookup_many`` hashes whatever strings it
    is handed, so querying a ``name_norm``-keyed index with ``name_key`` values is a
    silent miss rather than an error. The two columns are not the same string:
    ``name_key`` is ``name_norm`` with separators stripped.

    Returns packed ``(s1_position, entity_code)`` int64s, so a chunk's slice is two
    ``searchsorted`` calls on the sorted array rather than a per-cell recomputation.
    """
    started = time.time()
    positions = key_store.valid_positions()
    keys = [key_store.key_at_position(int(position)) for position in positions]
    lookups, counts = exact_index.lookup_many(keys)
    owners, entity_codes = exact_index.expand(lookups, counts)
    if owners.size == 0:
        log.info("%s: exact-name index found no pairs", label)
        return _EMPTY_INT64
    packed = np.unique(pack_pairs(positions[owners], entity_codes))
    log.info(
        "%s: %s exact-name pairs over %s S1 entities in %.1f s",
        label,
        fmt_int(packed.size),
        fmt_int(len(keys)),
        time.time() - started,
    )
    return packed


def resolve_exact_packed(
    config: dict,
    args: argparse.Namespace,
    exact_indexes: dict[str, Optional[ExactNameIndex]],
    s1_store: _NameKeyStore,
    artifact_root: Path,
    ground_truth,
    n_entities: int,
    fingerprint: str,
    log: logging.Logger,
) -> dict[str, np.ndarray]:
    """The exact-name candidate set per source, keyed the way the index is keyed.

    A store over ``name_norm`` is built only when a loaded index actually asks for it
    (``key_field != name_key``), and one store serves every source that shares the
    field - the config gives all sources the same ``blocking.exact_name.key``, but the
    index records the field it was built with, so that record is what is trusted.
    """
    wanted = {
        source: index.key_field
        for source, index in exact_indexes.items()
        if index is not None
    }
    stores: dict[str, _NameKeyStore] = {}
    for field in sorted(set(wanted.values())):
        if field == NAME_KEY:
            stores[field] = s1_store
            continue
        directory = artifact_root / f"source1_{field}"
        if args.resume and _artifact_ready(directory, fingerprint):
            log.info("resuming the source1 %s store (for the exact-name lookup)", field)
            stores[field] = _NameKeyStore.load(directory)
            continue
        store = build_name_key_store(
            s1_chunks(config, args, ["entity_id", field]),
            "entity_id",
            field,
            log,
            f"[source1] {field}",
            ground_truth=ground_truth,
            n_entities=n_entities,
        )
        store.save(directory)
        _write_artifact_marker(directory, fingerprint, store.describe())
        stores[field] = store

    out: dict[str, np.ndarray] = {}
    for source, index in exact_indexes.items():
        if index is None:
            continue
        out[source] = exact_packed_pairs(index, stores[wanted[source]], log, f"[{source}] exact")
    return out


def exact_slice(exact_packed: dict[str, np.ndarray], source: str, start: int, stop: int) -> np.ndarray:
    """Exact pairs whose S1 position lies in ``[start, stop)``.

    Packed pairs sort by (position, entity code), so a chunk's slice is two
    ``searchsorted`` calls.
    """
    packed = exact_packed.get(source)
    if packed is None or packed.size == 0:
        return _EMPTY_INT64
    low = np.searchsorted(packed, start * PAIR_MULTIPLIER)
    high = np.searchsorted(packed, stop * PAIR_MULTIPLIER)
    return packed[low:high]


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------
def evaluate_grid(
    args: argparse.Namespace,
    config: dict,
    indexes: dict[str, _TrigramIndex],
    selections: dict[str, _S1Selection],
    store_dirs: dict[str, str],
    exact_indexes: dict[str, Optional[ExactNameIndex]],
    exact_packed: dict[str, np.ndarray],
    ground_truth,
    cells: Sequence[tuple[int, int]],
    cell_volume: dict[tuple[int, int], int],
    n_entities: int,
    n_target_records: Optional[int],
    s1_store: _NameKeyStore,
    log: logging.Logger,
    timings: dict[str, float],
) -> tuple[list[dict], dict]:
    """Retrieve, verify and score every cell inside the candidate-row budget."""
    stage = time.time()
    source_codes = (2, 3)
    has_exact = bool(exact_packed)
    sets = list(CANDIDATE_SETS) if has_exact else [SET_CHAR]
    thresholds = list(args.jaccard_thresholds)
    needs_split = bool(config.get("evaluation", {}).get("split", {}).get("enabled", False))
    val_mask = split_mask_for(ground_truth, config, SPLIT_VAL) if needs_split else None

    # One evaluator, its truth structures shared by every accumulator set. k_values=()
    # on purpose: file-order recall@K is arbitrary for an unranked blocker, so
    # reporting it would put a meaningless number in the output. It becomes meaningful
    # once the blocker emits scores and belongs back in then.
    evaluation = CandidateEvaluation(
        ground_truth, n_target_records=n_target_records, k_values=(), log=log
    )

    # Workers are clamped to the work available, so the clamp needs a real chunk count.
    largest_volume = max(cell_volume.values()) if cell_volume else 0
    n_verify_chunks = max(1, int(largest_volume // max(1, args.verify_chunk_pairs)))
    workers = resolve_workers(
        args.workers,
        config.get("compute", {}).get("num_workers", 0),
        n_verify_chunks,
        log,
        "verify workers",
    )
    window, verify_chunk_pairs = plan_inflight_window(
        workers,
        args.verify_chunk_pairs,
        VERIFY_BYTES_PER_PAIR,
        _payload_budget(config),
        log,
        "verification payload",
    )
    log.info(
        "verification: %d workers, in-flight window %d, %s pairs per payload",
        workers,
        window,
        fmt_int(verify_chunk_pairs),
    )

    exact_only_metrics = None
    if has_exact:
        accumulator = _AccumulatorSet(n_entities, source_codes)
        for packed in exact_packed.values():
            owners, target_codes = unpack_pairs(packed)
            accumulator.add(owners, target_codes, evaluation.true_pair_codes)
        bind_accumulators(evaluation, accumulator)
        exact_only_metrics = extract_metrics(evaluation)
        log.info(
            "exact-name blocking alone: pair recall %.4f, %s candidates",
            exact_only_metrics["blocking_recall_pair"] or 0.0,
            fmt_int(exact_only_metrics["n_candidate_pairs"]),
        )
        del accumulator

    accumulators_per_cell = 6 * n_entities * 8 * len(thresholds) * len(sets)
    log.info(
        "evaluation: %d cells candidate; accumulator peak about %s (%d sets x %d thresholds)",
        len(cells),
        human_bytes(accumulators_per_cell),
        len(sets),
        len(thresholds),
    )

    grid: list[dict] = []
    exact_rows: list[dict] = []
    for cap, rarest_k in cells:
        volume = cell_volume[(cap, rarest_k)]
        if args.max_candidate_rows and volume > args.max_candidate_rows:
            log.warning(
                "cell (cap=%s, K=%s): expansion bound %s exceeds the %s budget - bound "
                "reported, recall NOT evaluated",
                cap,
                rarest_k,
                fmt_int(volume),
                fmt_int(args.max_candidate_rows),
            )
            grid.append(
                {
                    "df_cap": cap,
                    "rarest_k": rarest_k,
                    "volume": volume,
                    "volume_kind": BOUND_KIND,
                    "evaluated": False,
                    "skip_reason": "expansion_bound_above_max_candidate_rows",
                    "max_candidate_rows": args.max_candidate_rows,
                    "sets": [],
                }
            )
            continue

        cell_started = time.time()
        accumulators = {
            (set_name, threshold): _AccumulatorSet(n_entities, source_codes)
            for set_name in sets
            for threshold in thresholds
        }
        retrieved = 0
        verified = 0

        for source, index in indexes.items():
            filtered = index.filtered(cap, rarest_k)
            selection = selections[source]
            for start in range(0, n_entities, args.chunk_rows):
                stop = min(start + args.chunk_rows, n_entities)
                codes, positions, ranks, dfs = selection.slice_full(start, stop)
                if codes.size == 0:
                    continue
                keep = selection.cell_filter(ranks, dfs, cap, rarest_k)
                if not keep.any():
                    continue
                key_positions, counts = filtered.lookup_many(codes[keep])
                owners, target_codes = filtered.expand(key_positions, counts)
                if owners.size == 0:
                    continue

                packed = np.unique(pack_pairs(positions[keep][owners], target_codes))
                retrieved += int(packed.size)
                s1_positions, target_codes = unpack_pairs(packed)
                similarities = verify_similarities(
                    source,
                    s1_positions,
                    _target_rows(store_dirs, source, target_codes),
                    s1_store.key_at_position,
                    workers=workers,
                    window=window,
                    chunk_pairs=verify_chunk_pairs,
                    store_dirs=store_dirs,
                )
                verified += int(similarities.size)

                extra = exact_slice(exact_packed, source, start, stop)
                for threshold in thresholds:
                    kept = similarities >= threshold
                    if kept.any():
                        accumulators[(SET_CHAR, threshold)].add(
                            s1_positions[kept], target_codes[kept], evaluation.true_pair_codes
                        )
                    if SET_CHAR_PLUS_EXACT not in sets:
                        continue
                    if extra.size == 0:
                        union = packed[kept]
                    elif not kept.any():
                        union = extra
                    else:
                        union = np.unique(np.concatenate((packed[kept], extra)))
                    if union.size == 0:
                        continue
                    union_positions, union_codes = unpack_pairs(union)
                    accumulators[(SET_CHAR_PLUS_EXACT, threshold)].add(
                        union_positions, union_codes, evaluation.true_pair_codes
                    )

        entries: list[dict] = []
        for threshold in thresholds:
            for set_name in sets:
                accumulator = accumulators[(set_name, threshold)]
                bind_accumulators(evaluation, accumulator)
                entry = {
                    "set": set_name,
                    "threshold": float(threshold),
                    "metrics": extract_metrics(evaluation),
                }
                if val_mask is not None:
                    entry["metrics_val"] = extract_metrics(evaluation, val_mask, SPLIT_VAL)
                entries.append(entry)
                exact_rows.extend(
                    exact_volume_rows(
                        {"df_cap": cap, "rarest_k": rarest_k}, entry, accumulator, n_entities
                    )
                )
        del accumulators
        gc.collect()

        log.info(
            "cell (cap=%s, K=%s): bound %s, distinct %s candidates, %s verified, %.1f s",
            cap,
            rarest_k,
            fmt_int(volume),
            fmt_int(retrieved),
            fmt_int(verified),
            time.time() - cell_started,
        )
        grid.append(
            {
                "df_cap": cap,
                "rarest_k": rarest_k,
                "volume": volume,
                "volume_kind": BOUND_KIND,
                "evaluated": True,
                "pairs_retrieved": retrieved,
                "pairs_verified": verified,
                "elapsed_seconds": round(time.time() - cell_started, 1),
                "sets": entries,
            }
        )

    timings["evaluation_stage"] = round(time.time() - stage, 1)
    extras = {
        "workers": workers,
        "verify_window": window,
        "verify_chunk_pairs": verify_chunk_pairs,
        "sets": sets,
        "split_enabled": needs_split,
        "exact_available": {
            source: (exact_indexes.get(source) is not None) for source in indexes
        },
        "exact_only_metrics": exact_only_metrics,
        "exact_volume_rows": exact_rows,
    }
    return grid, extras


_STORE_CACHE: dict[str, _NameKeyStore] = {}


def _target_rows(store_dirs: dict[str, str], source: str, target_codes: np.ndarray) -> np.ndarray:
    """Target store rows for entity codes, loading each store once and caching it."""
    store = _STORE_CACHE.get(source)
    if store is None:
        store = _NameKeyStore.load(Path(store_dirs[source]))
        _STORE_CACHE[source] = store
    return store.row_for_code(target_codes)


def _payload_budget(config: dict) -> int:
    """RAM the queued verification payloads may occupy.

    Defaults to a quarter of currently-available RAM, matching the analyzer's
    ``compute.payload_budget_bytes`` convention, with a floor so a tiny box does not
    collapse the window to nothing.
    """
    configured = config.get("compute", {}).get("payload_budget_bytes")
    if configured:
        return int(configured)
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:
        available = None
    return max(256 * 1024**2, int(available) // 4) if available else 2 * 1024**3


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_markdown(report: dict) -> str:
    meta = report["meta"]
    lines: list[str] = []
    add = lines.append

    add("# char-3-gram blocker calibration")
    add("")
    add(f"* generated: {meta['generated_at']}")
    add(f"* prepared corpus: `{meta['prepared_dir']}`")
    add(f"* S1 entities: {meta['n_s1_entities']:,}  |  true pairs: {meta['n_true_pairs']:,}")
    add(f"* workers: {meta['workers']}  |  chunk rows: {meta['chunk_rows']:,}")
    add(
        f"* candidate-row budget per cell: {meta['max_candidate_rows']:,} "
        f"({'explicit guard' if meta['max_candidate_rows'] else 'disabled'})"
    )
    add(
        f"* cells: {meta['grid_cells']} total, {meta['cells_evaluated']} evaluated, "
        f"{meta['cells_skipped']} volume-only"
    )
    add(f"* elapsed: {meta['elapsed_minutes']:.1f} min")
    add("")

    add("## Grid")
    add("")
    add(f"* df caps: {meta['df_caps']}")
    add(f"* rarest-K: {meta['rarest_ks']}")
    add(f"* Jaccard thresholds: {meta['jaccard_thresholds']}")
    add(f"* candidate sets: {meta['candidate_sets']}")
    add("")

    add("## Candidate volume")
    add("")
    add(
        "`bound` counts the postings a cell's queries expand to and is an **upper bound** on the "
        "distinct candidate pairs (a target reached via two of an S1 entity's keys is counted "
        "twice there). `exact` is the distinct count, measured for evaluated cells. Both are "
        "reported for every evaluated cell; only `bound` exists for the rest."
    )
    add("")
    add(
        "| df cap | rarest-K | kind | set | Jaccard | candidates | avg/S1 | median | p90 | p99 | max | zero-candidate S1 |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in report["volume_curve"]:
        if row["scope"] != VOLUME_ALL:
            continue
        add(
            f"| {row['df_cap']} | {row['rarest_k']} | {row['kind']} | "
            f"{row.get('set') or ''} | "
            f"{('%.2f' % row['jaccard_threshold']) if row.get('jaccard_threshold') is not None else ''} | "
            f"{row['n_candidate_pairs']:,} | "
            f"{row['avg_candidates_per_s1']:.1f} | {row['median_candidates_per_s1']:.0f} | "
            f"{row['p90_candidates_per_s1']:.0f} | {row['p99_candidates_per_s1']:.0f} | "
            f"{row['max_candidates_per_s1']:,} | "
            f"{row['n_s1_with_zero_candidates']:,} "
            f"({row['fraction_s1_with_zero_candidates']*100:.2f}%) |"
        )
    add("")

    evaluated = [cell for cell in report["grid"] if cell.get("evaluated")]
    add("## Recall x volume, evaluated cells")
    add("")
    if not evaluated:
        add("No cell was evaluated. See the skipped table below.")
        add("")
    else:
        add(
            "| df cap | K | Jaccard | set | pair recall | macro entity recall | full-recall S1 | "
            "candidates | precision | reduction | F0.5 ceiling |"
        )
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        for cell in evaluated:
            for entry in cell["sets"]:
                m = entry["metrics"]
                add(
                    f"| {cell['df_cap']} | {cell['rarest_k']} | {entry['threshold']:.2f} | "
                    f"{entry['set']} | {m['blocking_recall_pair']*100:.2f}% | "
                    f"{m['macro_recall_entity']*100:.2f}% | {m['s1_full_recall_rate']*100:.2f}% | "
                    f"{m['n_candidate_pairs']:,} | {m['candidate_precision']*100:.3f}% | "
                    f"{m['reduction_ratio']:.1f}x | {m['f05_ceiling_from_macro_recall']:.4f} |"
                )
        add("")

    skipped = [cell for cell in report["grid"] if not cell.get("evaluated")]
    if skipped:
        add("## Cells measured but NOT evaluated")
        add("")
        add(
            "These exceeded the candidate-row budget. The number below is the measured expansion "
            "**upper bound**, not a truncated recall run - their recall is **not** measured and "
            "must not be read as full recall."
        )
        add("")
        add("| df cap | rarest-K | expansion bound | reason |")
        add("|---|---|---|---|")
        for cell in skipped:
            add(f"| {cell['df_cap']} | {cell['rarest_k']} | {cell['volume']:,} | {cell['skip_reason']} |")
        add("")

    add("## Entity reach and structural zeros")
    add("")
    add(
        "Each target source is keyed independently, so these counts are per source and an entity "
        "that is a structural zero for one source may be reachable in the other. A key can be "
        "lost three ways, counted separately at the trigram level: absent from this source's "
        "corpus (`df == 0`), too common here (`df` above the cap), or simply not among the "
        "entity's `K` rarest."
    )
    add("")
    add(
        "`s1_name_too_short_for_any_trigram` counts entities whose `name_key` is shorter than "
        "three code points. They have no trigram to key on, so they are unreachable by **any** "
        "trigram blocker - while Phase 0.1's `_trigram_jaccard` would still score a pair of them "
        "high, because it falls back to bare character sets. This is the one place where this "
        "blocker's reach is structurally narrower than the signal it thresholds."
    )
    add("")
    add("| source | rows read | no key here | name too short | dropped: absent | dropped: above cap | dropped: beyond K | entries/row |")
    add("|---|---|---|---|---|---|---|---|")
    for source, block in (report.get("structural") or {}).items():
        add(
            f"| {source} | {block.get('n_s1_rows_read', 0):,} | "
            f"{block.get('s1_with_no_key_in_this_source', 0):,} | "
            f"{block.get('s1_name_too_short_for_any_trigram', 0):,} | "
            f"{block.get('trigrams_dropped_absent_from_this_source', 0):,} | "
            f"{block.get('trigrams_dropped_above_df_cap', 0):,} | "
            f"{block.get('trigrams_dropped_beyond_rarest_k', 0):,} | "
            f"{block.get('entries_per_row', 0.0):.2f} |"
        )
    add("")

    if report.get("exact_only_metrics"):
        m = report["exact_only_metrics"]
        add("## Exact-name blocking alone (grid-independent)")
        add("")
        add(
            f"* pair recall: {m['blocking_recall_pair']*100:.2f}%  |  macro entity recall: "
            f"{m['macro_recall_entity']*100:.2f}%"
        )
        add(
            f"* candidates: {m['n_candidate_pairs']:,}  |  precision: "
            f"{m['candidate_precision']*100:.2f}%  |  full-recall S1: {m['s1_full_recall_rate']*100:.2f}%"
        )
        add("")

    add("### Token blocking, analytic reference only")
    add("")
    add("Measured by Phase 0, **not** by this script, and not enabled in config:")
    add("")
    for key, value in TOKEN_ANALYTIC_REFERENCE.items():
        add(f"* `{key}`: {value}")
    add("")

    if report.get("top_trigrams"):
        add("## Most frequent trigrams (evidence for the cap)")
        add("")
        add("| source | trigram | df | postings |")
        add("|---|---|---|---|")
        for row in report["top_trigrams"][:20]:
            add(f"| {row['source']} | `{row['trigram']}` | {row['df']:,} | {row['postings']:,} |")
        add("")

    add("## Caveats")
    add("")
    add("* No operating threshold is chosen here. The point of the run is the curve.")
    add(
        "* Verification calls Phase 0.1's `_trigram_jaccard` unchanged, so these thresholds are "
        "the same signal Phase 0 reported coverage for."
    )
    add(
        "* `char_plus_exact` is reported per cell; its delta against `char` is the measured value "
        "of the exact-name index once char retrieval is on."
    )
    add(
        "* The token blocker was not run. Its row in this report is a Phase 0 analytic reference, "
        "labelled as such."
    )
    add(
        "* Recall is measured against the training ground truth; per-cell validation-split metrics "
        "are in the JSON under `metrics_val`."
    )
    add("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _int_list(text: str) -> list[int]:
    return [int(part) for part in text.replace(",", " ").split()]


def _float_list(text: str) -> list[float]:
    return [float(part) for part in text.replace(",", " ").split()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="calibrate_char_blocker.py",
        description=(
            "Measure the char-3-gram blocker's recall x candidate-volume curve over a grid of "
            "(trigram df cap, rarest-K, Jaccard threshold). Retrieval is an inverted index; "
            "verification is bit-identical to Phase 0.1's _trigram_jaccard. No operating "
            "threshold is chosen here."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="config YAML path")
    parser.add_argument("--data-root", default=None, help="override paths.data_root")
    parser.add_argument("--work-dir", default=None, help="override paths.work_dir")
    parser.add_argument(
        "--split", default="train", choices=["train", "test"], help="prepared split to read"
    )
    parser.add_argument(
        "--sources", default=",".join(TARGET_SOURCES), help="target sources, indexed separately"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="where reports and artifacts go; default <work_dir>/calibration",
    )

    parser.add_argument(
        "--df-caps",
        type=_int_list,
        default=list(DEFAULT_DF_CAPS),
        help="trigram document-frequency caps to sweep; the maximum is what gets indexed",
    )
    parser.add_argument(
        "--rarest-ks",
        type=_int_list,
        default=list(DEFAULT_RAREST_KS),
        help="how many of an entity's rarest surviving trigrams act as its keys",
    )
    parser.add_argument(
        "--jaccard-thresholds",
        type=_float_list,
        default=list(DEFAULT_JACCARD_THRESHOLDS),
        help="exact 3-gram Jaccard thresholds a retrieved pair must clear",
    )

    parser.add_argument(
        "--max-candidate-rows",
        type=int,
        default=DEFAULT_MAX_CANDIDATE_ROWS,
        help=(
            "explicit, reported guard: a cell whose candidate volume exceeds this is measured "
            "but not evaluated. Never silently capped - the cell reports its true volume and "
            "'evaluated: false'. 0 disables the guard."
        ),
    )
    parser.add_argument("--workers", type=int, default=0, help="verification processes; 0 = auto")
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS, help="rows per chunk")
    parser.add_argument(
        "--verify-chunk-pairs",
        type=int,
        default=DEFAULT_VERIFY_CHUNK_PAIRS,
        help="pairs per verification payload",
    )
    parser.add_argument(
        "--limit-s1", type=int, default=None, help="max S1 rows to process (smoke tests)"
    )
    parser.add_argument(
        "--volume-only", action="store_true", help="measure the volume curve only; skip evaluation"
    )
    parser.add_argument("--top-trigrams", type=int, default=50, help="rows per source in the evidence CSV")
    parser.add_argument("--resume", action="store_true", help="reuse completed artifacts")
    parser.add_argument("--timings", action="store_true", help="record per-stage wall clock in the JSON")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Artifact fingerprinting and small helpers
# ---------------------------------------------------------------------------
def artifact_fingerprint(
    args: argparse.Namespace, config: dict, n_entities: int, target_rows: dict
) -> str:
    """Fingerprint every input the corpus artifacts depend on.

    A stale artifact silently reused would make the calibration unreproducible, so
    anything that changes a count belongs here: the prepared corpus, the source list,
    the index bounds, and the row counts themselves.
    """
    material = "|".join(
        str(part)
        for part in (
            ARTIFACT_VERSION,
            args.split,
            ",".join(sorted(args.sources)),
            max(args.df_caps),
            max(args.rarest_ks),
            args.chunk_rows,
            args.limit_s1,
            n_entities,
            config["resolved"]["prepared_dir"],
            ";".join(f"{key}={value}" for key, value in sorted(target_rows.items())),
        )
    )
    return hashlib.blake2b(material.encode("utf-8"), digest_size=16).hexdigest()


def _artifact_ready(directory: Path, fingerprint: str) -> bool:
    marker = directory / "artifact.done.json"
    if not marker.is_file():
        return False
    try:
        meta = read_json(marker)
    except Exception:
        return False
    return meta.get("version") == ARTIFACT_VERSION and meta.get("fingerprint") == fingerprint


def _write_artifact_marker(directory: Path, fingerprint: str, payload: dict) -> None:
    write_json(
        directory / "artifact.done.json",
        {"version": ARTIFACT_VERSION, "fingerprint": fingerprint, **payload},
    )


def target_row_counts(config: dict, split: str, sources: Sequence[str], log: logging.Logger) -> dict:
    """Rows per target source, from ``prepare_manifest.json`` so nothing is re-read.

    Only used to make ``reduction_ratio`` and ``n_possible_pairs`` meaningful - the
    evaluator emits both only when it is told how many target records exist - so a
    missing or unreadable manifest degrades to omitting them rather than failing.
    """
    manifest = Path(config["resolved"]["prepared_dir"]) / "prepare_manifest.json"
    if not manifest.is_file():
        log.warning("no prepare manifest at %s; pair-reduction metrics will be omitted", manifest)
        return {}
    try:
        entries = read_json(manifest).get("sources", [])
    except Exception as exc:  # a corrupt manifest must not be fatal
        log.warning("could not read %s (%s); pair-reduction metrics will be omitted", manifest, exc)
        return {}
    counts: dict[str, int] = {}
    for entry in entries:
        if entry.get("split") == split and entry.get("source") in sources:
            counts[entry["source"]] = int(entry.get("rows", 0))
    return counts


def count_s1_rows(config: dict, args: argparse.Namespace) -> int:
    """Number of S1 rows this run will read, without holding them."""
    total = 0
    for frame in iter_prepared(
        config, args.split, "source1", columns=["entity_id"], chunksize=args.chunk_rows
    ):
        total += len(frame)
        if args.limit_s1 is not None and total >= args.limit_s1:
            return args.limit_s1
    return total


def limited(chunks: Iterable[pd.DataFrame], limit: Optional[int]) -> Iterable[pd.DataFrame]:
    """Truncate a chunk stream to ``limit`` rows."""
    if limit is None:
        yield from chunks
        return
    seen = 0
    for frame in chunks:
        if seen >= limit:
            return
        if seen + len(frame) > limit:
            frame = frame.iloc[: limit - seen]
        seen += len(frame)
        yield frame


def s1_chunks(config: dict, args: argparse.Namespace, columns: Sequence[str]) -> Iterable[pd.DataFrame]:
    return limited(
        iter_prepared(config, args.split, "source1", columns=columns, chunksize=args.chunk_rows),
        args.limit_s1,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    started = time.time()
    log = setup_logging(LOG_NAME, level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        config = load_config(
            args.config,
            overrides={
                key: value
                for key, value in (("data_root", args.data_root), ("work_dir", args.work_dir))
                if value
            },
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2

    set_seed(config.get("project", {}).get("seed", 42))

    sources = [part.strip() for part in args.sources.split(",") if part.strip()]
    unknown = [source for source in sources if source not in TARGET_SOURCES]
    if unknown or not sources:
        log.error(
            "invalid --sources %r; expected a non-empty subset of %s",
            unknown or args.sources,
            list(TARGET_SOURCES),
        )
        return 2
    if not args.df_caps or any(cap <= 0 for cap in args.df_caps):
        log.error("--df-caps must be a non-empty list of positive ints: %s", args.df_caps)
        return 2
    if not args.rarest_ks or any(k <= 0 or k > 255 for k in args.rarest_ks):
        # Ranks are stored as uint8 because an entity can have at most a few hundred
        # trigrams and 255 is far above any useful K.
        log.error("--rarest-ks must be a non-empty list in [1, 255]: %s", args.rarest_ks)
        return 2
    if not args.jaccard_thresholds or any(not 0.0 < t <= 1.0 for t in args.jaccard_thresholds):
        log.error("--jaccard-thresholds must be a non-empty list in (0, 1]: %s", args.jaccard_thresholds)
        return 2
    if args.chunk_rows <= 0 or args.verify_chunk_pairs <= 0:
        log.error("--chunk-rows and --verify-chunk-pairs must be positive")
        return 2

    log.info(describe_environment(config))
    hardware = detect_hardware()
    log.info("hardware:\n%s", format_hardware_report(hardware))
    if args.split != "train":
        log.warning(
            "--split %s: the ground truth exists only for the train split, so recall here is "
            "measured against the TRAIN ground truth. Use this only for candidate generation.",
            args.split,
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config["resolved"]["work_dir"]) / "calibration"
    )
    ensure_dir(output_dir)
    artifact_root = output_dir / ARTIFACT_DIRNAME

    ground_truth = load_ground_truth(config, log=log)
    n_entities = ground_truth.n_entities
    s1_rows = count_s1_rows(config, args)
    if s1_rows <= 0:
        log.error("no source1 rows found; run scripts/prepare_data.py first")
        return 2
    log.info(
        "S1 rows to read: %s | ground-truth entities: %s | true pairs: %s",
        fmt_int(s1_rows),
        fmt_int(n_entities),
        fmt_int(ground_truth.n_matches),
    )

    target_rows = target_row_counts(config, args.split, sources, log)
    n_target_records = sum(target_rows.values()) if target_rows else None
    fingerprint = artifact_fingerprint(args, config, n_entities, target_rows)

    cells = [(cap, k) for cap in args.df_caps for k in args.rarest_ks]
    log.info(
        "grid: %d df caps x %d rarest-K = %d cells; %d thresholds x %d sets; indexing at "
        "(cap=%s, K=%s)",
        len(args.df_caps),
        len(args.rarest_ks),
        len(cells),
        len(args.jaccard_thresholds),
        len(CANDIDATE_SETS),
        max(args.df_caps),
        max(args.rarest_ks),
    )

    timings: dict[str, float] = {}
    indexes: dict[str, _TrigramIndex] = {}
    selections: dict[str, _S1Selection] = {}
    dfs: dict[str, _TrigramDf] = {}
    structural: dict[str, dict] = {}
    store_dirs: dict[str, str] = {}
    top_trigrams: list[dict] = []

    # ---- shared S1 name-key store, addressable by ground-truth position -----
    s1_store_dir = artifact_root / "source1_namekeys"
    stage = time.time()
    if args.resume and _artifact_ready(s1_store_dir, fingerprint):
        log.info("resuming the source1 name-key store")
        s1_store = _NameKeyStore.load(s1_store_dir)
    else:
        s1_store = build_name_key_store(
            s1_chunks(config, args, ["entity_id", NAME_KEY]),
            "entity_id",
            NAME_KEY,
            log,
            "[source1] namekeys",
            ground_truth=ground_truth,
            n_entities=n_entities,
        )
        s1_store.save(s1_store_dir)
        _write_artifact_marker(s1_store_dir, fingerprint, s1_store.describe())
    timings["source1_namekey_store"] = round(time.time() - stage, 1)

    # ---- per-source corpus artifacts ---------------------------------------
    for source in sources:
        stage = time.time()
        df_dir = artifact_root / source / "df"
        index_dir = artifact_root / source / "index"
        select_dir = artifact_root / source / "s1_selection"
        store_dir = artifact_root / source / "namekeys"

        if args.resume and _artifact_ready(df_dir, fingerprint):
            log.info("[%s] resuming the trigram df table", source)
            df = _TrigramDf.load(df_dir)
        else:
            df = count_trigram_df(
                iter_prepared(config, args.split, source, columns=[NAME_KEY], chunksize=args.chunk_rows),
                NAME_KEY,
                log,
                f"[{source}] df",
            )
            df.save(df_dir)
            _write_artifact_marker(df_dir, fingerprint, df.describe(args.df_caps))
        dfs[source] = df

        if args.resume and _artifact_ready(store_dir, fingerprint):
            log.info("[%s] resuming the target name-key store", source)
        else:
            store = build_name_key_store(
                iter_prepared(
                    config,
                    args.split,
                    source,
                    columns=["entity_id", NAME_KEY],
                    chunksize=args.chunk_rows,
                ),
                "entity_id",
                NAME_KEY,
                log,
                f"[{source}] namekeys",
            )
            store.save(store_dir)
            _write_artifact_marker(store_dir, fingerprint, store.describe())
        store_dirs[source] = str(store_dir)

        if args.resume and _artifact_ready(index_dir, fingerprint):
            log.info("[%s] resuming the trigram index", source)
            index = _TrigramIndex.load(index_dir, source)
        else:
            index = build_trigram_index(
                iter_prepared(
                    config,
                    args.split,
                    source,
                    columns=["entity_id", NAME_KEY],
                    chunksize=args.chunk_rows,
                ),
                df,
                "entity_id",
                NAME_KEY,
                max(args.df_caps),
                max(args.rarest_ks),
                source,
                log,
                f"[{source}] index",
            )
            index.save(index_dir)
            _write_artifact_marker(index_dir, fingerprint, index.describe())
        indexes[source] = index

        if args.resume and _artifact_ready(select_dir, fingerprint):
            log.info("[%s] resuming the S1 key selection", source)
            selection = _S1Selection.load(select_dir)
            structural[source] = read_json(select_dir / "artifact.done.json").get("breakdown", {})
        else:
            selection, breakdown = build_s1_selection(
                s1_chunks(config, args, ["entity_id", NAME_KEY]),
                df,
                ground_truth,
                "entity_id",
                NAME_KEY,
                max(args.df_caps),
                max(args.rarest_ks),
                n_entities,
                log,
                f"[{source}] s1",
            )
            selection.save(select_dir)
            _write_artifact_marker(select_dir, fingerprint, {"breakdown": breakdown})
            structural[source] = breakdown
        selections[source] = selection

        top_trigrams.extend(
            {"source": source, "trigram": text, "df": df_value, "postings": postings}
            for text, df_value, postings in index.top_keys(args.top_trigrams)
        )
        timings[f"{source}_artifacts"] = round(time.time() - stage, 1)
        log.info(
            "[%s] artifacts ready in %.1f min (rss %s)",
            source,
            (time.time() - stage) / 60.0,
            human_bytes(current_rss_bytes() or 0),
        )

    log.info("df tables: %s", {source: dfs[source].describe(args.df_caps) for source in sources})
    log.info("indexes: %s", {source: indexes[source].describe() for source in sources})
    log.info("S1 selections: %s", {source: selections[source].describe() for source in sources})

    # ---- volume sweep: every cell ------------------------------------------
    stage = time.time()
    volume_rows, cell_volume = sweep_volume(
        indexes, selections, cells, n_entities, args.chunk_rows, log
    )
    timings["volume_sweep"] = round(time.time() - stage, 1)

    # ---- exact-name index availability (read-only) -------------------------
    exact_indexes: dict[str, Optional[ExactNameIndex]] = {}
    for source in sources:
        directory = index_dir_for(config, args.split, source, BLOCKER_EXACT_NAME)
        try:
            exact_indexes[source] = ExactNameIndex.load(directory, log=None)
            log.info("[%s] exact-name index available: %s", source, exact_indexes[source].describe())
        except (FileNotFoundError, ValueError) as exc:
            exact_indexes[source] = None
            log.warning(
                "[%s] exact-name index unavailable (%s); the %s set will be reported as "
                "unavailable. Run: python scripts/build_indexes.py",
                source,
                exc,
                SET_CHAR_PLUS_EXACT,
            )

    # ---- the exact-name candidate set (grid-independent) -------------------
    stage = time.time()
    exact_packed: dict[str, np.ndarray] = {}
    if not args.volume_only and any(index is not None for index in exact_indexes.values()):
        exact_packed = resolve_exact_packed(
            config,
            args,
            exact_indexes,
            s1_store,
            artifact_root,
            ground_truth,
            n_entities,
            fingerprint,
            log,
        )
    timings["exact_pairs"] = round(time.time() - stage, 1)

    # ---- evaluation stage --------------------------------------------------
    extras: dict[str, Any] = {}
    if args.volume_only:
        log.info("--volume-only: skipping the evaluation stage")
        grid = [
            {
                "df_cap": cap,
                "rarest_k": rarest_k,
                "volume": cell_volume[(cap, rarest_k)],
                "volume_kind": BOUND_KIND,
                "evaluated": False,
                "skip_reason": "volume_only_requested",
                "sets": [],
            }
            for cap, rarest_k in cells
        ]
    else:
        grid, extras = evaluate_grid(
            args,
            config,
            indexes,
            selections,
            store_dirs,
            exact_indexes,
            exact_packed,
            ground_truth,
            cells,
            cell_volume,
            n_entities,
            n_target_records,
            s1_store,
            log,
            timings,
        )

    elapsed = time.time() - started
    timings["total"] = round(elapsed, 1)
    volume_rows.extend(extras.get("exact_volume_rows", []))

    report: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "log_name": LOG_NAME,
            "split": args.split,
            "ground_truth_split": "train",
            "sources": sources,
            "workers": extras.get("workers"),
            "verify_window": extras.get("verify_window"),
            "verify_chunk_pairs": extras.get("verify_chunk_pairs"),
            "chunk_rows": args.chunk_rows,
            "limit_s1": args.limit_s1,
            "s1_rows_read": int(s1_rows),
            "df_caps": list(args.df_caps),
            "rarest_ks": list(args.rarest_ks),
            "jaccard_thresholds": list(args.jaccard_thresholds),
            "max_candidate_rows": args.max_candidate_rows,
            "volume_only": bool(args.volume_only),
            "candidate_sets": extras.get("sets", [SET_CHAR]),
            "n_s1_entities": int(n_entities),
            "n_true_pairs": int(ground_truth.n_matches),
            "n_target_records": n_target_records,
            "prepared_dir": str(config["resolved"]["prepared_dir"]),
            "hardware": hardware,
            "elapsed_minutes": round(elapsed / 60.0, 2),
            "timings_seconds": timings if args.timings else None,
            "artifact_fingerprint": fingerprint,
            "resume_requested": bool(args.resume),
            "verification": "scripts/analyze_name_differences._trigram_jaccard (imported unchanged)",
            "grid_cells": len(cells),
            "cells_evaluated": sum(1 for cell in grid if cell.get("evaluated")),
            "cells_skipped": sum(1 for cell in grid if not cell.get("evaluated")),
            "split_metrics_enabled": extras.get("split_enabled"),
            "exact_index_available": extras.get("exact_available"),
            "trigram_semantics": "name_key, Phase 0.1 code-point trigram sets",
            "notes": [
                "No operating threshold is chosen by this script; it produces the evidence.",
                "The token blocker was NOT enabled and NOT run; see token_analytic_reference.",
                "Cells above --max-candidate-rows report their true volume and evaluated=false.",
                "The exact-name blocker is read from the existing index; it is not modified.",
            ],
        },
        "grid": grid,
        "volume_curve": volume_rows,
        "structural": structural,
        "top_trigrams": top_trigrams,
        "exact_only_metrics": extras.get("exact_only_metrics"),
        "artifacts": {
            "indexes": {source: indexes[source].describe() for source in sources},
            "df": {source: dfs[source].describe(args.df_caps) for source in sources},
            "s1_selection": {source: selections[source].describe() for source in sources},
            "s1_namekey_store": s1_store.describe(),
            "target_row_counts": target_rows,
            "exact_candidate_pairs": {
                source: int(packed.size) for source, packed in exact_packed.items()
            },
            "exact_key_fields": {
                source: (index.key_field if index is not None else None)
                for source, index in exact_indexes.items()
            },
        },
        "token_analytic_reference": TOKEN_ANALYTIC_REFERENCE,
    }

    json_path = output_dir / "char_blocker_calibration.json"
    write_json(json_path, report)
    log.info("wrote %s", json_path)

    csv_rows: list[dict] = []
    for cell in grid:
        if not cell.get("evaluated"):
            continue
        for entry in cell["sets"]:
            row = {
                "df_cap": cell["df_cap"],
                "rarest_k": cell["rarest_k"],
                "jaccard_threshold": entry["threshold"],
                "set": entry["set"],
                "volume": cell["volume"],
            }
            for key in REPORTED_METRICS:
                row[key] = entry["metrics"].get(key)
            for prefix, block in (entry["metrics"].get("per_source") or {}).items():
                for key, value in block.items():
                    row[f"{prefix}_{key}"] = value
            csv_rows.append(row)
    write_csv(output_dir / "char_blocker_calibration.csv", csv_rows)
    write_csv(output_dir / "char_blocker_volume.csv", volume_rows)
    write_csv(output_dir / "char_blocker_top_trigrams.csv", top_trigrams)

    markdown_path = output_dir / "char_blocker_calibration.md"
    markdown_path.write_text(build_markdown(report), encoding="utf-8")
    log.info(
        "wrote %s, char_blocker_calibration.csv, char_blocker_volume.csv, "
        "char_blocker_top_trigrams.csv, char_blocker_calibration.md",
        json_path.name,
    )
    log.info(
        "done: %d cells, %d evaluated, %d volume-only; elapsed %.1f min; rss %s",
        len(cells),
        report["meta"]["cells_evaluated"],
        report["meta"]["cells_skipped"],
        elapsed / 60.0,
        human_bytes(current_rss_bytes() or 0),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
