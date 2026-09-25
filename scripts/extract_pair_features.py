#!/usr/bin/env python
"""Step 3 de-risk: materialize the first matcher features on a sampled candidate set.

This is a **measurement script, not a pipeline stage**. It exists to answer one
question before anyone writes the matcher: can the pair-feature stage run over the
production candidate set at all, and at what cost?

It deliberately does NOT:

* process all 336M pairs (it samples S1 entities and keeps every candidate of the
  sampled entities, so the per-entity structure the macro metric depends on is
  preserved);
* train anything, tune anything, or touch the blockers;
* read ground truth. The validation split is derived from the S1 id alone, with
  ``assign_splits`` - the same pure function ``src/evaluation.py`` uses - so no
  label can leak into a feature and the split cannot drift from the evaluator's.

Why a two-phase design
----------------------
``s1_candidate_count`` is one of the most valuable features in the first matcher
(it tells the model how crowded an entity's candidate list is), and it has to be
known before the sampled rows are featurized. Phase 1 streams the candidate file
once, keeps the sampled rows, and accumulates the per-S1 counts on the way past;
phase 2 reads only the small sampled file, joins it to the prepared text, and
computes features. That buys two things: the scan - the dominant cost - is
measured on its own rather than mixed into the feature timing, and all the
string work happens on a few million rows instead of 336M.

Note what the counts are and are not. Because the sample keeps **whole** entities,
an entity's rows in the sample are all of its rows in the file, so the counts are
also derivable from the sample alone - phase 1 computes them only because that
pass already touches every row. Phase 2 therefore re-counts them independently and
compares: any mismatch means an entity was split across the sample boundary, which
would silently corrupt the feature, and the run fails rather than reporting it.

    python scripts/extract_pair_features.py
    python scripts/extract_pair_features.py --sample-fraction 0.05
    python scripts/extract_pair_features.py --limit-rows 200000        # smoke test

Outputs, all inside one experiment directory so nothing lands in the production
``outputs/candidates/`` tree::

    sample_candidates.tsv        the sampled candidate rows, verbatim
    features.tsv                 ids + features, one row per candidate pair
    feature_missingness.csv      per-feature missing rate
    step3_features_report.json   every measurement this experiment must report
    extract_pair_features.log    the run log
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

# Make ``import src.*`` work when the script is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import (  # noqa: E402
    BLOCKER_CHAR_NGRAM,
    BLOCKER_EXACT_NAME,
    BLOCKER_TOKEN,
    UNION_BLOCKERS,
    _trigram_jaccard,
    evidence_columns_for,
)
from src.data_loader import (  # noqa: E402
    ChunkWriter,
    assign_splits,
    candidates_path,
    describe_environment,
    iter_tsv,
    load_config,
    prepared_path,
)
from src.utils import (  # noqa: E402
    current_rss_bytes,
    ensure_dir,
    fmt_int,
    human_bytes,
    log_memory,
    peak_rss_bytes,
    set_seed,
    setup_logging,
    stable_hash64,
    write_json,
)

LOG_NAME = "extract_pair_features"

# Candidate schema (see scripts/generate_candidates.py). The evidence columns are
# present only when the blocker that produces them was enabled.
CANDIDATE_S1_COLUMN = "source1_entity_id"
CANDIDATE_TARGET_COLUMN = "matched_entity_id"
CANDIDATE_SOURCE_COLUMN = "source"
CANDIDATE_BLOCKERS_COLUMN = "blockers"

# Prepared schema (see scripts/prepare_data.py).
PREPARED_ID_COLUMN = "entity_id"
PREPARED_NAME_NORM = "name_norm"
PREPARED_NAME_KEY = "name_key"
PREPARED_ADDRESS_NORM = "address_norm"
PREPARED_COUNTRY_NORM = "country_norm"

PREPARED_COLUMNS = (
    PREPARED_ID_COLUMN,
    PREPARED_NAME_NORM,
    PREPARED_NAME_KEY,
    PREPARED_ADDRESS_NORM,
    PREPARED_COUNTRY_NORM,
)

# The full-corpus production acceptance target, used only for extrapolation. It is
# written down rather than re-measured: this experiment's job is to extrapolate,
# not to re-run generation.
FULL_CANDIDATE_PAIRS = 336_056_756
HPC_WORKERS = 48

# Evidence columns carried by the candidate file, and the dtypes to parse them as.
# A blank means "the blocker that measures this value did not propose this pair",
# which is the normal case in a union - so blank must become NaN, never 0.
EVIDENCE_FLOAT_COLUMNS = evidence_columns_for(UNION_BLOCKERS)

# Feature dtypes. Fixed up front so the matrix is dense, typed and small, and so
# the report can state exactly what the full-scale matrix would cost.
FEATURE_DTYPES: dict[str, str] = {
    # -- name --
    "name_norm_equal": "uint8",
    "name_key_equal": "uint8",
    "name_token_jaccard": "float32",
    "name_token_set_ratio": "float32",
    "name_token_sort_ratio": "float32",
    "name_partial_ratio": "float32",
    "name_char3_jaccard": "float32",
    "name_length_ratio": "float32",
    "name_token_count_diff": "int16",
    "name_first_token_equal": "uint8",
    # -- address --
    "address_norm_equal": "uint8",
    "address_token_jaccard": "float32",
    "address_shared_token_count": "int16",
    "address_length_ratio": "float32",
    "s1_address_missing": "uint8",
    "target_address_missing": "uint8",
    "both_address_missing": "uint8",
    # -- blocker evidence + provenance --
    "token_df": "float32",
    "char_jaccard": "float32",
    "blocker_exact_name": "uint8",
    "blocker_token": "uint8",
    "blocker_char_ngram": "uint8",
    "n_blockers": "uint8",
    "s1_candidate_count": "int32",
    # -- other --
    "source_is_s2": "uint8",
    "country_equal": "uint8",
    "country_missing": "uint8",
    # -- integrity, not a feature --
    "text_join_ok": "uint8",
}

# Columns present in the written matrix that are NOT features and must be dropped
# before training. They are written so a failed join stays diagnosable per row, but
# a trainer that reads the whole matrix would hand the model a column describing
# whether the feature pipeline worked - which is not a property of the pair.
NON_FEATURE_COLUMNS = ("text_join_ok",)

# Features that are defined on [0, 1]. Checked on every batch so a scale error
# (e.g. leaving rapidfuzz on its 0-100 scale) trips the check instead of silently
# feeding the model a 0-100 column. ``token_df`` is deliberately NOT in this list:
# it is a document frequency, not a similarity.
UNIT_INTERVAL_FEATURES = (
    "name_token_jaccard",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_char3_jaccard",
    "name_length_ratio",
    "address_token_jaccard",
    "address_length_ratio",
    "char_jaccard",
)

INTEGRITY_COLUMNS = (CANDIDATE_S1_COLUMN, CANDIDATE_TARGET_COLUMN)

# Features that come from joining to the prepared text. A failed join blanks these
# and only these - provenance, evidence and the S1 candidate count are properties of
# the candidate file itself and stay valid either way.
TEXT_DERIVED_FEATURES = tuple(
    column
    for column in FEATURE_DTYPES
    if column.startswith(("name_", "address_", "country_"))
    or column in ("s1_address_missing", "target_address_missing", "both_address_missing")
)


def _sample_rss(tracker: list[int]) -> None:
    """Record the current RSS, keeping the running maximum.

    ``peak_rss_bytes()`` returns ``None`` on Windows (``resource`` is unavailable),
    so the highest *observed* RSS is tracked as a portable lower bound on the true
    peak. On Linux the two agree closely, because the run logs at every phase
    boundary and every ten feature batches.
    """
    value = current_rss_bytes()
    if value:
        tracker[0] = max(tracker[0], value)


def _peak_description(exact: Optional[int], sampled: int) -> tuple[Optional[int], str]:
    """The best peak-RSS number available, and where it came from."""
    if exact and sampled:
        return max(exact, sampled), "resource.getrusage high-water mark"
    if exact:
        return exact, "resource.getrusage high-water mark"
    if sampled:
        return sampled, "max observed RSS (getrusage unavailable on this platform)"
    return None, "unavailable"


# ---------------------------------------------------------------------------
# pure per-pair helpers (the semantics live here and nowhere else)
# ---------------------------------------------------------------------------
def token_set(text: str) -> frozenset[str]:
    """Distinct whitespace tokens of an already-normalized field."""
    return frozenset(text.split()) if text else frozenset()


def set_jaccard(left: frozenset, right: frozenset) -> float:
    """Jaccard overlap of two token sets; 0.0 when either side is empty.

    An empty side means "no tokens to compare", which is not evidence of a match,
    so the intersection-based 1.0 that an empty-vs-empty comparison would produce
    is deliberately not returned.
    """
    if not left or not right:
        return 0.0
    shared = len(left & right)
    return shared / (len(left) + len(right) - shared)


def length_ratio(left: str, right: str) -> float:
    """Shorter/longer character length; 0.0 when either side is empty."""
    a, b = len(left), len(right)
    if a == 0 or b == 0:
        return 0.0
    return min(a, b) / max(a, b)


def first_token(text: str) -> str:
    """First whitespace token, or "" for an empty field."""
    parts = text.split()
    return parts[0] if parts else ""


def parse_provenance(text: str) -> tuple[int, int, int, int, int]:
    """``(exact, token, char, n_blockers, unknowns)`` from a provenance string.

    Provenance survives the union as comma-joined ``sourceN:blocker`` labels (see
    ``union_blockers``), so the blocker part is what identifies the blocker.
    """
    exact = token = char = unknown = 0
    if text:
        for part in text.split(","):
            blocker = part.rsplit(":", 1)[-1]
            if blocker == BLOCKER_EXACT_NAME:
                exact = 1
            elif blocker == BLOCKER_TOKEN:
                token = 1
            elif blocker == BLOCKER_CHAR_NGRAM:
                char = 1
            else:
                unknown += 1
    return exact, token, char, exact + token + char, unknown


# ---------------------------------------------------------------------------
# prepared-text lookup
# ---------------------------------------------------------------------------
class PreparedLookup:
    """``entity_id`` -> the normalized text columns, for one source.

    A dict of positions plus one object array per column: the join is then a
    Python ``get`` per sampled row, which is cheap because only sampled rows are
    ever looked up. Loading the *whole* prepared file (all 10.3M target records)
    is deliberate - it is the honest measure of what the join costs in RAM, and it
    is the reason the sampled rows can be featurized in a single streaming pass.
    """

    def __init__(self, source: str, ids: Sequence[str], columns: dict[str, np.ndarray]) -> None:
        self.source = source
        self.ids = ids
        self.columns = columns
        self.index = {entity_id: position for position, entity_id in enumerate(ids)}

    @property
    def n_entities(self) -> int:
        return len(self.ids)

    def take(self, entity_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Positions and a found-mask for a batch of ids; -1 where absent."""
        index = self.index
        positions = np.fromiter(
            (index.get(entity_id, -1) for entity_id in entity_ids),
            dtype=np.int64,
            count=len(entity_ids),
        )
        return positions, positions >= 0

    def values(self, column: str, positions: np.ndarray, found: np.ndarray) -> np.ndarray:
        """Text for a column; "" where the id was not found."""
        values = self.columns[column]
        out = np.full(len(positions), "", dtype=object)
        if found.any():
            out[found] = values[positions[found]]
        return out

    TEXT_SAMPLE_ROWS = 2000

    def memory_bytes(self) -> int:
        """Estimated resident bytes of this lookup.

        The string payload of an object array is the dominant cost and is not in
        ``nbytes``, so it is estimated by sampling ``TEXT_SAMPLE_ROWS`` elements per
        column and extrapolating. This is an estimate, not a measurement: it exists
        to answer "how many GB per worker does the full prepared source cost?", and
        the process-wide RSS line in the log is the check on it.
        """
        total = sum(int(array.nbytes) for array in self.columns.values())
        total += len(self.ids) * 8 * (1 + len(self.columns)) + len(self.index) * 100
        for array in (self.ids, *self.columns.values()):
            n = len(array)
            if n == 0:
                continue
            if n <= self.TEXT_SAMPLE_ROWS:
                sample = array
            else:
                sample = array[:: n // self.TEXT_SAMPLE_ROWS]
            total += sum(sys.getsizeof(value) for value in sample) * (n / len(sample))
        return int(total)


def load_lookup(config: dict, split: str, source: str, log: logging.Logger) -> PreparedLookup:
    """Load one prepared source into a :class:`PreparedLookup`."""
    path = prepared_path(config, split, source)
    if not path.is_file():
        raise FileNotFoundError(
            f"prepared file missing: {path}\n"
            f"  Run first: python scripts/prepare_data.py --splits {split}"
        )
    parts: list[list[str]] = []
    buffers: dict[str, list[np.ndarray]] = {column: [] for column in PREPARED_COLUMNS if column != PREPARED_ID_COLUMN}
    chunksize = max(1, int(config.get("io", {}).get("chunksize", 500_000)))
    for frame in iter_tsv(path, columns=list(PREPARED_COLUMNS), chunksize=chunksize):
        parts.append(frame[PREPARED_ID_COLUMN].to_numpy(dtype=object))
        for column in buffers:
            buffers[column].append(frame[column].to_numpy(dtype=object))
    ids = np.concatenate(parts) if parts else np.empty(0, dtype=object)
    columns = {
        column: (np.concatenate(values) if values else np.empty(0, dtype=object))
        for column, values in buffers.items()
    }
    lookup = PreparedLookup(source, ids, columns)
    log.info(
        "  loaded %s prepared rows from %s (lookup ~%s, RSS now %s)",
        fmt_int(lookup.n_entities),
        path.name,
        human_bytes(lookup.memory_bytes()),
        human_bytes(current_rss_bytes() or 0),
    )
    return lookup


# ---------------------------------------------------------------------------
# phase 1: sample S1 entities out of the candidate file
# ---------------------------------------------------------------------------
def _sample_mask_for_ids(
    unique_ids: np.ndarray,
    cache: dict[str, int],
    config: dict,
    sub_threshold: int,
    log: logging.Logger,
) -> np.ndarray:
    """Which of ``unique_ids`` are in the sample. Pure function of the id.

    Two conditions, both deterministic functions of the S1 id and nothing else:
    the entity must be in the validation split (``assign_splits``, the same
    function ``src/evaluation.py`` uses), and it must fall below
    ``sub_threshold`` in a second bucket taken from the high bits of the same
    64-bit hash. Because both are functions of the id alone, every candidate row
    of an entity makes the same decision, so whole entities are kept together
    without needing the file to be grouped by S1.
    """
    section = config.get("evaluation", {}).get("split", {}) or {}
    val_fraction = section.get("val_fraction", 0.2)
    mode = section.get("mode", "hash")
    seed = config.get("project", {}).get("seed", 42)

    missing = [entity_id for entity_id in unique_ids if entity_id not in cache]
    if missing:
        series = pd.Series(missing, dtype=object)
        labels = assign_splits(series, val_fraction=val_fraction, mode=mode, seed=seed)
        is_val = labels == "val"
        # A different slice of the same hash than assign_splits uses, so the
        # subsample is independent of the split decision.
        buckets = (stable_hash64(series) // np.uint64(1_000_000)) % np.uint64(1_000_000)
        keep = is_val & (buckets < np.uint64(sub_threshold))
        for entity_id, flag in zip(missing, keep):
            cache[entity_id] = 1 if flag else 0
    return np.fromiter(
        (cache[entity_id] for entity_id in unique_ids), dtype=bool, count=len(unique_ids)
    )


def scan_and_sample(
    config: dict,
    args: argparse.Namespace,
    output_dir: Path,
    log: logging.Logger,
) -> dict[str, Any]:
    """Phase 1: stream the candidate file, write the sampled rows, count per S1."""
    source_path = candidates_path(config, args.candidates)
    if not source_path.is_file():
        raise FileNotFoundError(
            f"candidate file not found: {source_path}\n"
            f"  Run first: python scripts/generate_candidates.py --split {args.split}"
        )

    available = _peek_columns(source_path)
    missing = [
        column
        for column in (CANDIDATE_S1_COLUMN, CANDIDATE_TARGET_COLUMN, CANDIDATE_SOURCE_COLUMN)
        if column not in available
    ]
    if missing:
        raise ValueError(f"candidate file {source_path} is missing column(s) {missing}")

    read_columns = [
        column
        for column in (
            CANDIDATE_S1_COLUMN,
            CANDIDATE_TARGET_COLUMN,
            CANDIDATE_SOURCE_COLUMN,
            CANDIDATE_BLOCKERS_COLUMN,
            *EVIDENCE_FLOAT_COLUMNS,
        )
        if column in available
    ]
    evidence_columns = [column for column in EVIDENCE_FLOAT_COLUMNS if column in available]

    sub_threshold = int(round(min(max(args.sample_fraction, 0.0), 1.0) * 1_000_000))
    chunksize = args.chunksize or int(config.get("io", {}).get("chunksize", 500_000))

    log.info("scanning %s", source_path)
    log.info("  columns read : %s", ", ".join(read_columns))
    log.info("  evidence cols: %s", ", ".join(evidence_columns) or "(none)")
    log.info("  sample       : %.3f%% of validation S1 entities", args.sample_fraction * 100.0)

    cache: dict[str, int] = {}
    counts: dict[str, int] = {}
    seen_pairs: set[tuple[str, str]] = set()
    duplicate_pairs = 0
    rows_scanned = 0
    rows_sampled = 0
    seen_entities: set[str] = set()
    rss_tracker = [0]
    sample_path = output_dir / "sample_candidates.tsv.partial"

    started = time.time()
    with ChunkWriter(sample_path) as writer:
        chunks = iter_tsv(source_path, columns=read_columns, chunksize=chunksize)
        for chunk in chunks:
            if args.limit_rows is not None and rows_scanned >= args.limit_rows:
                break
            if args.limit_rows is not None and rows_scanned + len(chunk) > args.limit_rows:
                chunk = chunk.iloc[: args.limit_rows - rows_scanned]
            rows_scanned += len(chunk)

            s1_array = chunk[CANDIDATE_S1_COLUMN].to_numpy(dtype=object)
            # Factorize first: a 500k-row chunk holds only a few thousand distinct
            # S1 entities, so the sample decision is made a few thousand times
            # instead of 500k times.
            codes, uniques = pd.factorize(s1_array, sort=False)
            # Exact global distinct count: union the per-chunk unique ids. This costs one
            # set insert per per-chunk-distinct id, and generate_candidates.py emits a
            # whole entity's candidate block contiguously, so a 500k-row chunk holds a few
            # thousand distinct ids rather than 500k - the insert count stays near the true
            # distinct count (2.2M) instead of near the row count (336M). It is NOT a sum
            # of per-chunk distinct counts, which would double-count ids straddling chunks.
            seen_entities.update(uniques)
            keep_by_entity = _sample_mask_for_ids(
                uniques, cache, config, sub_threshold, log
            )
            keep = keep_by_entity[codes] if len(keep_by_entity) else np.zeros(len(chunk), dtype=bool)
            if not keep.any():
                _sample_rss(rss_tracker)
                continue

            sampled = chunk.loc[keep]
            rows_sampled += len(sampled)
            writer.append(sampled)

            # Per-S1 candidate counts: the S1's size in the WHOLE file, not in the
            # sample. This is the feature the matcher needs to know how crowded an
            # entity's candidate list is.
            chunk_counts = np.bincount(codes[keep], minlength=len(uniques))
            for position in np.flatnonzero(chunk_counts):
                entity_id = uniques[position]
                counts[entity_id] = counts.get(entity_id, 0) + int(chunk_counts[position])

            # Exact duplicate detection within the sample. A duplicated pair in the
            # file puts both copies in the sample (same S1), so the sampled rate is
            # an unbiased estimate of the file-wide rate - and it needs a set of
            # only the sampled pairs rather than of all 336M.
            for entity_id, target_id in zip(
                sampled[CANDIDATE_S1_COLUMN].to_numpy(dtype=object),
                sampled[CANDIDATE_TARGET_COLUMN].to_numpy(dtype=object),
            ):
                key = (entity_id, target_id)
                if key in seen_pairs:
                    duplicate_pairs += 1
                else:
                    seen_pairs.add(key)
            _sample_rss(rss_tracker)

    scan_seconds = time.time() - started

    if rows_sampled == 0:
        raise RuntimeError(
            "no candidate row matched the sample; is the candidate file empty, or "
            "is --sample-fraction too small?"
        )
    sample_path.replace(output_dir / "sample_candidates.tsv")

    log.info(
        "  scanned %s rows in %.1f s (%.0f rows/s); sampled %s pairs over %s S1 entities",
        fmt_int(rows_scanned),
        scan_seconds,
        rows_scanned / max(scan_seconds, 1e-9),
        fmt_int(rows_sampled),
        fmt_int(len(counts)),
    )
    log_memory(log, "after scan")

    return {
        "source_path": str(source_path),
        "read_columns": read_columns,
        "evidence_columns": evidence_columns,
        "rows_scanned": rows_scanned,
        "rows_sampled": rows_sampled,
        "limited": args.limit_rows is not None,
        "n_s1_entities_sampled": len(counts),
        "n_s1_entities_seen": len(seen_entities),
        "n_s1_entity_id_set_bytes": sum(
            sys.getsizeof(entity_id) for entity_id in seen_entities
        ),
        "duplicate_sampled_pairs": duplicate_pairs,
        "per_s1_counts": counts,
        "scan_seconds": scan_seconds,
        "rss_sampled_peak": rss_tracker[0],
        "sample_path": output_dir / "sample_candidates.tsv",
        "sample_fraction": args.sample_fraction,
    }


def _peek_columns(path: Path) -> list[str]:
    """Read just the header of a (possibly compressed) TSV."""
    import gzip

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    return header.rstrip("\n").split("\t")


# ---------------------------------------------------------------------------
# phase 2: join + features
# ---------------------------------------------------------------------------
def _bump(integrity: dict[str, int], key: str, amount: int) -> None:
    """Increment one integrity counter, creating it if the caller did not pre-seed it.

    ``build_features`` accumulates into a caller-owned dict, so a caller that passes
    ``{}`` must get a working counter rather than a ``KeyError`` deep in the join. The
    key is created even when ``amount`` is 0, so a counter that happened to stay at
    zero is still present (and readable) rather than missing.
    """
    integrity[key] = integrity.get(key, 0) + int(amount)


def build_features(
    frame: pd.DataFrame,
    lookups: dict[str, PreparedLookup],
    s1_lookup: PreparedLookup,
    counts: dict[str, int],
    integrity: dict[str, int],
) -> pd.DataFrame:
    """One row of features per candidate pair. Never drops a pair."""
    n = len(frame)
    s1_ids = frame[CANDIDATE_S1_COLUMN].to_numpy(dtype=object)
    target_ids = frame[CANDIDATE_TARGET_COLUMN].to_numpy(dtype=object)
    source_labels = frame[CANDIDATE_SOURCE_COLUMN].to_numpy(dtype=object)

    # -- join ---------------------------------------------------------------
    s1_positions, s1_found = s1_lookup.take(s1_ids)
    target_positions = np.full(n, -1, dtype=np.int64)
    target_found = np.zeros(n, dtype=bool)
    for label, lookup in lookups.items():
        selected = np.flatnonzero(source_labels == label)
        if not selected.size:
            continue
        positions, found = lookup.take(target_ids[selected])
        target_positions[selected] = positions
        target_found[selected] = found
    unknown_source = ~np.isin(source_labels, list(lookups))

    text_join_ok = s1_found & target_found & ~unknown_source
    _bump(integrity, "s1_join_failures", (~s1_found).sum())
    _bump(integrity, "target_join_failures", (~target_found & ~unknown_source).sum())
    _bump(integrity, "unknown_source_labels", unknown_source.sum())

    s1_name_norm = s1_lookup.values(PREPARED_NAME_NORM, s1_positions, s1_found)
    s1_name_key = s1_lookup.values(PREPARED_NAME_KEY, s1_positions, s1_found)
    s1_address = s1_lookup.values(PREPARED_ADDRESS_NORM, s1_positions, s1_found)
    s1_country = s1_lookup.values(PREPARED_COUNTRY_NORM, s1_positions, s1_found)

    target_name_norm = np.full(n, "", dtype=object)
    target_name_key = np.full(n, "", dtype=object)
    target_address = np.full(n, "", dtype=object)
    target_country = np.full(n, "", dtype=object)
    for label, lookup in lookups.items():
        selected = np.flatnonzero((source_labels == label) & target_found)
        if not selected.size:
            continue
        positions = target_positions[selected]
        target_name_norm[selected] = lookup.values(PREPARED_NAME_NORM, positions, target_found[selected])
        target_name_key[selected] = lookup.values(PREPARED_NAME_KEY, positions, target_found[selected])
        target_address[selected] = lookup.values(PREPARED_ADDRESS_NORM, positions, target_found[selected])
        target_country[selected] = lookup.values(PREPARED_COUNTRY_NORM, positions, target_found[selected])

    # -- name ---------------------------------------------------------------
    s1_tokens = [token_set(text) for text in s1_name_norm]
    target_tokens = [token_set(text) for text in target_name_norm]
    name_token_jaccard = np.fromiter(
        (set_jaccard(a, b) for a, b in zip(s1_tokens, target_tokens)),
        dtype=np.float64,
        count=n,
    )

    try:
        from rapidfuzz import fuzz

        token_set_ratio = np.fromiter(
            (fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(s1_name_norm, target_name_norm)),
            dtype=np.float64,
            count=n,
        )
        token_sort_ratio = np.fromiter(
            (fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(s1_name_norm, target_name_norm)),
            dtype=np.float64,
            count=n,
        )
        partial_ratio = np.fromiter(
            (fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(s1_name_norm, target_name_norm)),
            dtype=np.float64,
            count=n,
        )
        integrity["rapidfuzz_available"] = 1
    except ImportError:  # pragma: no cover - depends on the environment
        token_set_ratio = np.full(n, np.nan)
        token_sort_ratio = np.full(n, np.nan)
        partial_ratio = np.full(n, np.nan)
        integrity["rapidfuzz_available"] = 0

    # The same trigram Jaccard the char blocker verifies with, on the same field
    # (name_key), so the feature and the blocker cannot disagree.
    name_char3_jaccard = np.fromiter(
        (_trigram_jaccard(a, b) for a, b in zip(s1_name_key, target_name_key)),
        dtype=np.float64,
        count=n,
    )
    name_length_ratio = np.fromiter(
        (length_ratio(a, b) for a, b in zip(s1_name_norm, target_name_norm)),
        dtype=np.float64,
        count=n,
    )
    name_token_count_diff = np.fromiter(
        (abs(len(a) - len(b)) for a, b in zip(s1_tokens, target_tokens)),
        dtype=np.int64,
        count=n,
    )
    name_first_token_equal = np.fromiter(
        (1 if first_token(a) and first_token(a) == first_token(b) else 0
         for a, b in zip(s1_name_norm, target_name_norm)),
        dtype=np.int64,
        count=n,
    )

    # -- address ------------------------------------------------------------
    # ``address_norm_equal`` deliberately requires a non-blank address. Two records
    # that both have no address compare equal as empty strings, and that "" == ""
    # carries no evidence of a match - it would hand the model a spurious positive.
    # The missingness flags below let it learn the absence explicitly instead.
    address_norm_equal = np.fromiter(
        (1 if a and a == b else 0 for a, b in zip(s1_address, target_address)),
        dtype=np.int64,
        count=n,
    )
    s1_address_missing = np.fromiter((1 if not a else 0 for a in s1_address), dtype=np.int64, count=n)
    target_address_missing = np.fromiter((1 if not a else 0 for a in target_address), dtype=np.int64, count=n)
    both_address_missing = (s1_address_missing & target_address_missing).astype(np.int64)

    s1_address_tokens = [token_set(text) for text in s1_address]
    target_address_tokens = [token_set(text) for text in target_address]
    address_token_jaccard = np.fromiter(
        (set_jaccard(a, b) for a, b in zip(s1_address_tokens, target_address_tokens)),
        dtype=np.float64,
        count=n,
    )
    address_shared_token_count = np.fromiter(
        (len(a & b) for a, b in zip(s1_address_tokens, target_address_tokens)),
        dtype=np.int64,
        count=n,
    )
    address_length_ratio = np.fromiter(
        (length_ratio(a, b) for a, b in zip(s1_address, target_address)),
        dtype=np.float64,
        count=n,
    )

    # -- blocker evidence + provenance --------------------------------------
    provenance = frame[CANDIDATE_BLOCKERS_COLUMN].to_numpy(dtype=object) if CANDIDATE_BLOCKERS_COLUMN in frame else np.full(n, "", dtype=object)
    parsed = [parse_provenance(text) for text in provenance]
    blocker_exact = np.fromiter((p[0] for p in parsed), dtype=np.int64, count=n)
    blocker_token = np.fromiter((p[1] for p in parsed), dtype=np.int64, count=n)
    blocker_char = np.fromiter((p[2] for p in parsed), dtype=np.int64, count=n)
    n_blockers = np.fromiter((p[3] for p in parsed), dtype=np.int64, count=n)
    _bump(integrity, "unknown_blocker_labels", sum(p[4] for p in parsed))

    evidence: dict[str, np.ndarray] = {}
    for column in EVIDENCE_FLOAT_COLUMNS:
        if column in frame:
            # Blank -> NaN, which is the whole point: a blank evidence cell means
            # the blocker that measures it did not propose this pair.
            parsed_column = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            _bump(integrity, f"{column}_blank", np.isnan(parsed_column).sum())
            evidence[column] = parsed_column
        else:
            evidence[column] = np.full(n, np.nan)

    s1_candidate_count = np.fromiter(
        (counts.get(entity_id, -1) for entity_id in s1_ids), dtype=np.int64, count=n
    )
    _bump(integrity, "missing_s1_counts", (s1_candidate_count < 0).sum())

    # -- other --------------------------------------------------------------
    source_is_s2 = (source_labels == "S2").astype(np.int64)
    country_equal = np.fromiter(
        (1 if a and a == b else 0 for a, b in zip(s1_country, target_country)),
        dtype=np.int64,
        count=n,
    )
    country_missing = np.fromiter(
        (1 if not a or not b else 0 for a, b in zip(s1_country, target_country)),
        dtype=np.int64,
        count=n,
    )

    features = pd.DataFrame(
        {
            # Like address_norm_equal, both equality flags require a non-blank left
            # side: "" == "" is not evidence that two records share a name, and
            # handing the model that 1 is a spurious positive - which F0.5 punishes
            # four times harder than it punishes a miss. A blank name therefore
            # scores 0, the same as two different names.
            "name_norm_equal": np.fromiter(
                (1 if a and a == b else 0 for a, b in zip(s1_name_norm, target_name_norm)),
                dtype=np.int64,
                count=n,
            ),
            "name_key_equal": np.fromiter(
                (1 if a and a == b else 0 for a, b in zip(s1_name_key, target_name_key)),
                dtype=np.int64,
                count=n,
            ),
            "name_token_jaccard": name_token_jaccard,
            "name_token_set_ratio": token_set_ratio,
            "name_token_sort_ratio": token_sort_ratio,
            "name_partial_ratio": partial_ratio,
            "name_char3_jaccard": name_char3_jaccard,
            "name_length_ratio": name_length_ratio,
            "name_token_count_diff": name_token_count_diff,
            "name_first_token_equal": name_first_token_equal,
            "address_norm_equal": address_norm_equal,
            "address_token_jaccard": address_token_jaccard,
            "address_shared_token_count": address_shared_token_count,
            "address_length_ratio": address_length_ratio,
            "s1_address_missing": s1_address_missing,
            "target_address_missing": target_address_missing,
            "both_address_missing": both_address_missing,
            "token_df": evidence["token_df"],
            "char_jaccard": evidence["char_jaccard"],
            "blocker_exact_name": blocker_exact,
            "blocker_token": blocker_token,
            "blocker_char_ngram": blocker_char,
            "n_blockers": n_blockers,
            "s1_candidate_count": s1_candidate_count,
            "source_is_s2": source_is_s2,
            "country_equal": country_equal,
            "country_missing": country_missing,
            "text_join_ok": text_join_ok.astype(np.int64),
        },
        index=frame.index,
    )

    # A failed text join leaves every TEXT-DERIVED feature meaningless. Blank those
    # rather than let a "" comparison masquerade as a measurement, and keep the row:
    # one row per candidate pair, always. Provenance, evidence and the S1 candidate
    # count come from the candidate file, not from the join, so they stay valid.
    failed = ~text_join_ok
    if failed.any():
        for column in TEXT_DERIVED_FEATURES:
            if FEATURE_DTYPES[column] == "float32":
                features.loc[failed, column] = np.nan
            else:
                features.loc[failed, column] = 0

    for column, dtype in FEATURE_DTYPES.items():
        features[column] = features[column].astype(dtype)

    ids = pd.DataFrame(
        {
            CANDIDATE_S1_COLUMN: s1_ids,
            CANDIDATE_TARGET_COLUMN: target_ids,
            CANDIDATE_SOURCE_COLUMN: source_labels,
        },
        index=frame.index,
    )
    return pd.concat([ids, features], axis=1)


def extract_features(
    config: dict,
    args: argparse.Namespace,
    scan: dict[str, Any],
    output_dir: Path,
    log: logging.Logger,
) -> dict[str, Any]:
    """Phase 2: join the sampled rows to prepared text and write the features."""
    log.info("loading prepared text for the join")
    split = args.split
    s1_lookup = load_lookup(config, split, "source1", log)
    lookups = {
        label: load_lookup(config, split, source, log)
        for source, label in (("source2", "S2"), ("source3", "S3"))
    }
    lookup_bytes = s1_lookup.memory_bytes() + sum(item.memory_bytes() for item in lookups.values())
    log_memory(log, "after prepared lookups")
    rss_after_lookup = current_rss_bytes()
    rss_tracker = [scan.get("rss_sampled_peak", 0)]
    _sample_rss(rss_tracker)

    feature_path = output_dir / "features.tsv.partial"
    integrity: dict[str, int] = {
        "s1_join_failures": 0,
        "target_join_failures": 0,
        "unknown_source_labels": 0,
        "unknown_blocker_labels": 0,
        "missing_s1_counts": 0,
        "rapidfuzz_available": 0,
    }
    for column in EVIDENCE_FLOAT_COLUMNS:
        integrity[f"{column}_blank"] = 0

    missing_counts: dict[str, int] = {column: 0 for column in FEATURE_DTYPES}
    blank_counts: dict[str, int] = {column: 0 for column in INTEGRITY_COLUMNS}
    value_min: dict[str, float] = {}
    value_max: dict[str, float] = {}
    out_of_range: dict[str, int] = {}
    dtypes_seen: dict[str, str] = {}
    rows_written = 0
    matrix_bytes = 0

    counts = scan["per_s1_counts"]
    sample_counts: dict[str, int] = {}
    started = time.time()
    with ChunkWriter(feature_path) as writer:
        for batch_index, frame in enumerate(
            iter_tsv(scan["sample_path"], chunksize=args.feature_batch_size), start=1
        ):
            features = build_features(frame, lookups, s1_lookup, counts, integrity)
            # Independent re-count of the sampled rows, to prove the whole-entity
            # invariant held. Cheap: one factorize per batch.
            batch_codes, batch_ids = pd.factorize(
                frame[CANDIDATE_S1_COLUMN].to_numpy(dtype=object), sort=False
            )
            batch_counts = np.bincount(batch_codes, minlength=len(batch_ids))
            for position in np.flatnonzero(batch_counts):
                entity_id = batch_ids[position]
                sample_counts[entity_id] = sample_counts.get(entity_id, 0) + int(batch_counts[position])
            rows_written += len(features)
            matrix_bytes += int(
                sum(features[column].to_numpy().nbytes for column in FEATURE_DTYPES)
            )
            for column in FEATURE_DTYPES:
                if column not in dtypes_seen:
                    dtypes_seen[column] = str(features[column].dtype)
            for column in INTEGRITY_COLUMNS:
                if column in features:
                    blank_counts[column] += int((features[column] == "").sum())
            for column, dtype in FEATURE_DTYPES.items():
                values = features[column].to_numpy()
                if dtype == "float32":
                    missing_counts[column] += int(np.isnan(values).sum())
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        value_min[column] = min(value_min.get(column, np.inf), float(finite.min()))
                        value_max[column] = max(value_max.get(column, -np.inf), float(finite.max()))
                        if column in UNIT_INTERVAL_FEATURES:
                            bad = int(np.count_nonzero((finite < 0.0) | (finite > 1.0)))
                            if bad:
                                out_of_range[column] = out_of_range.get(column, 0) + bad
                elif column == "s1_candidate_count":
                    value_min[column] = min(value_min.get(column, np.inf), float(values.min()))
                    value_max[column] = max(value_max.get(column, -np.inf), float(values.max()))
            if batch_index % 10 == 0:
                log_memory(log, f"features: {fmt_int(rows_written)} rows")
            _sample_rss(rss_tracker)
            writer.append(features)

    feature_seconds = time.time() - started
    feature_path.replace(output_dir / "features.tsv")

    # The whole-entity invariant: an entity's rows in the sample must be exactly
    # its rows in the file. A mismatch means the sample split an entity, which
    # would make s1_candidate_count wrong for it - silently, and in a feature the
    # matcher leans on. Fail loudly instead.
    mismatches = {
        entity_id: (counts.get(entity_id, 0), sample_counts[entity_id])
        for entity_id in sample_counts
        if counts.get(entity_id, 0) != sample_counts[entity_id]
    }
    extra_in_scan = set(counts) - set(sample_counts)
    integrity["count_mismatches"] = len(mismatches) + len(extra_in_scan)
    integrity["count_mismatch_examples"] = {str(k): v for k, v in list(mismatches.items())[:5]}
    if integrity["count_mismatches"]:
        # Not fatal here: the report still has to be written so the failure is
        # visible on disk. ``main`` turns a non-zero count into a non-zero exit.
        log.error(
            "whole-entity invariant violated: %s entities disagree between the scan "
            "and the sample (examples: %s)",
            fmt_int(integrity["count_mismatches"]),
            integrity["count_mismatch_examples"],
        )

    log.info(
        "  featurized %s pairs in %.1f s (%.0f pairs/s)",
        fmt_int(rows_written),
        feature_seconds,
        rows_written / max(feature_seconds, 1e-9),
    )
    log_memory(log, "after features")

    peak = peak_rss_bytes()
    peak_value, peak_source = _peak_description(peak, rss_tracker[0])
    features_path = output_dir / "features.tsv"
    sample_bytes = scan["sample_path"].stat().st_size

    result = {
        "rows_featurized": rows_written,
        "feature_seconds": feature_seconds,
        "matrix_bytes": matrix_bytes,
        "dtypes_seen": dtypes_seen,
        "missing_counts": missing_counts,
        "blank_counts": blank_counts,
        "value_min": value_min,
        "value_max": value_max,
        "out_of_range": out_of_range,
        "integrity": integrity,
        "lookup_bytes": lookup_bytes,
        "rss_after_lookup_bytes": rss_after_lookup,
        "peak_rss_bytes": peak,
        "sampled_peak_rss_bytes": rss_tracker[0],
        "peak_rss_value": peak_value,
        "peak_rss_source": peak_source,
        "feature_path": features_path,
        "feature_bytes": features_path.stat().st_size,
        "sample_bytes": sample_bytes,
    }
    return result


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def summarize(
    features: dict[str, Any],
    scan: dict[str, Any],
    total_seconds: float,
    log: logging.Logger,
) -> dict[str, Any]:
    """Assemble the report, including the full-scale extrapolation."""
    rows = features["rows_featurized"]
    per_row_bytes = features["matrix_bytes"] / max(rows, 1)
    per_row_tsv = features["feature_bytes"] / max(rows, 1)

    # The scan cost is measured on the whole file already - phase 1 reads every
    # row - so it needs no scaling. Only the feature work is extrapolated.
    # Unless --limit-rows stopped the scan early, in which case the scan is scaled
    # too, and the report says the scan timing is an extrapolation rather than a
    # measurement.
    feature_seconds_full = features["feature_seconds"] * (FULL_CANDIDATE_PAIRS / max(rows, 1))
    if scan["limited"]:
        scan_full = scan["scan_seconds"] * (FULL_CANDIDATE_PAIRS / max(scan["rows_scanned"], 1))
    else:
        scan_full = scan["scan_seconds"]
    total_single = scan_full + feature_seconds_full
    total_parallel = scan_full + feature_seconds_full / HPC_WORKERS
    scan_was_measured = not scan["limited"]
    matrix_full = per_row_bytes * FULL_CANDIDATE_PAIRS
    tsv_full = per_row_tsv * FULL_CANDIDATE_PAIRS

    # 540 GB is the HPC node's RAM; the matrix has to fit with room for the
    # lookups, one batch and the model that follows it.
    fits_in_ram = matrix_full < 0.25 * 540 * 1024**3

    # A per-pair cost measured on a handful of rows is noise. Say so rather than
    # let a linear extrapolation from it be quoted as a projection.
    if rows >= 1_000_000:
        confidence = "high (>=1M sampled pairs)"
    elif rows >= 100_000:
        confidence = "moderate (100k-1M sampled pairs)"
    else:
        confidence = (
            f"low ({fmt_int(rows)} sampled pairs): the per-pair cost is not stable "
            f"at this size, so the projections below are indicative only"
        )

    report = {
        "sample": {
            "n_s1_entities_sampled": scan["n_s1_entities_sampled"],
            "n_s1_entities_seen": scan["n_s1_entities_seen"],
            "n_candidate_pairs_sampled": rows,
            "rows_scanned": scan["rows_scanned"],
            "sample_fraction": scan["sample_fraction"],
            "sample_candidates_file_rows": rows,
        },        "timing": {
            "scan_seconds": round(scan["scan_seconds"], 2),
            "feature_seconds": round(features["feature_seconds"], 2),
            "total_seconds": round(total_seconds, 2),
            "scan_rows_per_sec": round(scan["rows_scanned"] / max(scan["scan_seconds"], 1e-9), 1),
            "feature_pairs_per_sec": round(rows / max(features["feature_seconds"], 1e-9), 1),
        },
        "memory": {
            "peak_rss_bytes": features["peak_rss_value"],
            "peak_rss": human_bytes(features["peak_rss_value"] or 0),
            "peak_rss_source": features["peak_rss_source"],
            "rss_after_prepared_lookups": human_bytes(features["rss_after_lookup_bytes"] or 0),
            "prepared_lookup_estimate": human_bytes(features["lookup_bytes"]),
            "n_s1_id_set_bytes": scan["n_s1_entity_id_set_bytes"],
        },
        "features": {
            "n_columns": len(FEATURE_DTYPES),
            "n_features": len(FEATURE_DTYPES) - len(NON_FEATURE_COLUMNS),
            "non_feature_columns": list(NON_FEATURE_COLUMNS),
            "dtypes": features["dtypes_seen"],
            "matrix_bytes": features["matrix_bytes"],
            "matrix_row_bytes": round(per_row_bytes, 2),
            "matrix_size": human_bytes(features["matrix_bytes"]),
            "missingness": {
                column: {
                    "rate": round(
                        features["missing_counts"].get(column, 0) / max(rows, 1), 6
                    ),
                    "count": int(features["missing_counts"].get(column, 0)),
                    "dtype": features["dtypes_seen"].get(column),
                    "min": features["value_min"].get(column),
                    "max": features["value_max"].get(column),
                    "out_of_unit_range": int(features["out_of_range"].get(column, 0)),
                }
                for column in FEATURE_DTYPES
            },
        },
        "integrity": {
            **features["integrity"],
            "duplicate_sampled_pairs": scan["duplicate_sampled_pairs"],
            "duplicate_rate": round(
                scan["duplicate_sampled_pairs"] / max(rows, 1), 8
            ),
            "blank_id_columns": features["blank_counts"],
            "matrix_columns": list(FEATURE_DTYPES),
        },
        "outputs": {
            "sample_candidates": {
                "path": str(scan["sample_path"]),
                "bytes": features["sample_bytes"],
                "size": human_bytes(features["sample_bytes"]),
            },
            "features": {
                "path": str(features["feature_path"]),
                "bytes": features["feature_bytes"],
                "size": human_bytes(features["feature_bytes"]),
            },
        },
        "extrapolation": {
            "full_candidate_pairs": FULL_CANDIDATE_PAIRS,
            "assumed_workers": HPC_WORKERS,
            "confidence": confidence,
            "scan_seconds_measured_on_full_file": round(scan["scan_seconds"], 2),
            "scan_seconds_full_scale": round(scan_full, 2),
            "scan_timing_is_a_measurement": scan_was_measured,
            "feature_seconds_single_process": round(feature_seconds_full, 1),
            "total_seconds_single_process": round(total_single, 1),
            "total_seconds_at_48_workers": round(total_parallel, 1),
            "total_hours_at_48_workers": round(total_parallel / 3600.0, 2),
            "matrix_bytes_full": int(matrix_full),
            "matrix_size_full": human_bytes(matrix_full),
            "features_tsv_bytes_full": int(tsv_full),
            "features_tsv_size_full": human_bytes(tsv_full),
            "matrix_fits_in_ram": bool(fits_in_ram),
            "assumptions": [
                "the 48-worker figure assumes the feature kernel scales near-linearly with "
                "workers; the kernel is CPU-bound and shares no state between workers, but "
                "this has NOT been measured - measure it on the node before quoting it",
                "the per-pair cost depends on name length, and this sample's names may be "
                "shorter (cheaper) than the real ones, so the throughput here is an upper "
                "bound on the real run",
            ] + (
                []
                if scan_was_measured
                else [
                    "the scan timing is NOT a measurement: --limit-rows stopped the scan "
                    "early, so scan_seconds_full_scale is a linear extrapolation from a "
                    "prefix of the file",
                ]
            ),
        },
    }

    log.info("=" * 78)
    log.info("sample : %s S1 entities, %s candidate pairs", fmt_int(report["sample"]["n_s1_entities_sampled"]), fmt_int(rows))
    log.info("scan   : %.1f s (%.0f rows/s) for %s rows", scan["scan_seconds"], report["timing"]["scan_rows_per_sec"], fmt_int(scan["rows_scanned"]))
    log.info("features: %.1f s (%.0f pairs/s)", features["feature_seconds"], report["timing"]["feature_pairs_per_sec"])
    log.info("peak RSS: %s", report["memory"]["peak_rss"])
    log.info(
        "full %s pairs: ~%.2f h at %d workers, matrix %s",
        fmt_int(FULL_CANDIDATE_PAIRS),
        report["extrapolation"]["total_hours_at_48_workers"],
        HPC_WORKERS,
        report["extrapolation"]["matrix_size_full"],
    )
    log.info("=" * 78)
    return report


def write_missingness_csv(report: dict[str, Any], path: Path) -> None:
    """Per-feature missingness as a CSV, for eyeballing."""
    rows = [
        {
            "feature": column,
            "dtype": entry["dtype"],
            "missing_rate": entry["rate"],
            "missing_count": entry["count"],
            "min": entry["min"],
            "max": entry["max"],
            "out_of_unit_range": entry["out_of_unit_range"],
        }
        for column, entry in report["features"]["missingness"].items()
    ]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3 de-risk: sampled pair features for the matcher."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--candidates", default="candidate_pairs", help="candidate file stem")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=0.03,
        help="fraction of VALIDATION S1 entities to sample (whole entities). "
        "0.03 of a 20%% val split is ~13k entities, ~2M candidate pairs",
    )
    parser.add_argument("--chunksize", type=int, default=None, help="candidate rows per chunk")
    parser.add_argument("--feature-batch-size", type=int, default=200_000, help="rows per feature batch")
    parser.add_argument("--limit-rows", type=int, default=None, help="max candidate rows to scan (smoke test)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="experiment directory (default: <candidates_dir>/../experiments/step3_features)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(
        args.config, overrides={"data_root": args.data_root, "work_dir": args.work_dir}
    )
    # Keep the experiment in its own directory next to (never inside) the
    # production outputs, so nothing here can be mistaken for a pipeline artifact.
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = Path(config["resolved"]["candidates_dir"]).parent / "experiments" / "step3_features"
    ensure_dir(output_dir)

    log = setup_logging(
        LOG_NAME,
        log_dir=output_dir,
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    set_seed(config.get("project", {}).get("seed", 42))

    log.info("=" * 78)
    log.info("extract_pair_features: Step 3 de-risk (sampled pair features)")
    log.info(describe_environment(config))
    log.info("output_dir=%s", output_dir)
    log.info("=" * 78)

    started = time.time()
    scan = scan_and_sample(config, args, output_dir, log)
    features = extract_features(config, args, scan, output_dir, log)
    report = summarize(features, scan, time.time() - started, log)

    report["inputs"] = {
        "candidates": scan["source_path"],
        "split": args.split,
        "sample_fraction": args.sample_fraction,
        "chunksize": args.chunksize,
        "feature_batch_size": args.feature_batch_size,
        "limit_rows": args.limit_rows,
        "read_columns": scan["read_columns"],
        "evidence_columns": scan["evidence_columns"],
        "config_path": config.get("config_path"),
    }
    write_json(output_dir / "step3_features_report.json", report)
    write_missingness_csv(report, output_dir / "feature_missingness.csv")
    log.info("report: %s", output_dir / "step3_features_report.json")

    if features["out_of_range"]:
        log.error("unit-interval features outside [0, 1]: %s", features["out_of_range"])
        return 1
    if features["integrity"].get("count_mismatches"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
