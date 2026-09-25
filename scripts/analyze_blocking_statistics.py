#!/usr/bin/env python
"""Phase 0.2-0.5: measurements that decide what the blocker should be.

This is a MEASUREMENT script. It builds no blocker, trains no model, writes no
candidate file and changes nothing in the pipeline - it reads the prepared
tables, the ground truth and (read-only, for counting) the persisted exact-name
indexes, and writes statistics. Four questions are answered, one per phase:

* **0.2 signal coverage and complementarity** - for every TRUE pair, which of
  the four country-agnostic signals reach it: exact ``name_norm``, a rare name
  token, character 3-grams of ``name_key``, or address-token Jaccard? Then the
  JOINT view no single-signal phase can give: the coverage of every combination,
  the IRREDUCIBLE RESIDUE (pairs no signal reaches), and what the residue looks
  like - cross-tabbed with Phase 0.1's name-difference categories and with
  script difference. This is the recall ceiling of any cheap lexical blocker.
  It replaces the earlier country-agreement analysis: country has two values in
  training, unseen values in test, and must never become a blocking rule.
* **0.3 address overlap** - among TRUE pairs, how much do the normalized
  addresses overlap (shared tokens, Jaccard, overlap coefficient)? Is there
  enough signal to score or block on?
* **0.4 token frequency and candidate census** - how large are the posting lists
  a token blocker would touch, do the true pairs whose names DIFFER still share
  a token rare enough to block on, and - over ALL 2.2M source1 entities, not
  just the matched ones - how many candidates would each entity generate, which
  entities would generate none at all, and what is the candidate-to-truth ratio
  that macro F0.5 is sensitive to.
* **0.5 zero-match entities** - the S1 entities with no true match at all. How
  many of them would an exact-name blocker propose candidates for anyway? Those
  are pure false positives, so this measures how dangerous the population is.

Design
------
``iter_prepared`` streams the normalized tables, ``GroundTruth`` supplies the
true pairs, ``np.searchsorted`` resolves ids to attributes, and the per-pair
string work is sharded across worker processes (pure CPU work - there is no GPU
path worth using for token comparison). There is no cross-product anywhere:
Phases 0.2-0.4 walk the O(n_true_pairs) pair list, and Phases 0.4/0.5 walk the
O(n_target) corpus once per source. Coverage is reported PAIR-weighted and
S1-ENTITY-weighted, because the challenge metric macro-averages over entities:
an entity whose pairs are all unreachable scores zero however few pairs it has.

The whole run is THREE streaming passes - source1, then source2, then source3 -
because one pass serves every phase: the token frequency counts, the zero-match
probe and the resolution of the matched targets' attributes all consume the same
chunk before it is discarded.

Memory
------
Peak RSS is dominated by two things (roughly 3-6GB on the full training set):
the token -> document-frequency counters (one per requested source, plus the
transient union used for the combined scope), and the resolved name/address/key
strings for the S1 entities and the matched targets. The counters hold COUNTS
only - never token -> entities, which would be ~100x larger. Nothing scales with
n_S1 x n_target. ``--sources source2`` roughly halves the peak. The candidate
census loads one exact-name index at a time (~250MB at S2 scale) and does not
materialize a single candidate pair.

Outputs (``<work_dir>/analysis`` unless ``--output-dir`` is given):

* ``blocking_statistics_report.json``  - all counts, machine-readable
* ``blocking_statistics_report.md``    - human-readable summary
* ``signal_coverage.csv``              - tidy: section, scope, signal, key, count, share
* ``candidate_census.csv``             - tidy: section, key, count, share, detail
* ``address_overlap.csv``              - tidy: scope, group, metric, statistic, value, unit
* ``token_frequency.csv``              - tidy: section=top_token|posting_cap, key, count, share
* ``zero_match_statistics.csv``        - tidy: section, key, count, share

No per-pair output is written - 7.6M rows would be a large artifact for no gain.

Usage::

    # HPC, full run
    python scripts/analyze_blocking_statistics.py --workers 16 --chunk-pairs 200000

    # skip the Phase 0.1 cross-tab (about half the pair-pass cost)
    python scripts/analyze_blocking_statistics.py --workers 16 --no-name-categories

    # local smoke test on a tiny synthetic fixture
    python scripts/analyze_blocking_statistics.py --config /tmp/smoke.yaml --split train

NOTE: the full run reads the whole training ground truth and the S2/S3 name and
address columns and is an HPC job. Do not run it on a laptop.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import logging
import math
import os
import sys
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.blocking import (  # noqa: E402
    BLOCKER_EXACT_NAME,
    ExactNameIndex,
    index_dir_for,
)
from src.data_loader import (  # noqa: E402
    describe_environment,
    iter_prepared,
    load_config,
    load_ground_truth,
    prepared_path,
)
from src.normalization import ADDRESS_NORM, NAME_KEY, NAME_NORM  # noqa: E402
from src.utils import (  # noqa: E402
    current_rss_bytes,
    decode_entity_id,
    detect_hardware,
    encode_entity_ids,
    fmt_int,
    format_hardware_report,
    human_bytes,
    log_memory,
    peak_rss_bytes,
    plan_inflight_window,
    read_json,
    resolve_workers,
    setup_logging,
    set_seed,
    stable_hash64,
    write_json,
)

LOG_NAME = "analyze_blocking_statistics"

SOURCE_CODES = {"source2": 2, "source3": 3}

SCRIPTS_DIR = Path(__file__).resolve().parent

# Columns resolved for the target entities that appear in the ground truth.
TARGET_COLUMNS = (NAME_NORM, ADDRESS_NORM)

# Cap for the token -> document-frequency counters. A full run stays far below
# this; the guard exists so a pathological corpus fails loudly instead of being
# OOM-killed.
MAX_UNIQUE_TOKENS = 40_000_000

# Candidate-posting caps to test in Phase 0.4. An IDF-filtered token blocker
# would drop tokens whose posting list is longer than some cap; the table says
# how many tokens each cap removes and how much candidate volume they carry.
POSTING_CAP_THRESHOLDS = (100, 500, 1_000, 5_000, 10_000, 50_000, 100_000)

# Rarity caps used for the "is at least one shared token usable for blocking?"
# question in Phase 0.4. The same numbers, a different question: the cap is
# applied per true pair, not to the corpus.
RARITY_CAP_THRESHOLDS = (100, 500, 1_000, 5_000, 10_000, 50_000)

# Cap used for the rough candidate-volume estimate in Phase 0.4.
CANDIDATE_VOLUME_CAP = 1_000

# Share of currently-available RAM the queued worker payloads may occupy, and the
# floor used when the machine's memory cannot be read. See _payload_budget_bytes.
PAYLOAD_BUDGET_FRACTION = 0.25
PAYLOAD_BUDGET_FLOOR = 64 * 1024 * 1024

# ---------------------------------------------------------------------------
# Phase 0.2: the four country-agnostic signals
# ---------------------------------------------------------------------------
# Every signal is a pure function of the two normalized records, so all four
# behave identically on a country/language/script the training split never saw.
# Nothing here reads the country column.
EXACT_SIGNAL = "exact_name"
TOKEN_SIGNAL = "rare_token_name"
CHAR_SIGNAL = "char_3gram_name"
ADDRESS_SIGNAL = "address_jaccard"
SIGNALS = (EXACT_SIGNAL, TOKEN_SIGNAL, CHAR_SIGNAL, ADDRESS_SIGNAL)

SIGNAL_LABELS = {
    EXACT_SIGNAL: "exact name_norm",
    TOKEN_SIGNAL: "rare name token",
    CHAR_SIGNAL: "char 3-gram",
    ADDRESS_SIGNAL: "address Jaccard",
}

# The operating point the joint/union tables are computed at. NOT a blocker
# choice: the sensitivity tables below report every threshold in the grids so
# the reader can see what a different choice would buy. These values are the
# ones Phase 0.3/0.4 already use for their headline numbers, so the tables agree
# with each other.
REFERENCE_TOKEN_CAP = CANDIDATE_VOLUME_CAP
REFERENCE_CHAR_JACCARD = 0.5
REFERENCE_ADDRESS_JACCARD = 0.5

# Threshold grids for the coverage-versus-cost sensitivity tables.
SIGNAL_TOKEN_CAP_GRID = RARITY_CAP_THRESHOLDS
SIGNAL_CHAR_JACCARD_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SIGNAL_ADDRESS_JACCARD_GRID = (0.2, 0.3, 0.5, 0.7, 0.8, 0.9)

# The "no signal reaches this pair" count is the headline number of Phase 0.2,
# so its 2^4-1 non-empty signal combinations are all reported. Named explicitly
# rather than generated, so the report labels cannot silently reorder.
COMBINATION_ORDER = (
    (EXACT_SIGNAL,),
    (TOKEN_SIGNAL,),
    (CHAR_SIGNAL,),
    (ADDRESS_SIGNAL,),
    (EXACT_SIGNAL, TOKEN_SIGNAL),
    (EXACT_SIGNAL, CHAR_SIGNAL),
    (EXACT_SIGNAL, ADDRESS_SIGNAL),
    (TOKEN_SIGNAL, CHAR_SIGNAL),
    (TOKEN_SIGNAL, ADDRESS_SIGNAL),
    (CHAR_SIGNAL, ADDRESS_SIGNAL),
    (EXACT_SIGNAL, TOKEN_SIGNAL, CHAR_SIGNAL),
    (EXACT_SIGNAL, TOKEN_SIGNAL, ADDRESS_SIGNAL),
    (EXACT_SIGNAL, CHAR_SIGNAL, ADDRESS_SIGNAL),
    (TOKEN_SIGNAL, CHAR_SIGNAL, ADDRESS_SIGNAL),
    SIGNALS,
)
NO_SIGNAL_LABEL = "none"

# Residue pairs listed in the report so the number has a face. They are the
# pairs no cheap signal reaches, i.e. exactly the ones an embedding model would
# have to rescue.
RESIDUE_EXAMPLES = 25

# ---------------------------------------------------------------------------
# Phase 0.4: candidate census over all source1 entities
# ---------------------------------------------------------------------------
# Entities per block in the census sweep. The census walks 2.2M entities and
# looks every name token up in the document-frequency table; blocking it keeps
# the temporary token-hash array bounded (~n * 6 tokens) instead of materializing
# ~11M hashes at once.
CENSUS_BLOCK = 200_000

# Entities listed in the "largest estimated candidate volume" table.
CENSUS_TOP_ENTITIES = 25

# Exact-name keys listed with the largest posting list. This is the duplicate /
# near-duplicate target question in its exact form.
CENSUS_TOP_KEYS = 25

# Distribution statistics reported for every per-pair metric.
QUANTILES: dict[str, float] = {
    "p10": 0.10,
    "p25": 0.25,
    "p50": 0.50,
    "p75": 0.75,
    "p90": 0.90,
    "p95": 0.95,
    "p99": 0.99,
}

# Token counts are bucketed to keep the histograms finite.
SHARED_NAME_TOKEN_MAX = 8
SHARED_ADDRESS_TOKEN_MAX = 8

# Candidate-count buckets for zero-match S1 entities, as (low, high) inclusive.
CANDIDATE_BUCKETS: tuple[tuple[int, int], ...] = ((1, 1), (2, 5), (6, 20), (21, 100), (101, 1_000))
CANDIDATE_BUCKET_OVERFLOW = 1_000

# Address-Jaccard bucket edges for the zero-match false-positive candidates, and
# the matching labels (one more label than edges: the first bucket is exactly 0).
# Bucket 0 is the interesting one: a candidate the exact-name blocker proposes
# with no address support at all.
ADDRESS_JACCARD_EDGES = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
ADDRESS_JACCARD_LABELS = (
    "0",
    "(0,0.1]",
    "(0.1,0.2]",
    "(0.2,0.3]",
    "(0.3,0.5]",
    "(0.5,0.7]",
    "(0.7,0.9]",
    "(0.9,1]",
)
assert len(ADDRESS_JACCARD_LABELS) == len(ADDRESS_JACCARD_EDGES) + 1

# Sentinel document frequency for a shared token missing from the corpus table.
# Never expected - every shared token of a true pair is by construction present
# in the target source - but a miss must not be mistaken for "no shared token".
UNKNOWN_DF = 1 << 62


# ---------------------------------------------------------------------------
# Phase 0.1 classifier (imported, never copied)
# ---------------------------------------------------------------------------
_PHASE01: Any = None


def _phase01():
    """Import ``scripts/analyze_name_differences.py`` (Phase 0.1) on first use.

    Imported rather than reimplemented so the categories the residue is
    cross-tabbed against can never drift from the ones Phase 0.1 reports. The
    import happens on first use inside whatever process needs it (the pair
    workers), so no process pays for it unless the cross-tab is enabled.
    """
    global _PHASE01
    if _PHASE01 is None:
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        import analyze_name_differences as module  # noqa: PLC0415 - deliberate lazy import

        _PHASE01 = module
    return _PHASE01


def _name_difference_categories() -> list[str]:
    """Phase 0.1's category names, in its precedence order."""
    return list(_phase01().CATEGORIES)


def _name_difference_reasons() -> list[str]:
    """Phase 0.1's ``unknown_other`` reason names."""
    return list(_phase01().UNKNOWN_REASONS)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _token_set(text: str) -> set[str]:
    """Distinct tokens of an already-normalized field.

    ``str.split()`` with no argument splits on any Unicode whitespace run and
    drops empty fields, so it is both Unicode-safe and the right semantics for a
    normalized field (the normalizer has already folded punctuation to spaces).

    A SET, not a list: Phase 0.4 counts DOCUMENT frequency, and one entity that
    repeats a token is still one document. A list would inflate every posting
    size by the repetition factor.
    """
    return set(text.split()) if text else set()


def _plural(count: int, noun: str) -> str:
    """``"3 pairs"`` / ``"1 pair"`` - reports are read by people."""
    return f"{fmt_int(count)} {noun}" + ("" if int(count) == 1 else "s")


def _pct(numerator: float, denominator: float) -> float:
    """Percentage, rounded, 0.0 when the denominator is empty."""
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def _ratio(value: float) -> float:
    return round(float(value), 6)


def _distribution(values: np.ndarray, mask: Optional[np.ndarray] = None) -> dict[str, Any]:
    """Count, mean, min, max and the QUANTILES of ``values`` (optionally masked)."""
    if mask is not None:
        values = values[mask]
    values = np.asarray(values)
    n = int(values.size)
    if n == 0:
        return {"n": 0}
    numeric = values.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        numeric = numeric[np.isfinite(numeric)]
        if numeric.size == 0:
            return {"n": n, "finite": 0}
    quantiles = np.quantile(numeric, list(QUANTILES.values()))
    out: dict[str, Any] = {
        "n": n,
        "mean": _ratio(numeric.mean()),
        "min": _ratio(numeric.min()),
        "max": _ratio(numeric.max()),
    }
    for index, label in enumerate(QUANTILES):
        out[label] = _ratio(quantiles[index])
    return out


def _histogram(counts: np.ndarray, top_label: str) -> dict[str, int]:
    """Finite histogram whose last bucket means ``top_label``."""
    out = {str(i): int(counts[i]) for i in range(len(counts) - 1) if counts[i]}
    if counts[-1]:
        out[top_label] = int(counts[-1])
    return out


def _bucket_counts(values: np.ndarray, buckets: tuple[tuple[int, int], ...]) -> dict[str, int]:
    """Count values into inclusive ``(low, high)`` buckets plus an overflow bucket."""
    out: dict[str, int] = {}
    for low, high in buckets:
        label = str(low) if low == high else f"{low}-{high}"
        out[label] = int(((values >= low) & (values <= high)).sum())
    out[f"{CANDIDATE_BUCKET_OVERFLOW}+"] = int((values > CANDIDATE_BUCKET_OVERFLOW).sum())
    return out


def _jaccard_bucket_index(value: float) -> int:
    """Index into the zero-match address Jaccard histogram.

    Bucket 0 is exactly zero overlap; bucket ``i`` is the half-open interval
    between edges ``i-1`` and ``i``, and the final bucket is ``(0.9, 1]``. A
    Jaccard value is always in [0, 1], so the histogram needs exactly
    ``len(ADDRESS_JACCARD_LABELS)`` slots.
    """
    if value <= 0.0:
        return 0
    for index, edge in enumerate(ADDRESS_JACCARD_EDGES, start=1):
        if value <= edge:
            return index
    return len(ADDRESS_JACCARD_EDGES)


def _labelled_histogram(counts: np.ndarray, labels: Sequence[str]) -> dict[str, int]:
    """Non-empty buckets of ``counts`` keyed by ``labels`` (which must align)."""
    return {label: int(counts[index]) for index, label in enumerate(labels) if counts[index]}


def _combine(left: Optional[np.ndarray], right: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Intersect two optional boolean masks, treating ``None`` as "everything"."""
    if left is None:
        return right
    if right is None:
        return left
    return left & right


def _trigram_jaccard(a: str, b: str) -> float:
    """Character 3-gram Jaccard, from Phase 0.1's implementation.

    Delegated to Phase 0.1 rather than reimplemented: the char-3-gram signal has
    to mean the same thing here as the "shared character content" check there,
    or the coverage figures and the category table would disagree about the same
    pair. Runs in the worker, so the call is one dict lookup per pair.
    """
    return _phase01()._trigram_jaccard(a, b)


# ---------------------------------------------------------------------------
# Column resolution (streaming, sized by what is needed - never by the corpus)
# ---------------------------------------------------------------------------
def _prepared_header(path: Path) -> list[str]:
    """Read just the header of a prepared TSV (handles .gz)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    return header.rstrip("\n").split("\t")


def _unique(columns: Sequence[str]) -> list[str]:
    """Deduplicate a column list, preserving order.

    ``iter_prepared`` forwards this to pandas ``usecols``, and passing the same
    column twice there is not something to rely on. The lists below overlap by
    construction (a requested column is also read for the token counts).
    """
    seen: set[str] = set()
    out: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            out.append(column)
    return out


def _string_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    """A column as an object array of plain ``str``, missing values as "".

    Going through pandas' ``string`` dtype rather than ``fillna("").astype(str)``
    matters: a column that is entirely empty is read as float64, and ``astype(str)``
    on that would silently turn every missing value into the string "0.0".
    """
    return frame[column].astype("string").fillna("").to_numpy(dtype=object)


def _resolve_s1_columns(
    config: dict,
    ground_truth,
    columns: Sequence[str],
    log: logging.Logger,
) -> dict[str, Any]:
    """Resolve columns for every ground-truth source1 entity, by GT row position.

    Indexing by ground-truth position makes the per-pair arrays a single gather
    (``values[owners]``) instead of a dictionary lookup per pair.

    Returns:
        ``{column: object array}`` for the requested columns that the prepared
        table actually has. Nothing is derived from the country column - the
        analysis is country-agnostic by construction.
    """
    entity_column = config.get("columns", {}).get("entity_id", "entity_id")
    header = _prepared_header(prepared_path(config, "train", "source1"))
    wanted = [column for column in columns if column in header]
    for column in columns:
        if column not in header:
            log.warning("source1 prepared table has no %r column; that metric is skipped", column)

    out: dict[str, Any] = {
        column: np.full(ground_truth.n_entities, "", dtype=object) for column in wanted
    }
    found = 0
    for chunk in iter_prepared(config, "train", "source1", columns=_unique([entity_column, *wanted])):
        positions = ground_truth.positions_of(chunk[entity_column])
        keep = positions >= 0
        if not keep.any():
            continue
        positions = positions[keep]
        for column in wanted:
            out[column][positions] = _string_column(chunk, column)[keep]
        found += int(keep.sum())
    log.info(
        "source1: resolved %s of %s ground-truth entities",
        fmt_int(found),
        fmt_int(ground_truth.n_entities),
    )
    return out


# ---------------------------------------------------------------------------
# Phase 0.5: the zero-match probe
# ---------------------------------------------------------------------------
class _ZeroMatchProbe:
    """Counts the candidates an exact-name blocker WOULD propose for no-match S1 rows.

    The exact-name blocker proposes a pair when the two normalized keys are
    equal. For the S1 entities whose ground truth is empty, every such proposal
    is a false positive by definition - so counting them measures how exposed the
    exact blocker is on the no-match population, and whether the address field
    would have rescued it.

    This replicates the blocker's exact-match semantics directly (a dictionary
    lookup keyed on the S1 names) rather than loading the persisted index: the
    result is identical, it needs no index build, and no second index would have
    to exist to answer the ``name_key`` variant. ``tests/test_blocking_statistics.py``
    checks the two agree on a fixture.
    """

    def __init__(
        self,
        entity_ids: np.ndarray,
        names: Optional[np.ndarray],
        keys: Optional[np.ndarray],
        addresses: Optional[np.ndarray],
        sources: Sequence[str],
    ) -> None:
        self.sources = list(sources)
        self.entity_ids = entity_ids
        self.names = names if names is not None else np.empty(0, dtype=object)
        self.n_zero = len(entity_ids)
        self.keys_available = keys is not None

        self.by_name = self._index(self.names) if self.n_zero else {}
        self.by_key = self._index(keys) if (self.n_zero and keys is not None) else {}
        self.address_tokens: list[set[str]] = [
            _token_set(text) for text in (addresses if addresses is not None else [])
        ]

        shape = (self.n_zero,)
        self.exact_norm: dict[int, np.ndarray] = {
            code: np.zeros(shape, dtype=np.int32) for code in SOURCE_CODES.values()
        }
        self.key_match: dict[int, np.ndarray] = {
            code: np.zeros(shape, dtype=np.int32) for code in SOURCE_CODES.values()
        }
        # Best address Jaccard over all exact-name candidates; -1 = no candidate.
        self.best_addr_jaccard: dict[int, np.ndarray] = {
            code: np.full(shape, -1.0, dtype=np.float32) for code in SOURCE_CODES.values()
        }
        self.jaccard_hist = np.zeros(len(ADDRESS_JACCARD_LABELS), dtype=np.int64)
        self.n_exact_pairs = 0
        self.n_exact_pairs_addr_both = 0
        self.n_exact_pairs_addr_zero_jaccard = 0
        self.n_exact_pairs_addr_missing = 0

    @staticmethod
    def _index(values: np.ndarray) -> dict[str, np.ndarray]:
        """Group compact entity indices by exact key, skipping empty keys.

        Empty keys are skipped because the real index skips them too: a record
        whose normalized name is empty carries no blocking signal.
        """
        buckets: dict[str, list[int]] = {}
        for index, value in enumerate(values):
            text = str(value)
            if not text:
                continue
            buckets.setdefault(text, []).append(index)
        return {key: np.array(items, dtype=np.int32) for key, items in buckets.items()}

    def observe(
        self,
        source: str,
        names: np.ndarray,
        keys: Optional[np.ndarray],
        addresses: np.ndarray,
    ) -> None:
        """Fold one streamed target chunk into the per-entity candidate counts."""
        if not self.n_zero:
            return
        code = SOURCE_CODES[source]
        exact = self.exact_norm[code]
        by_name = self.by_name
        by_key = self.by_key
        best = self.best_addr_jaccard[code]
        address_tokens = self.address_tokens
        histogram = self.jaccard_hist
        n_pairs = 0
        n_both = 0
        n_zero_jaccard = 0
        n_missing = 0

        for row in range(len(names)):
            name = names[row]
            if name:
                hits = by_name.get(name)
                if hits is not None and hits.size:
                    exact[hits] += 1
                    n_pairs += int(hits.size)
                    address = addresses[row]
                    target_tokens = _token_set(address) if address else set()
                    for index in hits.tolist():
                        tokens = address_tokens[index]
                        if tokens and target_tokens:
                            union = len(tokens | target_tokens)
                            value = len(tokens & target_tokens) / union if union else 0.0
                            if value > best[index]:
                                best[index] = value
                            histogram[_jaccard_bucket_index(value)] += 1
                            n_both += 1
                            if value <= 0.0:
                                n_zero_jaccard += 1
                        else:
                            n_missing += 1
            if keys is not None:
                key = keys[row]
                if key:
                    key_hits = by_key.get(key)
                    if key_hits is not None and key_hits.size:
                        self.key_match[code][key_hits] += 1

        self.n_exact_pairs += n_pairs
        self.n_exact_pairs_addr_both += n_both
        self.n_exact_pairs_addr_zero_jaccard += n_zero_jaccard
        self.n_exact_pairs_addr_missing += n_missing

    # -- reporting ---------------------------------------------------------
    def _key_totals(self) -> tuple[np.ndarray, np.ndarray]:
        codes = sorted(SOURCE_CODES.values())
        norm_total = np.zeros(self.n_zero, dtype=np.int32)
        key_total = np.zeros(self.n_zero, dtype=np.int32)
        for code in codes:
            norm_total += self.exact_norm[code]
            key_total += self.key_match[code]
        return norm_total, key_total

    def _best_over_sources(self) -> np.ndarray:
        codes = sorted(SOURCE_CODES.values())
        best = np.full(self.n_zero, -1.0, dtype=np.float32)
        for code in codes:
            np.maximum(best, self.best_addr_jaccard[code], out=best)
        return best

    @staticmethod
    def _split(counts: np.ndarray, n_zero: int) -> dict[str, Any]:
        has = counts > 0
        return {
            "n_entities": int(has.sum()),
            "pct_of_zero_match": _pct(int(has.sum()), n_zero),
            "candidate_count": _distribution(counts[has].astype(np.float64)),
            "buckets": _bucket_counts(counts[has], CANDIDATE_BUCKETS),
        }

    def summary(self) -> dict[str, Any]:
        """Per-source candidate counts, distributions and the false-positive risk table."""
        if not self.n_zero:
            return {"n_zero_match_entities": 0}
        norm_total, key_total = self._key_totals()
        key_only = key_total - norm_total

        per_source = {
            source: self._split(self.exact_norm[SOURCE_CODES[source]], self.n_zero)
            for source in self.sources
        }

        best = self._best_over_sources()
        with_candidates = norm_total > 0
        best_with = best[with_candidates]
        best_with = best_with[best_with >= 0]

        return {
            "n_zero_match_entities": int(self.n_zero),
            "keys_available": bool(self.keys_available),
            "by_candidate_kind": {
                "exact_name_norm": {
                    "n_entities": int((norm_total > 0).sum()),
                    "pct_of_zero_match": _pct(int((norm_total > 0).sum()), self.n_zero),
                },
                "name_key_only": {
                    "n_entities": int((key_only > 0).sum()),
                    "pct_of_zero_match": _pct(int((key_only > 0).sum()), self.n_zero),
                },
                "no_candidate_at_all": {
                    "n_entities": int((key_total == 0).sum()),
                    "pct_of_zero_match": _pct(int((key_total == 0).sum()), self.n_zero),
                },
            },
            "exact_name_norm": self._split(norm_total, self.n_zero),
            "name_key_total": self._split(key_total, self.n_zero),
            "per_source": per_source,
            "false_positive_risk": {
                "n_candidate_pairs": self.n_exact_pairs,
                "n_pairs_both_addresses_present": self.n_exact_pairs_addr_both,
                "n_pairs_with_zero_address_overlap": self.n_exact_pairs_addr_zero_jaccard,
                "pct_pairs_with_zero_address_overlap": _pct(
                    self.n_exact_pairs_addr_zero_jaccard, self.n_exact_pairs_addr_both
                ),
                "n_pairs_with_an_empty_address": self.n_exact_pairs_addr_missing,
                "address_jaccard_histogram": _labelled_histogram(
                    self.jaccard_hist, ADDRESS_JACCARD_LABELS
                ),
                "best_address_jaccard_per_entity": _distribution(best_with),
                "n_entities_with_any_address_support": int((best_with >= 0.5).sum()),
            },
            "top_entities_by_candidate_count": self._top_entities(norm_total),
            "note": (
                "Every candidate counted here is an EXACT normalized-name match - that is what "
                "the exact-name blocker proposes - so a name-similarity distribution over these "
                "pairs would be a spike at 1.0 and is not reported. The informative cheap signal "
                "is the address overlap, which IS reported: it separates candidates an address "
                "overlap rule would reject outright from ones that need a similarity threshold. "
                "Counts cover the target sources analysed in THIS run - see Run details - so a "
                "run restricted with --sources understates the exposure. No threshold is chosen "
                "here."
            ),
        }

    def _top_entities(self, norm_total: np.ndarray, limit: int = 25) -> list[dict[str, Any]]:
        """The no-match S1 entities with the most exact-name candidates."""
        if not self.n_zero:
            return []
        order = np.argsort(-norm_total, kind="stable")[:limit]
        best = self._best_over_sources()
        rows = []
        for index in order.tolist():
            total = int(norm_total[index])
            if total <= 0:
                break
            value = float(best[index])
            rows.append(
                {
                    "source1_entity_id": str(self.entity_ids[index]),
                    "name_norm": str(self.names[index]),
                    "n_candidates": total,
                    "n_candidates_s2": int(self.exact_norm[SOURCE_CODES["source2"]][index]),
                    "n_candidates_s3": int(self.exact_norm[SOURCE_CODES["source3"]][index]),
                    "best_address_jaccard": round(value, 6) if value >= 0 else None,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# One streaming pass over a target source (serves Phases 0.2, 0.3, 0.4, 0.5)
# ---------------------------------------------------------------------------
def _scan_target_source(
    config: dict,
    source: str,
    needed_codes: np.ndarray,
    token_counter: Counter,
    probe: _ZeroMatchProbe,
    log: logging.Logger,
) -> tuple[dict[str, Any], int]:
    """Stream one target source once and serve every phase that needs it.

    A single pass does three jobs, because reading 5M rows three times would cost
    three times the I/O for the same answer:

    1. accumulate the token -> document-frequency counts for Phase 0.4,
    2. feed the zero-match probe for Phase 0.5,
    3. resolve the attributes of the targets that appear in the ground truth,
       for Phases 0.2 and 0.3 (name, name_key and address).

    Returns:
        ``(resolved, rows_seen)`` - the resolved attribute arrays (sized by the
        number of needed codes, empty when this source contributes no pair) and
        the number of rows scanned, which is the entity count Phase 0.4 reports
        frequencies against. ``rows_seen`` also counts the tokens that had to be
        absent from the corpus, which is what makes an entity a structural zero.
    """
    entity_column = config.get("columns", {}).get("entity_id", "entity_id")
    header = _prepared_header(prepared_path(config, "train", source))

    resolved: dict[str, Any] = {}
    wanted = [column for column in TARGET_COLUMNS if column in header]
    for column in TARGET_COLUMNS:
        if column not in header:
            log.warning("%s prepared table has no %r column", source, column)

    has_key = NAME_KEY in header
    has_address = ADDRESS_NORM in header
    if not has_key and probe.n_zero:
        log.warning(
            "%s has no %s column; the zero-match name_key counts are unavailable",
            source,
            NAME_KEY,
        )

    columns = [entity_column, NAME_NORM, *wanted]
    if has_key:
        columns.append(NAME_KEY)
    columns = _unique(columns)

    use_resolution = len(needed_codes) > 0
    if use_resolution:
        for column in wanted:
            resolved[column] = np.full(len(needed_codes), "", dtype=object)
        if has_key:
            # Needed by the char-3-gram signal, which is computed on the
            # separator-free key so that "blue sky" and "bluesky" agree.
            resolved[NAME_KEY] = np.full(len(needed_codes), "", dtype=object)

    rows_seen = 0
    resolved_rows = 0
    for chunk in iter_prepared(config, "train", source, columns=columns):
        rows_seen += len(chunk)
        names = _string_column(chunk, NAME_NORM)
        addresses = (
            _string_column(chunk, ADDRESS_NORM) if has_address else np.full(len(chunk), "", dtype=object)
        )
        keys = _string_column(chunk, NAME_KEY) if has_key else None

        # -- Phase 0.4: token document frequency ---------------------------
        local = Counter()
        for name in names:
            local.update(_token_set(name))
        token_counter.update(local)
        if len(token_counter) > MAX_UNIQUE_TOKENS:
            raise SystemExit(
                f"token dictionary grew past {fmt_int(MAX_UNIQUE_TOKENS)} entries; "
                "the corpus does not look like business names - aborting rather than "
                "exhausting memory instead of producing a report"
            )

        # -- Phase 0.5: would an exact blocker propose candidates here? ------
        if probe.n_zero:
            probe.observe(source, names, keys, addresses)

        # -- Phases 0.2/0.3: attributes of the targets we truly match -------
        if use_resolution:
            codes = encode_entity_ids(chunk[entity_column])
            slots = np.searchsorted(needed_codes, codes)
            np.clip(slots, 0, max(len(needed_codes) - 1, 0), out=slots)
            hit = needed_codes[slots] == codes
            if hit.any():
                slots = slots[hit]
                for column in wanted:
                    resolved[column][slots] = _string_column(chunk, column)[hit]
                if has_key:
                    resolved[NAME_KEY][slots] = _string_column(chunk, NAME_KEY)[hit]
                resolved_rows += int(hit.sum())

        log.info(
            "  %s: scanned %s rows | %s unique tokens | %s rows matched",
            source,
            fmt_int(rows_seen),
            fmt_int(len(token_counter)),
            fmt_int(resolved_rows),
        )

    if use_resolution:
        log.info(
            "%s: resolved %s of %s needed target records",
            source,
            fmt_int(resolved_rows),
            fmt_int(len(needed_codes)),
        )
    return resolved, rows_seen


# ---------------------------------------------------------------------------
# Phases 0.3/0.4: per-pair string statistics, sharded
# ---------------------------------------------------------------------------
def _pair_statistics(
    payload: tuple[list[str], list[str], list[str], list[str], list[str], list[str]],
    classify_names: bool = True,
) -> dict[str, np.ndarray]:
    """Per-pair statistics for one chunk of true pairs.

    Runs in a worker process and returns a dict of aligned ``n``-length arrays:

    * ``name_counts`` ``int32[n, 3]`` - distinct S1 tokens, distinct target
      tokens, shared tokens.
    * ``addr_counts`` ``int32[n, 3]`` - the same for the normalized addresses.
    * ``name_equal`` ``uint8[n]`` - 1 where ``name_norm`` is byte-identical,
      i.e. exactly the pairs the exact-name blocker can see.
    * ``char_sim`` ``float32[n]`` - character 3-gram Jaccard of the two
      ``name_key`` values: Phase 0.2's third signal.
    * ``category`` / ``reason`` ``int16[n]`` - Phase 0.1's name-difference
      category and ``unknown_other`` reason, or -1 when the cross-tab is off.
    * ``shared_hashes`` ``uint64[sum(shared)]`` - hashes of the shared NAME
      tokens, packed per pair. Hashes rather than strings so the parent can look
      every shared token up in one vectorized ``searchsorted``; the parent owns
      the frequency table and the workers never need a copy of it.
    """
    names_a, names_b, keys_a, keys_b, addrs_a, addrs_b = payload
    n = len(names_a)
    name_counts = np.zeros((n, 3), dtype=np.int32)
    addr_counts = np.zeros((n, 3), dtype=np.int32)
    name_equal = np.zeros(n, dtype=np.uint8)
    char_sim = np.zeros(n, dtype=np.float32)
    categories = np.full(n, -1, dtype=np.int16)
    reasons = np.full(n, -1, dtype=np.int16)
    shared_parts: list[np.ndarray] = []
    classify_pair = _phase01().classify_pair if classify_names else None

    for row in range(n):
        name_a = names_a[row]
        name_b = names_b[row]
        tokens_a = _token_set(name_a)
        tokens_b = _token_set(name_b)
        name_counts[row, 0] = len(tokens_a)
        name_counts[row, 1] = len(tokens_b)
        if name_a == name_b:
            name_equal[row] = 1
        shared = tokens_a & tokens_b
        name_counts[row, 2] = len(shared)
        if shared:
            # sorted() for determinism: the hashes are packed per pair in a fixed
            # order, so aggregates do not depend on set iteration order.
            shared_parts.append(stable_hash64(sorted(shared)))

        char_sim[row] = _trigram_jaccard(keys_a[row], keys_b[row])

        if classify_pair is not None:
            category, reason, _, _ = classify_pair(name_a, name_b)
            categories[row] = category
            reasons[row] = reason

        address_a = addrs_a[row]
        address_b = addrs_b[row]
        addr_tokens_a = _token_set(address_a)
        addr_tokens_b = _token_set(address_b)
        addr_counts[row, 0] = len(addr_tokens_a)
        addr_counts[row, 1] = len(addr_tokens_b)
        addr_counts[row, 2] = len(addr_tokens_a & addr_tokens_b)

    return {
        "name_counts": name_counts,
        "addr_counts": addr_counts,
        "name_equal": name_equal,
        "char_sim": char_sim,
        "category": categories,
        "reason": reasons,
        "shared_hashes": (
            np.concatenate(shared_parts) if shared_parts else np.empty(0, dtype=np.uint64)
        ),
    }


def _iter_pair_chunks(
    names_a: np.ndarray,
    names_b: np.ndarray,
    keys_a: np.ndarray,
    keys_b: np.ndarray,
    addrs_a: np.ndarray,
    addrs_b: np.ndarray,
    n_pairs: int,
    chunk_pairs: int,
) -> Iterator[tuple[np.ndarray, tuple[list[str], list[str], list[str], list[str], list[str], list[str]]]]:
    """Yield ``(indices, payload)`` for each chunk of pairs.

    A fresh ``arange`` per chunk rather than one materialized index array: the
    caller must not hold 7.6M int64 just to slice what it already has.
    """
    for start in range(0, n_pairs, chunk_pairs):
        block = np.arange(start, min(start + chunk_pairs, n_pairs), dtype=np.int64)
        yield block, (
            names_a[block].tolist(),
            names_b[block].tolist(),
            keys_a[block].tolist(),
            keys_b[block].tolist(),
            addrs_a[block].tolist(),
            addrs_b[block].tolist(),
        )


def _resolve_workers(requested: int, configured: int, n_chunks: int, log: logging.Logger) -> int:
    """Decide the worker count: CLI > config > physical cores, clamped to the chunks.

    Delegates to :func:`src.utils.resolve_workers` so this script and Phase 0.1
    cannot drift apart. The auto default is the **physical** core count: this stage
    is bound by python string/token work rather than by floating-point units, so
    hyperthread siblings mostly add contention, and the previous hard cap of 8
    threw away most of a large HPC node.
    """
    return resolve_workers(requested, configured, n_chunks, logger=log, label="pair workers")


def _estimate_payload_bytes_per_pair(
    names_a: np.ndarray,
    names_b: np.ndarray,
    keys_a: np.ndarray,
    keys_b: np.ndarray,
    addrs_a: np.ndarray,
    addrs_b: np.ndarray,
    sample: int = 2_000,
) -> int:
    """Rough bytes one pair costs once queued as a worker payload.

    Only used to size the in-flight window, so a sample is enough; the point is to
    catch the case where a large worker count multiplied by a large chunk size
    queues gigabytes of strings. Each string is charged its character width plus
    the ~49-byte python object header, which is what actually lands in the pickled
    payload.

    Returns:
        Estimated bytes per pair, never below 1.
    """
    n = len(names_a)
    if n == 0:
        return 1
    step = max(1, n // sample)
    index = slice(None, None, step)
    arrays = (names_a, names_b, keys_a, keys_b, addrs_a, addrs_b)
    total = 0
    counted = 0
    for array in arrays:
        values = array[index]
        if len(values) == 0:
            continue
        total += sum(len(value) + 49 for value in values)
        counted += len(values)
    if counted == 0:
        return 1
    # ``counted`` covers all six fields, so scale back up to one pair.
    return max(1, int(total * len(arrays) / counted))


def _payload_budget_bytes(config: dict, hardware: dict) -> int:
    """RAM the in-flight worker payloads may occupy.

    A fraction of what is free *right now* rather than a fixed number: the report
    arrays and the ground truth are already resident by the time the pool starts,
    so the budget has to be measured against the machine as it currently stands.
    ``compute.payload_budget_bytes`` pins it when a site wants to.
    """
    pinned = config.get("compute", {}).get("payload_budget_bytes")
    if pinned:
        return max(1, int(pinned))
    available = hardware.get("ram_available") or hardware.get("ram_total") or 0
    if not available:
        return PAYLOAD_BUDGET_FLOOR
    return max(PAYLOAD_BUDGET_FLOOR, min(int(available * PAYLOAD_BUDGET_FRACTION), int(available)))


# ---------------------------------------------------------------------------
# Phase 0.2: signal coverage and complementarity
# ---------------------------------------------------------------------------
def _address_jaccard(addr_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(both_addresses_present, jaccard)`` from the per-pair address token counts.

    One definition, used by both the address phase and the coverage phase, so the
    address signal and the address report can never disagree about a pair.
    """
    tokens_a = addr_counts[:, 0].astype(np.int64)
    tokens_b = addr_counts[:, 1].astype(np.int64)
    shared = addr_counts[:, 2].astype(np.int64)
    both_present = (tokens_a > 0) & (tokens_b > 0)
    union = tokens_a + tokens_b - shared
    jaccard = np.divide(
        shared, union, out=np.zeros(len(shared), dtype=np.float64), where=union > 0
    )
    return both_present, jaccard


def _signal_coverage(
    name_equal: np.ndarray,
    min_df: np.ndarray,
    char_sim: np.ndarray,
    addr_counts: np.ndarray,
    categories: np.ndarray,
    reasons: np.ndarray,
    categories_available: bool,
    source_of_pair: np.ndarray,
    owners: np.ndarray,
    entity_ids: np.ndarray,
    target_codes: np.ndarray,
    names_a: np.ndarray,
    names_b: np.ndarray,
    n_entities: int,
    sources: Sequence[str],
    log: logging.Logger,
) -> dict[str, Any]:
    """Which signals reach each true pair, jointly, plus what escapes all of them.

    The four signals are pure functions of the two normalized records - none of
    them reads a country - so a signal that covers the training population
    behaves the same way on a country or script the training split never saw.

    The pair list here is exactly the TRUE pairs, so every figure is a coverage
    (recall) figure. Nothing in this function says anything about precision:
    that needs candidate generation, which is Phase 1.
    """
    n_pairs = len(name_equal)
    both_present, address_jaccard = _address_jaccard(addr_counts)

    masks: dict[str, np.ndarray] = {
        # name_norm byte-identical: exactly what the shipped exact blocker sees.
        EXACT_SIGNAL: name_equal.astype(bool),
        # Shares a name token rare enough to be worth probing. min_df > 0 is
        # equivalent to "shares at least one name token": a shared token is by
        # construction present in the target corpus, so its df is >= 1.
        TOKEN_SIGNAL: (min_df > 0) & (min_df <= REFERENCE_TOKEN_CAP),
        CHAR_SIGNAL: char_sim >= np.float32(REFERENCE_CHAR_JACCARD),
        ADDRESS_SIGNAL: both_present & (address_jaccard >= REFERENCE_ADDRESS_JACCARD),
    }
    reached_by_any = np.zeros(n_pairs, dtype=bool)
    for mask in masks.values():
        reached_by_any |= mask
    residue = ~reached_by_any

    pairs_per_entity = np.bincount(owners, minlength=n_entities)
    with_pairs = pairs_per_entity > 0
    n_with_pairs = int(with_pairs.sum())

    def _entity_counts(mask: np.ndarray) -> np.ndarray:
        """How many of each entity's analysed pairs the mask reaches."""
        return np.bincount(owners[mask], minlength=n_entities)

    def _coverage(mask: np.ndarray) -> dict[str, Any]:
        n = int(mask.sum())
        n_entities_hit = int((_entity_counts(mask) > 0).sum())
        return {
            "n_pairs": n,
            "pct_of_pairs": _pct(n, n_pairs),
            "n_entities": n_entities_hit,
            "pct_of_entities_with_pairs": _pct(n_entities_hit, n_with_pairs),
        }

    # -- individual signals -------------------------------------------------
    signals = {
        signal: {
            **_coverage(mask),
            "label": SIGNAL_LABELS[signal],
            "reference_threshold": {
                EXACT_SIGNAL: None,
                TOKEN_SIGNAL: REFERENCE_TOKEN_CAP,
                CHAR_SIGNAL: REFERENCE_CHAR_JACCARD,
                ADDRESS_SIGNAL: REFERENCE_ADDRESS_JACCARD,
            }[signal],
        }
        for signal, mask in masks.items()
    }

    # -- union, and the combinations that make it up ------------------------
    by_combination: list[dict[str, Any]] = []
    for combo in COMBINATION_ORDER:
        mask = np.ones(n_pairs, dtype=bool)
        for signal in SIGNALS:
            mask &= masks[signal] if signal in combo else ~masks[signal]
        entry = _coverage(mask)
        entry["signals"] = "+".join(combo)
        by_combination.append(entry)
    residue_coverage = _coverage(residue)
    residue_coverage["signals"] = NO_SIGNAL_LABEL
    by_combination.append(residue_coverage)

    # -- entity outcomes (what macro F0.5 actually averages over) -----------
    reached_per_entity = _entity_counts(reached_by_any)
    all_reached = with_pairs & (reached_per_entity == pairs_per_entity)
    none_reached = with_pairs & (reached_per_entity == 0)
    partial = with_pairs & ~all_reached & ~none_reached

    # -- per source --------------------------------------------------------
    per_source: dict[str, Any] = {}
    for source in sources:
        mask_source = source_of_pair == SOURCE_CODES[source]
        n_source_pairs = int(mask_source.sum())
        entry: dict[str, Any] = {
            "n_pairs": n_source_pairs,
            "signals": {
                signal: {
                    "n_pairs": int((signal_mask & mask_source).sum()),
                    "pct_of_pairs": _pct(int((signal_mask & mask_source).sum()), n_source_pairs),
                }
                for signal, signal_mask in masks.items()
            },
        }
        union_source = reached_by_any & mask_source
        entry["union"] = {
            "n_pairs": int(union_source.sum()),
            "pct_of_pairs": _pct(int(union_source.sum()), n_source_pairs),
        }
        residue_source = residue & mask_source
        entry["residue"] = {
            "n_pairs": int(residue_source.sum()),
            "pct_of_pairs": _pct(int(residue_source.sum()), n_source_pairs),
        }
        per_source[source] = entry

    # -- coverage versus threshold-versus-cost -----------------------------
    def _token_rows() -> list[dict[str, Any]]:
        rows = []
        for cap in SIGNAL_TOKEN_CAP_GRID:
            mask = (min_df > 0) & (min_df <= cap)
            n = int(mask.sum())
            postings = min_df[mask]
            rows.append(
                {
                    "threshold": cap,
                    "n_pairs": n,
                    "pct_of_pairs": _pct(n, n_pairs),
                    # The token blocker probes one posting list per pair, so the
                    # sum of the probed list sizes IS the pre-deduplication
                    # candidate volume. This is a real cost, not an estimate.
                    "candidate_cost": {
                        "kind": "posting_list",
                        "n_candidates": int(postings.sum()),
                        "mean_per_pair": _ratio(float(postings.mean())) if n else 0.0,
                        "max_per_pair": int(postings.max()) if n else 0,
                    },
                }
            )
        return rows

    def _threshold_rows(grid: Sequence[float], mask_of) -> list[dict[str, Any]]:
        rows = []
        for threshold in grid:
            mask = mask_of(threshold)
            n = int(mask.sum())
            rows.append(
                {
                    "threshold": threshold,
                    "n_pairs": n,
                    "pct_of_pairs": _pct(n, n_pairs),
                    "candidate_cost": {
                        "kind": "not_estimable_in_phase_0",
                        "n_candidates": None,
                        "mean_per_pair": None,
                        "max_per_pair": None,
                    },
                }
            )
        return rows

    sensitivity = {
        TOKEN_SIGNAL: _token_rows(),
        CHAR_SIGNAL: _threshold_rows(
            SIGNAL_CHAR_JACCARD_GRID, lambda t: char_sim >= np.float32(t)
        ),
        # Address Jaccard is a FILTER, not a generator: it is applied to pairs
        # some other signal already proposed, so it can only remove candidates.
        # Its cost column therefore says "adds none" rather than "unknown".
        ADDRESS_SIGNAL: [
            {**row, "candidate_cost": {"kind": "filter", "n_candidates": 0,
                                       "mean_per_pair": 0.0, "max_per_pair": 0}}
            for row in _threshold_rows(
                SIGNAL_ADDRESS_JACCARD_GRID, lambda t: both_present & (address_jaccard >= t)
            )
        ],
    }

    # -- the residue: what no cheap signal reaches -------------------------
    residue_report: dict[str, Any] = {
        "n_pairs": int(residue.sum()),
        "pct_of_pairs": _pct(int(residue.sum()), n_pairs),
        "n_entities": int(_coverage(residue)["n_entities"]),
        "definition": (
            "no cheap lexical signal reaches the pair: the two names are not equal, "
            "share no name token, have character 3-gram Jaccard below "
            f"{REFERENCE_CHAR_JACCARD}, and the addresses do not overlap at "
            f"Jaccard {REFERENCE_ADDRESS_JACCARD}"
        ),
    }
    if categories_available:
        category_names = _name_difference_categories()
        counts = np.bincount(categories[residue], minlength=len(category_names))
        residue_report["by_name_difference_category"] = {
            name: {"n_pairs": int(counts[index]), "pct_of_residue": _pct(int(counts[index]), int(residue.sum()))}
            for index, name in enumerate(category_names)
            if counts[index]
        }
        reason_names = _name_difference_reasons()
        reason_counts = np.bincount(reasons[residue], minlength=len(reason_names))
        residue_report["by_name_difference_reason"] = {
            name: {"n_pairs": int(reason_counts[index]), "pct_of_residue": _pct(int(reason_counts[index]), int(residue.sum()))}
            for index, name in enumerate(reason_names)
            if reason_counts[index] and name != "not_applicable"
        }
        # Script difference is read off Phase 0.1's transliteration_script
        # category, which is assigned before any distance metric runs.
        transliteration = _phase01().CATEGORY_INDEX["transliteration_script"]
        n_different_script = int((categories[residue] == transliteration).sum())
        residue_report["by_script_relation"] = {
            "different_script": n_different_script,
            "same_script": int(residue.sum()) - n_different_script,
        }
    else:
        residue_report["by_name_difference_category"] = {}
        residue_report["by_name_difference_reason"] = {}
        residue_report["by_script_relation"] = {}
        residue_report["note"] = "name-category cross-tab disabled (--no-name-categories)"

    residue_report["per_source"] = {
        source: {
            "n_pairs": per_source[source]["residue"]["n_pairs"],
            "pct_of_pairs": per_source[source]["residue"]["pct_of_pairs"],
        }
        for source in sources
    }
    examples = []
    for index in np.flatnonzero(residue)[:RESIDUE_EXAMPLES].tolist():
        row: dict[str, Any] = {
            "source1_entity_id": str(entity_ids[owners[index]]),
            "target_entity_id": str(decode_entity_id(int(target_codes[index]))),
            "source": next(
                (name for name, code in SOURCE_CODES.items() if code == int(source_of_pair[index])),
                "",
            ),
            "name_norm_source1": str(names_a[index]),
            "name_norm_target": str(names_b[index]),
        }
        if categories_available:
            category_names = _name_difference_categories()
            row["name_difference_category"] = category_names[int(categories[index])]
            row["name_difference_reason"] = _name_difference_reasons()[int(reasons[index])]
        examples.append(row)
    residue_report["examples"] = examples

    # -- coverage sliced by Phase 0.1's categories -------------------------
    by_category: dict[str, Any] = {}
    if categories_available:
        for index, name in enumerate(_name_difference_categories()):
            mask = categories == index
            total = int(mask.sum())
            if not total:
                continue
            by_category[name] = {
                "n_pairs": total,
                "signals": {
                    signal: {
                        "n_pairs": int((signal_mask & mask).sum()),
                        "pct_of_category": _pct(int((signal_mask & mask).sum()), total),
                    }
                    for signal, signal_mask in masks.items()
                },
                "union": {
                    "n_pairs": int((reached_by_any & mask).sum()),
                    "pct_of_category": _pct(int((reached_by_any & mask).sum()), total),
                },
                "residue": {
                    "n_pairs": int((residue & mask).sum()),
                    "pct_of_category": _pct(int((residue & mask).sum()), total),
                },
            }

    log.info(
        "0.2 coverage: exact %.2f%%, token %.2f%%, char %.2f%%, address %.2f%%; "
        "union %.2f%%, residue %s pairs (%.2f%%)",
        signals[EXACT_SIGNAL]["pct_of_pairs"],
        signals[TOKEN_SIGNAL]["pct_of_pairs"],
        signals[CHAR_SIGNAL]["pct_of_pairs"],
        signals[ADDRESS_SIGNAL]["pct_of_pairs"],
        _pct(int(reached_by_any.sum()), n_pairs),
        fmt_int(int(residue.sum())),
        residue_report["pct_of_pairs"],
    )

    return {
        "reference_point": {
            TOKEN_SIGNAL: REFERENCE_TOKEN_CAP,
            CHAR_SIGNAL: REFERENCE_CHAR_JACCARD,
            ADDRESS_SIGNAL: REFERENCE_ADDRESS_JACCARD,
        },
        "signal_definitions": {
            EXACT_SIGNAL: f"{NAME_NORM} byte-identical",
            TOKEN_SIGNAL: f"shares a {NAME_NORM} token with document frequency <= cap",
            CHAR_SIGNAL: f"character 3-gram Jaccard of {NAME_KEY} >= threshold",
            ADDRESS_SIGNAL: f"address-token Jaccard >= threshold, both addresses non-empty",
        },
        "n_pairs": n_pairs,
        "n_source1_entities_with_analysed_pairs": n_with_pairs,
        "signals": signals,
        "union": {
            **_coverage(reached_by_any),
            "by_combination": by_combination,
            "entity_outcomes": {
                "n_entities": n_with_pairs,
                "all_pairs_reached": {
                    "n_entities": int(all_reached.sum()),
                    "pct": _pct(int(all_reached.sum()), n_with_pairs),
                },
                "some_pairs_reached": {
                    "n_entities": int(partial.sum()),
                    "pct": _pct(int(partial.sum()), n_with_pairs),
                },
                "no_pair_reached": {
                    "n_entities": int(none_reached.sum()),
                    "pct": _pct(int(none_reached.sum()), n_with_pairs),
                },
                "definition": (
                    "per source1 entity, over the pairs analysed in this run: every pair "
                    "reached by the union, some but not all, or none. 'no pair reached' "
                    "entities score zero on a blocking-limited metric however many true "
                    "pairs they have, which is why the entity view is reported next to "
                    "the pair view."
                ),
            },
        },
        "per_source": per_source,
        "sensitivity": sensitivity,
        "residue": residue_report,
        "coverage_by_name_difference_category": by_category,
        "note": (
            "Coverage is measured on TRUE pairs, so it is a RECALL ceiling, not a "
            "precision estimate: a signal can reach every true pair and still be "
            "useless if its posting lists are enormous. That is why every threshold is "
            "reported with its candidate cost, and why the census in Phase 0.4 measures "
            "the volume over all source1 entities. The reference point is a reading "
            "position for the tables, not a blocker configuration - no threshold is "
            "chosen by this script."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 0.3: address overlap
# ---------------------------------------------------------------------------
def _address_overlap(
    addr_counts: np.ndarray,
    source_of_pair: np.ndarray,
    name_equal: np.ndarray,
    sources: Sequence[str],
) -> dict[str, Any]:
    """Per-pair address token overlap, sliced by source and by name equality.

    The cross-tab is split on whether the two names are byte-identical, because
    that is the one split that changes what the address figure MEANS: when the
    names already agree the exact blocker has proposed the pair and the address
    is free confirmation, whereas when the names differ the address is the only
    thing left that could confirm it. (It used to be split on country agreement;
    country is not a blocking signal and is not available for unseen test
    countries, so the split that survives is the one that describes the actual
    decision the pipeline has to make.)
    """
    tokens_a = addr_counts[:, 0].astype(np.int64)
    tokens_b = addr_counts[:, 1].astype(np.int64)
    shared = addr_counts[:, 2].astype(np.int64)
    n = len(shared)
    both_present, jaccard = _address_jaccard(addr_counts)
    minimum = np.minimum(tokens_a, tokens_b)
    overlap = np.divide(shared, minimum, out=np.zeros(n, dtype=np.float64), where=minimum > 0)

    groups: dict[str, Optional[np.ndarray]] = {
        "both_addresses_present": both_present,
        "name_norm_identical": name_equal.astype(bool),
        "name_norm_differs": ~name_equal.astype(bool),
    }

    def _slice(mask: Optional[np.ndarray]) -> dict[str, Any]:
        n_pairs = n if mask is None else int(mask.sum())

        def _d(values: np.ndarray, extra: Optional[np.ndarray] = None) -> dict[str, Any]:
            return _distribution(values, _combine(mask, extra))

        counts = {
            "n_pairs": n_pairs,
            "n_both_addresses_present": int(_combine(both_present, mask).sum()),
            "n_either_address_missing": int(_combine(~both_present, mask).sum()),
            "n_zero_shared_tokens": int(_combine(shared == 0, mask).sum()),
            "n_at_least_1_shared_token": int(_combine(shared >= 1, mask).sum()),
            "n_at_least_2_shared_tokens": int(_combine(shared >= 2, mask).sum()),
            "n_at_least_3_shared_tokens": int(_combine(shared >= 3, mask).sum()),
            "n_jaccard_at_least_0_5": int(_combine(both_present & (jaccard >= 0.5), mask).sum()),
            "n_jaccard_at_least_0_8": int(_combine(both_present & (jaccard >= 0.8), mask).sum()),
            "n_overlap_coefficient_at_least_0_8": int(
                _combine(both_present & (overlap >= 0.8), mask).sum()
            ),
            "n_overlap_coefficient_equal_1": int(
                _combine(both_present & (overlap >= 1.0), mask).sum()
            ),
        }
        comparable = counts["n_both_addresses_present"]
        return {
            "source1_address_tokens": _d(tokens_a),
            "target_address_tokens": _d(tokens_b),
            "shared_address_tokens": _d(shared),
            "jaccard": _d(jaccard, both_present),
            "overlap_coefficient": _d(overlap, both_present),
            "counts": counts,
            "pct": {
                "zero_shared_tokens_of_comparable": _pct(counts["n_zero_shared_tokens"], comparable),
                "jaccard_at_least_0_5_of_comparable": _pct(
                    counts["n_jaccard_at_least_0_5"], comparable
                ),
                "overlap_coefficient_at_least_0_8_of_comparable": _pct(
                    counts["n_overlap_coefficient_at_least_0_8"], comparable
                ),
            },
        }

    slices: dict[str, Any] = {"all": _slice(None)}
    slices["all"]["by_group"] = {
        name: _slice(group) for name, group in groups.items()
    }
    for source in sources:
        mask = source_of_pair == SOURCE_CODES[source]
        entry = _slice(mask)
        entry["by_group"] = {name: _slice(_combine(mask, group)) for name, group in groups.items()}
        slices[source] = entry

    shared_histogram = np.bincount(
        np.minimum(shared, SHARED_ADDRESS_TOKEN_MAX), minlength=SHARED_ADDRESS_TOKEN_MAX + 1
    )
    return {
        "slices": slices,
        "shared_token_histogram": _histogram(shared_histogram, f"{SHARED_ADDRESS_TOKEN_MAX}+"),
        "note": (
            "Shared tokens are counted, not weighted: a token like 'road' or 'street' is "
            "shared by most addresses and carries almost no blocking signal, so these "
            "figures are an upper bound on the useful overlap. Phase 0.4 supplies the "
            "frequency needed to weight them."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 0.4: token frequency and pair rarity
# ---------------------------------------------------------------------------
def _posting_summary(
    tokens: list[str],
    document_frequency: np.ndarray,
    n_entities: int,
    top_tokens: int,
) -> dict[str, Any]:
    """Posting-size distribution, cap table and the most frequent tokens."""
    n_unique = len(tokens)
    if not n_unique:
        return {"n_unique_tokens": 0, "n_entities_in_scope": n_entities}
    total_postings = int(document_frequency.sum())

    cap_rows = []
    for cap in POSTING_CAP_THRESHOLDS:
        over = document_frequency > cap
        n_over = int(over.sum())
        postings_over = int(document_frequency[over].sum())
        cap_rows.append(
            {
                "cap": cap,
                "n_tokens_over_cap": n_over,
                "pct_tokens_over_cap": _pct(n_over, n_unique),
                "n_postings_over_cap": postings_over,
                "pct_postings_over_cap": _pct(postings_over, total_postings),
            }
        )

    # heapq.nlargest rather than a full sort: the table can hold millions of
    # entries and only a few hundred are reported. Ties are broken by token so
    # the output is deterministic.
    order = heapq.nlargest(top_tokens, range(n_unique), key=document_frequency.__getitem__)
    order.sort(key=lambda index: (-int(document_frequency[index]), tokens[index]))
    top_rows = [
        {
            "rank": rank,
            "token": tokens[index],
            "document_frequency": int(document_frequency[index]),
            "pct_of_entities": _pct(int(document_frequency[index]), n_entities),
            # IDF-like: log(N / df). Reported for reference only: the blocker
            # design is not settled and this is not a tuned weight.
            "idf": (
                round(math.log(n_entities / document_frequency[index]), 6)
                if document_frequency[index] > 0 and n_entities
                else None
            ),
        }
        for rank, index in enumerate(order, start=1)
    ]

    return {
        "n_entities_in_scope": n_entities,
        "n_unique_tokens": n_unique,
        "n_token_occurrences": total_postings,
        "postings_per_entity": _ratio(total_postings / n_entities) if n_entities else None,
        "posting_size": {
            "mean": _ratio(float(document_frequency.mean())),
            "median": _ratio(float(np.median(document_frequency))),
            "min": int(document_frequency.min()),
            "max": int(document_frequency.max()),
            **{
                label: _ratio(float(np.quantile(document_frequency, quantile)))
                for label, quantile in QUANTILES.items()
            },
            "p99_9": _ratio(float(np.quantile(document_frequency, 0.999))),
        },
        "posting_caps": cap_rows,
        "top_tokens": top_rows,
    }


def _token_frequency(
    counters: dict[str, Counter],
    rows_per_source: dict[str, int],
    top_tokens: int,
    log: logging.Logger,
) -> tuple[dict[str, dict[str, Any]], np.ndarray, np.ndarray]:
    """Per-source and combined token statistics, plus the hash table for lookups.

    Returns:
        ``(per_scope, df_hashes, df_values)`` where ``df_hashes`` (sorted uint64)
        and ``df_values`` are the combined document frequency of every token in
        the analysed sources - the table the pair-rarity lookup needs. Every
        shared token of a true pair is in it by construction: the token is in an
        S1 name and in a matched target name, and every matched target lives in
        one of the analysed sources.
    """
    per_scope: dict[str, dict[str, Any]] = {}
    for source, counter in counters.items():
        tokens = list(counter.keys())
        frequency = np.fromiter(counter.values(), dtype=np.int64, count=len(tokens))
        per_scope[source] = _posting_summary(
            tokens, frequency, rows_per_source.get(source, 0), top_tokens
        )

    if len(counters) == 1:
        counter = next(iter(counters.values()))
        tokens = list(counter.keys())
        combined = np.fromiter(counter.values(), dtype=np.int64, count=len(tokens))
    else:
        # The union holds references to the same string objects, so it costs a
        # set of pointers rather than a second copy of every token. A third
        # Counter would be ~2x the memory of the sources it summarises.
        union: set[str] = set()
        for counter in counters.values():
            union.update(counter.keys())
        tokens = list(union)
        counters_list = list(counters.values())
        combined = np.fromiter(
            (sum(counter.get(token, 0) for counter in counters_list) for token in tokens),
            dtype=np.int64,
            count=len(tokens),
        )
    per_scope["all"] = _posting_summary(
        tokens, combined, sum(rows_per_source.values()), top_tokens
    )

    log.info("hashing %s unique tokens for the posting lookup table", fmt_int(len(tokens)))
    hashes = stable_hash64(tokens)
    order = np.argsort(hashes, kind="stable")
    df_hashes = hashes[order]
    df_values = combined[order]
    return per_scope, df_hashes, df_values


def _pair_rarity(
    name_counts: np.ndarray,
    name_equal: np.ndarray,
    min_df: np.ndarray,
    source_of_pair: np.ndarray,
    sources: Sequence[str],
    log: logging.Logger,
) -> dict[str, Any]:
    """Do true pairs share a name token rare enough to be worth blocking on?

    ``min_df[i]`` is the document frequency of the RAREST shared name token of
    pair ``i`` (0 when the pair shares no token at all). "At least one shared
    token with df <= cap" is therefore exactly ``min_df <= cap``, so one integer
    per pair answers the question for every cap at once.
    """
    n_pairs = len(min_df)
    has_shared = name_counts[:, 2] > 0
    unknown = int((min_df[has_shared] > UNKNOWN_DF // 2).sum()) if n_pairs else 0
    if unknown:
        log.warning(
            "%s shared tokens were missing from the frequency table and are treated as "
            "unusably frequent",
            fmt_int(unknown),
        )

    differing = ~name_equal
    subsets: dict[str, np.ndarray] = {
        "all_true_pairs": np.ones(n_pairs, dtype=bool),
        "name_norm_differs": differing,
        "name_norm_identical": ~differing,
    }
    for source in sources:
        subsets[f"name_norm_differs_{source}"] = differing & (source_of_pair == SOURCE_CODES[source])

    def _usability(mask: np.ndarray) -> list[dict[str, Any]]:
        total = int(mask.sum())
        rows = []
        for cap in RARITY_CAP_THRESHOLDS:
            usable = int((mask & has_shared & (min_df <= cap) & (min_df > 0)).sum())
            rows.append(
                {
                    "cap": cap,
                    "n_pairs_with_usable_token": usable,
                    "pct_of_subset": _pct(usable, total),
                    "n_pairs_missed": total - usable,
                    "pct_missed": _pct(total - usable, total),
                }
            )
        return rows

    def _min_df_dist(mask: np.ndarray) -> dict[str, Any]:
        usable = mask & has_shared & (min_df > 0) & (min_df <= UNKNOWN_DF // 2)
        if not usable.any():
            return {"n": 0}
        return _distribution(min_df[usable].astype(np.float64))

    def _candidate_volume(mask: np.ndarray, cap: int) -> dict[str, Any]:
        """Rough candidate volume a per-S1 rarest-token blocker would generate.

        Each S1 probes the posting list of its rarest usable shared token, so
        summing those posting sizes over the subset is an upper bound on the
        candidate rows before deduplication.
        """
        usable = mask & has_shared & (min_df <= cap) & (min_df > 0)
        if not usable.any():
            return {"n_pairs": 0, "estimated_candidates": 0}
        postings = min_df[usable]
        return {
            "n_pairs": int(usable.sum()),
            "estimated_candidates": int(postings.sum()),
            "mean_postings_per_pair": _ratio(float(postings.mean())),
            "max_postings_per_pair": int(postings.max()),
        }

    shared_histogram = np.bincount(
        np.minimum(name_counts[:, 2], SHARED_NAME_TOKEN_MAX), minlength=SHARED_NAME_TOKEN_MAX + 1
    )
    differing_histogram = np.bincount(
        np.minimum(name_counts[differing][:, 2], SHARED_NAME_TOKEN_MAX),
        minlength=SHARED_NAME_TOKEN_MAX + 1,
    )

    return {
        "subset_sizes": {name: int(mask.sum()) for name, mask in subsets.items()},
        "has_shared_token_at_all": {
            name: {
                "n_pairs": int((mask & has_shared).sum()),
                "pct_of_subset": _pct(int((mask & has_shared).sum()), int(mask.sum())),
            }
            for name, mask in subsets.items()
        },
        "shared_name_token_histogram": {
            "all_true_pairs": _histogram(shared_histogram, f"{SHARED_NAME_TOKEN_MAX}+"),
            "name_norm_differs": _histogram(differing_histogram, f"{SHARED_NAME_TOKEN_MAX}+"),
        },
        "rarest_shared_token_df": {name: _min_df_dist(mask) for name, mask in subsets.items()},
        "usability_by_cap": {name: _usability(mask) for name, mask in subsets.items()},
        "candidate_volume_at_cap": {
            name: _candidate_volume(mask, CANDIDATE_VOLUME_CAP) for name, mask in subsets.items()
        },
        "candidate_volume_cap": CANDIDATE_VOLUME_CAP,
        "note": (
            "A pair with no shared name token is unreachable by a token blocker no matter "
            "how the cap is set; 'usable' means it shares at least one token whose posting "
            "list is short enough to probe. This is measurement: no blocker is implemented "
            "and no cap is chosen here."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 0.4: the candidate-volume / reachability census over ALL source1 rows
# ---------------------------------------------------------------------------
def _df_lookup(hashes: np.ndarray, df_hashes: np.ndarray, df_values: np.ndarray) -> np.ndarray:
    """Document frequency for each hash, 0 for a token absent from the table."""
    if df_hashes.size == 0:
        return np.zeros(len(hashes), dtype=np.int64)
    slots = np.searchsorted(df_hashes, hashes)
    clipped = np.minimum(slots, df_hashes.size - 1)
    hit = (slots < df_hashes.size) & (df_hashes[clipped] == hashes)
    return np.where(hit, df_values[clipped], 0).astype(np.int64)


def _rarest_name_token(name: str, df_hashes: np.ndarray, df_values: np.ndarray) -> tuple[str, int]:
    """The corpus-present token of ``name`` with the smallest document frequency.

    Recomputed for the handful of entities the census lists by name, so the
    table can show WHY an entity has the volume it has.
    """
    tokens = sorted(_token_set(name))
    if not tokens:
        return "", 0
    frequencies = _df_lookup(stable_hash64(tokens), df_hashes, df_values)
    present = frequencies > 0
    if not present.any():
        return tokens[0], 0
    index = int(np.argmin(np.where(present, frequencies, UNKNOWN_DF)))
    return tokens[index], int(frequencies[index])


def _candidate_census(
    names: np.ndarray,
    entity_ids: np.ndarray,
    df_hashes: np.ndarray,
    df_values: np.ndarray,
    pairs_per_entity: np.ndarray,
    exact_candidates: Optional[np.ndarray],
    log: logging.Logger,
) -> dict[str, Any]:
    """How many candidates would each of the 2.2M source1 entities generate?

    Phases 0.2-0.4 measure the TRUE pairs. That is the wrong population for the
    question that decides the blocker: macro F0.5 divides by the number of
    source1 ENTITIES, so the cost of a candidate is paid per entity, and an
    entity that generates no candidate at all scores zero however good the
    scorer is. This walks every source1 row - matched or not - and estimates
    the candidate volume of one rarest-usable-token probe per entity.

    The estimate is a single token's posting-list length, the same quantity
    ``_pair_rarity`` sums over pairs, so the pair view and the entity view use
    one yardstick. Summed over entities it is an upper bound on candidate rows
    before deduplication (two entities probing the same list are two probes).

    No candidate pair is materialized: the whole census is a handful of integers
    per entity, ``O(n_source1)`` memory.
    """
    n_entities = len(names)
    blocks = (n_entities + CENSUS_BLOCK - 1) // CENSUS_BLOCK if n_entities else 0
    log.info(
        "census: estimating candidate volume for %s source1 entities in %s blocks",
        fmt_int(n_entities),
        fmt_int(blocks),
    )

    # Per entity: how many name tokens, how many of them occur anywhere in the
    # analysed target corpus, and the smallest such document frequency. One
    # integer per entity answers every cap in the grid, exactly as min_df does
    # for a pair: "has a token usable at cap" is "rarest <= cap".
    n_tokens = np.zeros(n_entities, dtype=np.int64)
    n_present = np.zeros(n_entities, dtype=np.int64)
    rarest = np.zeros(n_entities, dtype=np.int64)

    for block in range(blocks):
        start = block * CENSUS_BLOCK
        stop = min(start + CENSUS_BLOCK, n_entities)
        rows = [_token_set(str(names[index])) for index in range(start, stop)]
        counts = np.fromiter((len(row) for row in rows), dtype=np.int64, count=stop - start)
        n_tokens[start:stop] = counts
        flat = [token for row in rows for token in row]
        if not flat:
            continue
        frequencies = _df_lookup(stable_hash64(flat), df_hashes, df_values)
        # A token with df == 0 occurs in NO target record: it cannot generate a
        # candidate, so it is masked out of the minimum rather than being
        # mistaken for an excellent rare token.
        values = np.where(frequencies > 0, frequencies, UNKNOWN_DF)
        nonempty = counts > 0
        starts = np.cumsum(counts)[nonempty] - counts[nonempty]
        block_rarest = np.minimum.reduceat(values, starts)
        block_rarest = np.where(block_rarest >= UNKNOWN_DF, 0, block_rarest).astype(np.int64)
        n_present[start:stop][nonempty] = np.add.reduceat((frequencies > 0).astype(np.int64), starts)
        rarest[start:stop][nonempty] = block_rarest
        if block % 20 == 19:
            log.info("census: %s / %s entities scanned", fmt_int(stop), fmt_int(n_entities))

    usable = (rarest > 0) & (rarest <= REFERENCE_TOKEN_CAP)
    estimate = np.where(usable, rarest, 0).astype(np.int64)
    n_usable_entities = int(usable.sum())
    total_estimate = int(estimate.sum())

    # -- structural zeros: entities no token blocker can reach at all -------
    empty_name = n_tokens == 0
    no_present_token = (n_tokens > 0) & (n_present == 0)
    all_too_frequent = (n_present > 0) & ~usable
    # The three reasons partition the structural zeros exactly: an entity has no
    # usable token because it has no token, because none of its tokens occur in
    # the corpus, or because every one of them is above the cap.
    assert int((empty_name | no_present_token | all_too_frequent).sum()) == int((~usable).sum())
    structural_zero = {
        "n_entities": n_entities - n_usable_entities,
        "pct": _pct(n_entities - n_usable_entities, n_entities),
        "definition": (
            f"no name token with 0 < document frequency <= {REFERENCE_TOKEN_CAP}, so a "
            "rarest-token probe returns nothing at all"
        ),
        "by_reason": {
            "empty_name_norm": int(empty_name.sum()),
            "no_token_in_target_corpus": int(no_present_token.sum()),
            "every_token_above_the_cap": int(all_too_frequent.sum()),
        },
    }

    # -- volume ------------------------------------------------------------
    nonzero = estimate[estimate > 0]
    volume = {
        "cap": REFERENCE_TOKEN_CAP,
        "total_estimated_candidate_rows": total_estimate,
        "mean_per_entity": _ratio(total_estimate / n_entities) if n_entities else 0.0,
        "entities_with_a_candidate": {
            "n_entities": n_usable_entities,
            "pct": _pct(n_usable_entities, n_entities),
        },
        # Over every source1 entity, zeros included: this is the distribution
        # macro F0.5 feels, because the zeros are entities that cannot score.
        "estimated_candidates_all_entities": _distribution(estimate.astype(np.float64)),
        # Over the reachable entities only: how large a probe is when it happens.
        "estimated_candidates_reachable_entities": _distribution(nonzero.astype(np.float64)),
        "reachable_entities_p50_p90_p99_max": {
            **{
                label: _ratio(float(np.quantile(nonzero, QUANTILES[label]))) if nonzero.size else 0.0
                for label in ("p50", "p90", "p99")
            },
            "max": int(nonzero.max()) if nonzero.size else 0,
        },
        "histogram_reachable_entities": _bucket_counts(nonzero, CANDIDATE_BUCKETS),
    }

    by_cap = []
    for cap in RARITY_CAP_THRESHOLDS:
        reachable_at_cap = (rarest > 0) & (rarest <= cap)
        postings = rarest[reachable_at_cap]
        by_cap.append(
            {
                "cap": cap,
                "n_entities_with_a_candidate": int(reachable_at_cap.sum()),
                "pct_of_entities": _pct(int(reachable_at_cap.sum()), n_entities),
                "n_structural_zero_entities": n_entities - int(reachable_at_cap.sum()),
                "pct_structural_zero": _pct(n_entities - int(reachable_at_cap.sum()), n_entities),
                "total_estimated_candidate_rows": int(postings.sum()),
                "mean_postings_per_reached_entity": (
                    _ratio(float(postings.mean())) if postings.size else 0.0
                ),
                "max_postings_per_reached_entity": int(postings.max()) if postings.size else 0,
            }
        )

    # -- candidate-to-truth ratio -----------------------------------------
    with_pairs = pairs_per_entity > 0
    n_with_pairs = int(with_pairs.sum())
    analysed_pairs = int(pairs_per_entity[with_pairs].sum())
    candidates_on_entities_with_pairs = int(estimate[with_pairs].sum())
    denom = pairs_per_entity[with_pairs].astype(np.float64)
    per_entity_ratio = np.divide(
        estimate[with_pairs].astype(np.float64),
        denom,
        out=np.zeros(n_with_pairs, dtype=np.float64),
        where=denom > 0,
    )
    starved = with_pairs & (estimate < pairs_per_entity)
    zero_candidate_with_pairs = with_pairs & (estimate == 0)
    ratio_report = {
        "aggregate": (
            _ratio(candidates_on_entities_with_pairs / analysed_pairs) if analysed_pairs else 0.0
        ),
        "definition": (
            "estimated candidates of an entity divided by its analysed true pairs. Below "
            "1 means the probe cannot even cover the entity's own matches, so the "
            "denominator of recall@K is out of reach however good the scoring is."
        ),
        "n_entities_with_analysed_pairs": n_with_pairs,
        "n_source1_entities_without_analysed_pairs": n_entities - n_with_pairs,
        "n_analysed_true_pairs": analysed_pairs,
        "per_entity": _distribution(per_entity_ratio),
        "n_entities_below_one": int(starved.sum()),
        "pct_entities_below_one": _pct(int(starved.sum()), n_with_pairs),
        "n_entities_with_zero_candidates": int(zero_candidate_with_pairs.sum()),
        "pct_entities_with_zero_candidates": _pct(int(zero_candidate_with_pairs.sum()), n_with_pairs),
    }

    order = heapq.nlargest(CENSUS_TOP_ENTITIES, range(n_entities), key=estimate.__getitem__)
    order.sort(key=lambda index: (-int(estimate[index]), str(entity_ids[index])))
    top_entities = []
    for rank, index in enumerate(order, start=1):
        if estimate[index] <= 0:
            break
        token, frequency = _rarest_name_token(str(names[index]), df_hashes, df_values)
        top_entities.append(
            {
                "rank": rank,
                "source1_entity_id": str(entity_ids[index]),
                "name_norm": str(names[index]),
                "n_name_tokens": int(n_tokens[index]),
                "rarest_usable_token": token,
                "rarest_usable_token_df": frequency,
                "estimated_candidates": int(estimate[index]),
                "analysed_true_pairs": int(pairs_per_entity[index]),
            }
        )

    report: dict[str, Any] = {
        "n_source1_entities": n_entities,
        "token_signal": {
            **volume,
            "structural_zero_entities": structural_zero,
            "by_cap": by_cap,
            "candidate_to_truth_ratio": ratio_report,
            "top_entities_by_estimated_candidates": top_entities,
        },
        "exact_name_signal": (
            {"available": False, "reason": "no exact-name index was loaded"}
            if exact_candidates is None
            else _exact_per_entity_stats(exact_candidates, pairs_per_entity, entity_ids, names)
        ),
        "note": (
            "Estimated candidate volume is the size of ONE posting list per entity - the "
            "rarest usable token. It is an upper bound on candidate ROWS before "
            "deduplication (a real blocker may probe several tokens, or truncate per "
            "entity), and it deliberately ignores scoring: the questions here are whether "
            "the candidate space contains the entity's matches at all, and how much volume "
            "macro F0.5 has to process. Nothing here is a blocker threshold; the cap is "
            "the reference point shared with Phase 0.2."
        ),
    }

    if exact_candidates is not None:
        report["reachability"] = _reachability_overlap(
            estimate, exact_candidates, usable, pairs_per_entity, n_entities
        )

    log.info(
        "census: %s / %s source1 entities have a usable token at cap %s; %s estimated "
        "candidate rows in total",
        fmt_int(n_usable_entities),
        fmt_int(n_entities),
        REFERENCE_TOKEN_CAP,
        fmt_int(total_estimate),
    )
    return report


def _exact_per_entity_stats(
    exact_candidates: np.ndarray,
    pairs_per_entity: np.ndarray,
    entity_ids: np.ndarray,
    names: np.ndarray,
) -> dict[str, Any]:
    """Per-entity exact-name candidate counts, the same shape as the token view."""
    n_entities = len(exact_candidates)
    nonzero = exact_candidates[exact_candidates > 0]
    order = heapq.nlargest(CENSUS_TOP_ENTITIES, range(n_entities), key=exact_candidates.__getitem__)
    order.sort(key=lambda index: (-int(exact_candidates[index]), str(entity_ids[index])))
    return {
        "available": True,
        "n_entities_with_a_candidate": int((exact_candidates > 0).sum()),
        "pct_of_entities": _pct(int((exact_candidates > 0).sum()), n_entities),
        "total_candidate_rows": int(exact_candidates.sum()),
        "mean_per_entity": _ratio(float(exact_candidates.mean())) if n_entities else 0.0,
        "candidates_all_entities": _distribution(exact_candidates.astype(np.float64)),
        "candidates_reachable_entities": _distribution(nonzero.astype(np.float64)),
        "histogram_reachable_entities": _bucket_counts(nonzero, CANDIDATE_BUCKETS),
        "top_entities_by_exact_candidates": [
            {
                "rank": rank,
                "source1_entity_id": str(entity_ids[index]),
                "name_norm": str(names[index]),
                "exact_candidates": int(exact_candidates[index]),
                "analysed_true_pairs": int(pairs_per_entity[index]),
            }
            for rank, index in enumerate(order, start=1)
            if exact_candidates[index] > 0
        ],
    }


def _reachability_overlap(
    token_estimate: np.ndarray,
    exact_candidates: np.ndarray,
    token_usable: np.ndarray,
    pairs_per_entity: np.ndarray,
    n_entities: int,
) -> dict[str, Any]:
    """Can either cheap signal propose anything at all for an entity?

    The two fail for different reasons - the exact key is absent, or no token is
    rare enough - so their union is the real reachable population, and the
    entities in neither are the ones only a semantic method (or a name-only
    index) could ever bring back.
    """
    exact_usable = exact_candidates > 0
    neither = ~token_usable & ~exact_usable
    with_pairs = pairs_per_entity > 0
    n_with_pairs = int(with_pairs.sum())
    n_neither_with_pairs = int((neither & with_pairs).sum())
    return {
        "both_signals_reach": int((token_usable & exact_usable).sum()),
        "token_only": int((token_usable & ~exact_usable).sum()),
        "exact_name_only": int((~token_usable & exact_usable).sum()),
        "neither_signal_reaches": {
            "n_entities": int(neither.sum()),
            "pct": _pct(int(neither.sum()), n_entities),
            "n_entities_with_analysed_pairs": n_neither_with_pairs,
            "pct_of_entities_with_analysed_pairs": _pct(n_neither_with_pairs, n_with_pairs),
            "definition": (
                "no usable name token AND no exact-name candidate: both cheap signals "
                "return nothing for this entity, so its matches cannot be in the "
                "candidate set at any scoring quality"
            ),
        },
        "note": (
            "Per-entity reachability, not per-pair coverage. An entity can be reachable "
            "and still miss individual matches; Phase 0.2 measures that side."
        ),
    }


def _exact_name_census(
    config: dict,
    split: str,
    sources: Sequence[str],
    s1_names: np.ndarray,
    log: logging.Logger,
) -> tuple[dict[str, Any], Optional[np.ndarray]]:
    """Read the persisted exact-name indexes READ-ONLY and count what they hold.

    Two questions only the built index can answer: how large are the posting
    lists a whole-name probe touches (the duplicate / near-duplicate target
    question in its exact form), and how many source1 entities does an exact
    probe reach for real - over the whole 2.2M population, not just the pairs
    that happen to be true.

    The index is loaded, queried and dropped; nothing is written to it and no
    candidate pair is materialized. Returns the report plus the per-entity
    candidate counts (``None`` when no index could be loaded), which the census
    cross-tabulates against the token estimate.
    """
    per_source: dict[str, Any] = {}
    candidates_total: Optional[np.ndarray] = None

    for source in sources:
        directory = index_dir_for(config, split, source, BLOCKER_EXACT_NAME)
        try:
            index = ExactNameIndex.load(directory, log)
        except (FileNotFoundError, ValueError) as exc:
            log.warning(
                "exact-name census for %s skipped: %s\n  Build it with: "
                "python scripts/build_indexes.py",
                source,
                exc,
            )
            per_source[source] = {
                "available": False,
                "index_dir": str(directory),
                "reason": str(exc),
            }
            continue

        posting_sizes = np.diff(index.postings_offsets).astype(np.float64)
        order = heapq.nlargest(CENSUS_TOP_KEYS, range(index.n_unique_keys), key=posting_sizes.__getitem__)
        order.sort(key=lambda position: (-posting_sizes[position], index.key_at(position)))
        entry: dict[str, Any] = {
            "available": True,
            "index_dir": str(directory),
            "key_field": index.key_field,
            "describe": index.describe(),
            "duplicate_keys": {
                "n_keys_shared_by_two_or_more_targets": int((posting_sizes >= 2).sum()),
                "n_keys_shared_by_three_or_more_targets": int((posting_sizes >= 3).sum()),
                "n_postings_in_duplicate_keys": int(posting_sizes[posting_sizes >= 2].sum()),
                "pct_keys_with_duplicates": _pct(
                    int((posting_sizes >= 2).sum()), index.n_unique_keys
                ),
            },
            "posting_size": {
                **_distribution(posting_sizes),
                "p50": _ratio(float(np.quantile(posting_sizes, 0.50))) if posting_sizes.size else 0.0,
                "p90": _ratio(float(np.quantile(posting_sizes, 0.90))) if posting_sizes.size else 0.0,
                "p99": _ratio(float(np.quantile(posting_sizes, 0.99))) if posting_sizes.size else 0.0,
                "max": int(posting_sizes.max()) if posting_sizes.size else 0,
            },
            "largest_posting_lists": [
                {
                    "rank": rank,
                    "key": index.key_at(position),
                    "postings": int(posting_sizes[position]),
                }
                for rank, position in enumerate(order, start=1)
                if posting_sizes[position] > 0
            ],
        }

        # Query with the index's OWN key field as the source1 column: the census
        # must count what the index can actually retrieve.
        if index.key_field != NAME_NORM:
            log.warning(
                "index for %s is keyed on %r, not %r - the source1 query column is "
                "being taken from the index, not assumed",
                source,
                index.key_field,
                NAME_NORM,
            )
        counts = np.zeros(len(s1_names), dtype=np.int64)
        for start in range(0, len(s1_names), CENSUS_BLOCK):
            stop = min(start + CENSUS_BLOCK, len(s1_names))
            _, block_counts = index.lookup_many(s1_names[start:stop])
            counts[start:stop] = block_counts
        n_hit = int((counts > 0).sum())
        entry["source1_query"] = {
            "n_entities_queried": int(len(s1_names)),
            "n_entities_with_a_candidate": n_hit,
            "pct_of_entities": _pct(n_hit, len(s1_names)),
            "total_candidate_rows": int(counts.sum()),
            "histogram_reachable_entities": _bucket_counts(
                counts[counts > 0].astype(np.int64), CANDIDATE_BUCKETS
            ),
            "note": (
                "The same lookup the blocker would run, over every source1 entity. These "
                "rows are true candidate rows (the index was actually probed); the token "
                "estimate in the census is the upper bound of the same quantity."
            ),
        }
        per_source[source] = entry
        candidates_total = counts if candidates_total is None else candidates_total + counts
        del index  # one index resident at a time (~250MB at source2 scale)

    report: dict[str, Any] = {
        "per_source": per_source,
        "n_sources_with_index": int(sum(1 for entry in per_source.values() if entry["available"])),
    }
    if candidates_total is None:
        report["available"] = False
        return report, None
    n_with_candidate = int((candidates_total > 0).sum())
    report.update(
        {
            "available": True,
            "n_source1_entities": int(len(s1_names)),
            "n_entities_with_any_exact_candidate": n_with_candidate,
            "pct_entities_with_any_exact_candidate": _pct(n_with_candidate, len(s1_names)),
            "total_candidate_rows": int(candidates_total.sum()),
            "candidates_all_entities": _distribution(candidates_total.astype(np.float64)),
            "candidates_reachable_entities": _distribution(
                candidates_total[candidates_total > 0].astype(np.float64)
            ),
            "note": (
                "The UNION over the analysed sources: an entity's candidate rows are the "
                "sum of the posting lists its key hits in each source. Read-only - no "
                "index was modified and no candidate file was written."
            ),
        }
    )
    return report, candidates_total


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _render_coverage(coverage: dict) -> list[str]:
    """Phase 0.2 in full: signals, combinations, residue and cost."""
    lines: list[str] = []
    lines.append("## Phase 0.2 - signal coverage and complementarity")
    lines.append("")
    lines.append(
        "Measured on the TRUE pairs, so every figure below is a RECALL ceiling. Which "
        "signals reach a pair says nothing about precision: that needs candidate "
        "generation (Phase 1) and the volume figures in the census. The four signals are "
        "pure functions of the two normalized records - none reads a country - so they "
        "behave the same way on a country, language or script the training split never saw."
    )
    lines.append("")
    lines.extend(
        _table(
            ["signal", "definition", "reference threshold", "true pairs reached", "% of pairs", "source1 entities reached", "% of entities with pairs"],
            [
                [
                    SIGNAL_LABELS[signal],
                    coverage["signal_definitions"][signal],
                    (
                        "-"
                        if entry["reference_threshold"] is None
                        else fmt_int(entry["reference_threshold"])
                        if signal == TOKEN_SIGNAL
                        else f"{entry['reference_threshold']}"
                    ),
                    fmt_int(entry["n_pairs"]),
                    f"{entry['pct_of_pairs']:.2f}",
                    fmt_int(entry["n_entities"]),
                    f"{entry['pct_of_entities_with_pairs']:.2f}",
                ]
                for signal, entry in coverage["signals"].items()
            ],
        )
    )
    lines.append("")
    lines.append(
        f"Union of all four signals: **{fmt_int(coverage['union']['n_pairs'])}** of "
        f"{fmt_int(coverage['n_pairs'])} true pairs ({coverage['union']['pct_of_pairs']:.2f}%), "
        f"reaching {fmt_int(coverage['union']['n_entities'])} of "
        f"{fmt_int(coverage['n_source1_entities_with_analysed_pairs'])} source1 entities with "
        f"analysed pairs ({coverage['union']['pct_of_entities_with_pairs']:.2f}%)."
    )
    lines.append("")
    lines.append("Which signals each pair is reached by (a pair appears in exactly one row):")
    lines.append("")
    lines.extend(
        _table(
            ["reached by", "true pairs", "% of pairs", "source1 entities", "% of entities"],
            [
                [
                    row["signals"].replace("+", " + ") if row["signals"] != NO_SIGNAL_LABEL else "**no signal**",
                    fmt_int(row["n_pairs"]),
                    f"{row['pct_of_pairs']:.2f}",
                    fmt_int(row["n_entities"]),
                    f"{row['pct_of_entities_with_pairs']:.2f}",
                ]
                for row in coverage["union"]["by_combination"]
                if row["n_pairs"] or row["signals"] == NO_SIGNAL_LABEL
            ],
        )
    )
    lines.append("")
    outcome = coverage["union"]["entity_outcomes"]
    lines.append(
        "Per source1 entity - what macro F0.5 averages over, so an entity in the first row "
        "can never score:"
    )
    lines.append("")
    lines.extend(
        _table(
            ["entity outcome", "entities", "% of entities with pairs"],
            [
                ["no pair reached", fmt_int(outcome["no_pair_reached"]["n_entities"]), f"{outcome['no_pair_reached']['pct']:.2f}"],
                ["some pairs reached", fmt_int(outcome["some_pairs_reached"]["n_entities"]), f"{outcome['some_pairs_reached']['pct']:.2f}"],
                ["all pairs reached", fmt_int(outcome["all_pairs_reached"]["n_entities"]), f"{outcome['all_pairs_reached']['pct']:.2f}"],
            ],
        )
    )
    lines.append("")
    lines.append("Per target source:")
    lines.append("")
    lines.extend(
        _table(
            ["source", "true pairs", *[SIGNAL_LABELS[s] for s in SIGNAL_LABELS], "union", "residue"],
            [
                [
                    source,
                    fmt_int(entry["n_pairs"]),
                    *[
                        f"{entry['signals'][signal]['n_pairs']} ({entry['signals'][signal]['pct_of_pairs']:.1f}%)"
                        for signal in SIGNAL_LABELS
                    ],
                    f"{entry['union']['n_pairs']} ({entry['union']['pct_of_pairs']:.1f}%)",
                    f"{entry['residue']['n_pairs']} ({entry['residue']['pct_of_pairs']:.1f}%)",
                ]
                for source, entry in coverage["per_source"].items()
            ],
        )
    )
    lines.append("")

    lines.append("### Coverage versus candidate cost")
    lines.append("")
    lines.append(
        "Every threshold is shown with the candidate volume it implies, so coverage and "
        "cost are read together rather than one at a time."
    )
    lines.append("")
    for signal, rows in coverage["sensitivity"].items():
        lines.append(f"{SIGNAL_LABELS[signal]} (`{signal}`):")
        lines.append("")
        lines.extend(
            _table(
                ["threshold", "true pairs reached", "% of pairs", "candidate cost", "candidate rows", "mean per pair"],
                [
                    [
                        fmt_int(row["threshold"]) if signal == TOKEN_SIGNAL else f"{row['threshold']}",
                        fmt_int(row["n_pairs"]),
                        f"{row['pct_of_pairs']:.2f}",
                        row["candidate_cost"]["kind"],
                        "n/a" if row["candidate_cost"]["n_candidates"] is None else fmt_int(row["candidate_cost"]["n_candidates"]),
                        "n/a" if row["candidate_cost"]["mean_per_pair"] is None else f"{row['candidate_cost']['mean_per_pair']:.1f}",
                    ]
                    for row in rows
                ],
            )
        )
        lines.append("")
    lines.append(
        "`posting_list` rows are a real cost (the sum of the probed posting lists). "
        "`filter` rows add no candidates - an address rule is applied to pairs another "
        "signal proposed, so it can only remove volume. `not_estimable_in_phase_0` means "
        "the cost depends on how the character signal would be inverted into an index, "
        "which is a Phase 1 decision; the coverage column is still exact."
    )
    lines.append("")

    residue = coverage["residue"]
    lines.append("### The irreducible residue")
    lines.append("")
    lines.append(
        f"**{fmt_int(residue['n_pairs'])}** true pairs ({residue['pct_of_pairs']:.2f}%) are "
        f"reached by NO signal, across {fmt_int(residue['n_entities'])} source1 entities. "
        f"{residue['definition']}. These are the pairs only a semantic method can recover, "
        "and their size is the hard floor on what any cheap lexical blocker can achieve."
    )
    lines.append("")
    if residue.get("per_source"):
        lines.append(
            "Per target source: "
            + ", ".join(
                f"{source} {fmt_int(entry['n_pairs'])} ({entry['pct_of_pairs']:.2f}%)"
                for source, entry in residue["per_source"].items()
            )
            + "."
        )
        lines.append("")
    if residue["by_name_difference_category"]:
        lines.append("What the residue is made of, by Phase 0.1 name-difference category:")
        lines.append("")
        lines.extend(
            _table(
                ["name-difference category", "residue pairs", "% of residue"],
                [
                    [name, fmt_int(entry["n_pairs"]), f"{entry['pct_of_residue']:.2f}"]
                    for name, entry in sorted(
                        residue["by_name_difference_category"].items(),
                        key=lambda item: -item[1]["n_pairs"],
                    )
                ],
            )
        )
        lines.append("")
        if residue["by_name_difference_reason"]:
            lines.append(
                "Reasons within `unknown_other` (where the classifier could not name a "
                "category): "
                + ", ".join(
                    f"`{name}` {fmt_int(entry['n_pairs'])}"
                    for name, entry in residue["by_name_difference_reason"].items()
                )
                + "."
            )
            lines.append("")
        script = residue.get("by_script_relation") or {}
        if script:
            lines.append(
                f"Script relation: {fmt_int(script['different_script'])} residue pairs are "
                f"cross-script (transliteration), {fmt_int(script['same_script'])} are "
                "within one script. Cross-script residue is the case a character-level "
                "signal cannot see by construction."
            )
            lines.append("")
    if residue["examples"]:
        lines.append(f"First {len(residue['examples'])} residue pairs:")
        lines.append("")
        lines.extend(
            _table(
                ["source1 entity", "target entity", "source", "source1 name_norm", "target name_norm", "category", "reason"],
                [
                    [
                        row["source1_entity_id"],
                        row["target_entity_id"],
                        row["source"],
                        row["name_norm_source1"],
                        row["name_norm_target"],
                        row.get("name_difference_category", "-"),
                        row.get("name_difference_reason", "-"),
                    ]
                    for row in residue["examples"][:15]
                ],
            )
        )
        lines.append("")
    if coverage["coverage_by_name_difference_category"]:
        lines.append("Coverage of every signal within each name-difference category:")
        lines.append("")
        lines.extend(
            _table(
                ["category", "pairs", *[SIGNAL_LABELS[s] for s in SIGNAL_LABELS], "union", "residue"],
                [
                    [
                        name,
                        fmt_int(entry["n_pairs"]),
                        *[
                            f"{entry['signals'][signal]['n_pairs']} ({entry['signals'][signal]['pct_of_category']:.0f}%)"
                            for signal in SIGNAL_LABELS
                        ],
                        f"{entry['union']['n_pairs']} ({entry['union']['pct_of_category']:.1f}%)",
                        f"{entry['residue']['n_pairs']} ({entry['residue']['pct_of_category']:.1f}%)",
                    ]
                    for name, entry in coverage["coverage_by_name_difference_category"].items()
                ],
            )
        )
        lines.append("")
    lines.append(f"> {coverage['note']}")
    lines.append("")
    return lines


def _render_census(census: dict, exact_index: dict) -> list[str]:
    """Phase 0.4's second half: candidate volume over every source1 entity."""
    lines: list[str] = []
    token = census["token_signal"]
    lines.append("### Candidate-volume census over all source1 entities")
    lines.append("")
    lines.append(
        f"The pair tables above are the matched population. This is the other "
        f"{fmt_int(census['n_source1_entities'])} rows - matched or not - because macro F0.5 "
        "divides by source1 entities, so an entity that generates no candidate scores zero "
        "however good the scorer is."
    )
    lines.append("")
    zero = token["structural_zero_entities"]
    lines.append(
        f"At cap {fmt_int(token['cap'])}: {fmt_int(token['entities_with_a_candidate']['n_entities'])} "
        f"entities ({token['entities_with_a_candidate']['pct']:.2f}%) generate a candidate; "
        f"**{fmt_int(zero['n_entities'])} ({zero['pct']:.2f}%) are structural zeros** - "
        f"{zero['definition']}. Of those, "
        + ", ".join(f"{name.replace('_', ' ')} {fmt_int(count)}" for name, count in zero["by_reason"].items())
        + "."
    )
    lines.append("")
    lines.extend(
        _table(
            ["estimated candidates", "all source1 entities", "reachable entities only"],
            [
                ["n", fmt_int(token["estimated_candidates_all_entities"].get("n", 0)), fmt_int(token["estimated_candidates_reachable_entities"].get("n", 0))],
                ["mean", f"{token['estimated_candidates_all_entities'].get('mean', 0):.2f}", f"{token['estimated_candidates_reachable_entities'].get('mean', 0):.2f}"],
                ["p50", fmt_int(token["estimated_candidates_all_entities"].get("p50", 0)), fmt_int(token["estimated_candidates_reachable_entities"].get("p50", 0))],
                ["p90", fmt_int(token["estimated_candidates_all_entities"].get("p90", 0)), fmt_int(token["estimated_candidates_reachable_entities"].get("p90", 0))],
                ["p99", fmt_int(token["estimated_candidates_all_entities"].get("p99", 0)), fmt_int(token["estimated_candidates_reachable_entities"].get("p99", 0))],
                ["max", fmt_int(token["estimated_candidates_all_entities"].get("max", 0)), fmt_int(token["estimated_candidates_reachable_entities"].get("max", 0))],
            ],
        )
    )
    lines.append("")
    lines.append(
        f"Total estimated candidate rows at cap {fmt_int(token['cap'])}: "
        f"**{fmt_int(token['total_estimated_candidate_rows'])}** "
        f"({token['mean_per_entity']:.2f} per source1 entity). The same quantity at every cap:"
    )
    lines.append("")
    lines.extend(
        _table(
            ["cap", "entities with a candidate", "%", "structural zeros", "%", "estimated candidate rows", "mean postings per reached entity", "max"],
            [
                [
                    fmt_int(row["cap"]),
                    fmt_int(row["n_entities_with_a_candidate"]),
                    f"{row['pct_of_entities']:.2f}",
                    fmt_int(row["n_structural_zero_entities"]),
                    f"{row['pct_structural_zero']:.2f}",
                    fmt_int(row["total_estimated_candidate_rows"]),
                    f"{row['mean_postings_per_reached_entity']:.1f}",
                    fmt_int(row["max_postings_per_reached_entity"]),
                ]
                for row in token["by_cap"]
            ],
        )
    )
    lines.append("")
    ratio = token["candidate_to_truth_ratio"]
    lines.append(
        f"Candidate-to-truth ratio over the {fmt_int(ratio['n_entities_with_analysed_pairs'])} "
        f"entities that have analysed pairs ({fmt_int(ratio['n_analysed_true_pairs'])} pairs): "
        f"aggregate **{ratio['aggregate']}**, per-entity median "
        f"{ratio['per_entity'].get('p50', 0):.2f}, p90 {ratio['per_entity'].get('p90', 0):.2f}, "
        f"p99 {ratio['per_entity'].get('p99', 0):.2f}, max "
        f"{ratio['per_entity'].get('max', 0):.2f}. "
        f"{fmt_int(ratio['n_entities_below_one'])} entities "
        f"({ratio['pct_entities_below_one']:.2f}%) sit below 1.0: "
        f"{ratio['definition']}"
    )
    lines.append("")
    if token["top_entities_by_estimated_candidates"]:
        lines.append("Largest estimated candidate volumes:")
        lines.append("")
        lines.extend(
            _table(
                ["rank", "source1 entity", "name_norm", "tokens", "rarest usable token", "its df", "estimated candidates", "analysed pairs"],
                [
                    [
                        row["rank"],
                        row["source1_entity_id"],
                        row["name_norm"],
                        fmt_int(row["n_name_tokens"]),
                        f"`{row['rarest_usable_token']}`",
                        fmt_int(row["rarest_usable_token_df"]),
                        fmt_int(row["estimated_candidates"]),
                        fmt_int(row["analysed_true_pairs"]),
                    ]
                    for row in token["top_entities_by_estimated_candidates"][:15]
                ],
            )
        )
        lines.append("")

    lines.append("### Exact-name index census (read-only)")
    lines.append("")
    if not exact_index.get("available"):
        lines.append(
            "No exact-name index could be read, so no real candidate rows were counted. "
            "Build the indexes with `python scripts/build_indexes.py`, then re-run."
        )
        lines.append("")
        for source, entry in exact_index.get("per_source", {}).items():
            if not entry["available"]:
                lines.append(f"- `{source}`: {entry['reason']}")
        lines.append("")
    else:
        lines.extend(
            _table(
                ["source", "targets indexed", "unique keys", "postings", "avg per key", "max per key", "keys with >= 2 targets", "% keys with duplicates"],
                [
                    [
                        source,
                        fmt_int(entry["describe"]["n_entities_indexed"]),
                        fmt_int(entry["describe"]["n_unique_keys"]),
                        fmt_int(entry["describe"]["n_postings"]),
                        f"{entry['describe']['avg_postings_per_key']}",
                        fmt_int(entry["describe"]["max_postings_per_key"]),
                        fmt_int(entry["duplicate_keys"]["n_keys_shared_by_two_or_more_targets"]),
                        f"{entry['duplicate_keys']['pct_keys_with_duplicates']:.2f}",
                    ]
                    for source, entry in exact_index["per_source"].items()
                    if entry["available"]
                ],
            )
        )
        lines.append("")
        lines.append(
            f"Over all {fmt_int(exact_index['n_source1_entities'])} source1 entities, an exact "
            f"probe returns at least one candidate for "
            f"**{fmt_int(exact_index['n_entities_with_any_exact_candidate'])}** "
            f"({exact_index['pct_entities_with_any_exact_candidate']:.2f}%), "
            f"{fmt_int(exact_index['total_candidate_rows'])} candidate rows in total. "
            "These are real candidate rows - the index was queried, not estimated."
        )
        lines.append("")
        for source, entry in exact_index["per_source"].items():
            if not entry["available"]:
                continue
            query = entry["source1_query"]
            lines.append(
                f"- `{source}`: {fmt_int(query['n_entities_with_a_candidate'])} entities "
                f"({query['pct_of_entities']:.2f}%) hit, "
                f"{fmt_int(query['total_candidate_rows'])} candidate rows."
            )
        lines.append("")
        for source, entry in exact_index["per_source"].items():
            if not entry["available"] or not entry["largest_posting_lists"]:
                continue
            lines.append(f"Largest posting lists in `{source}` (whole-name duplicate clusters):")
            lines.append("")
            lines.extend(
                _table(
                    ["rank", "name_norm key", "targets sharing it"],
                    [
                        [row["rank"], row["key"], fmt_int(row["postings"])]
                        for row in entry["largest_posting_lists"][:10]
                    ],
                )
            )
            lines.append("")

    if "reachability" in census:
        reach = census["reachability"]
        neither = reach["neither_signal_reaches"]
        lines.append("### Entity reachability of the two cheap signals")
        lines.append("")
        lines.extend(
            _table(
                ["reachable by", "entities", "% of source1"],
                [
                    ["both signals", fmt_int(reach["both_signals_reach"]), f"{_pct(reach['both_signals_reach'], census['n_source1_entities']):.2f}"],
                    ["token only", fmt_int(reach["token_only"]), f"{_pct(reach['token_only'], census['n_source1_entities']):.2f}"],
                    ["exact name only", fmt_int(reach["exact_name_only"]), f"{_pct(reach['exact_name_only'], census['n_source1_entities']):.2f}"],
                    ["**neither**", fmt_int(neither["n_entities"]), f"{neither['pct']:.2f}"],
                ],
            )
        )
        lines.append("")
        lines.append(
            f"Of the {fmt_int(neither['n_entities_with_analysed_pairs'])} entities that are "
            f"reachable by neither signal and have analysed pairs "
            f"({neither['pct_of_entities_with_analysed_pairs']:.2f}% of those with pairs), no "
            "candidate set generated by these two signals can contain their matches at any "
            "scoring quality. This is the entity-level floor, and it is a stronger statement "
            "than Phase 0.2's pair-level residue."
        )
        lines.append("")
    exact_signal = census["exact_name_signal"]
    if exact_signal.get("available"):
        lines.append(
            f"Exact-name candidates per entity (all {fmt_int(census['n_source1_entities'])} "
            f"rows): {fmt_int(exact_signal['n_entities_with_a_candidate'])} entities "
            f"({exact_signal['pct_of_entities']:.2f}%) with at least one, "
            f"{fmt_int(exact_signal['total_candidate_rows'])} candidate rows, mean "
            f"{exact_signal['mean_per_entity']:.3f} per entity, max "
            f"{fmt_int(exact_signal['candidates_all_entities'].get('max', 0))}."
        )
        lines.append("")
    lines.append(f"> {census['note']}")
    lines.append("")
    return lines


def _render_markdown(report: dict) -> str:
    """Human-readable summary: the four answers first, then the detail."""
    meta = report["meta"]
    coverage = report["signal_coverage"]
    census = report["candidate_census"]
    exact_index = report["exact_name_census"]
    overlap = report["address_overlap"]
    tokens = report["token_frequency"]["all"]
    rarity = report["pair_rarity"]
    zero = report["zero_match"]

    lines: list[str] = []
    lines.append("# Blocking statistics (Phase 0.2-0.5)")
    lines.append("")
    lines.append(
        f"Generated {meta['generated_at']} | true pairs analysed: "
        f"**{fmt_int(meta['n_true_pairs'])}** | sources: {', '.join(meta['sources'])} | "
        f"split: {meta['split']}"
    )
    lines.append("")
    lines.append(
        "Measurement only: no blocker, embedding, model or threshold is implemented or chosen "
        "by this script, and it does not modify any pipeline artifact."
    )
    lines.append("")
    lines.append("## What this run says")
    lines.append("")
    lines.append(
        f"- **0.2 coverage** - the four cheap signals reach "
        f"{coverage['union']['pct_of_pairs']:.2f}% of true pairs (exact name "
        f"{coverage['signals'][EXACT_SIGNAL]['pct_of_pairs']:.2f}%, rare token "
        f"{coverage['signals'][TOKEN_SIGNAL]['pct_of_pairs']:.2f}%, char 3-gram "
        f"{coverage['signals'][CHAR_SIGNAL]['pct_of_pairs']:.2f}%, address "
        f"{coverage['signals'][ADDRESS_SIGNAL]['pct_of_pairs']:.2f}%). Irreducible residue: "
        f"{_plural(coverage['residue']['n_pairs'], 'pair')} "
        f"({coverage['residue']['pct_of_pairs']:.2f}%)."
    )
    all_overlap = overlap["slices"]["all"]
    jac = all_overlap["jaccard"]
    lines.append(
        f"- **0.3 address** - over the {_plural(all_overlap['counts']['n_both_addresses_present'], 'pair')} "
        f"with two non-empty addresses: median Jaccard {jac.get('p50', 0):.3f}, mean "
        f"{jac.get('mean', 0):.3f}; "
        f"{_plural(all_overlap['counts']['n_zero_shared_tokens'], 'pair')} "
        f"({all_overlap['pct']['zero_shared_tokens_of_comparable']:.2f}%) share no token at all."
    )
    differing_size = rarity["subset_sizes"]["name_norm_differs"]
    usable = next(
        (row for row in rarity["usability_by_cap"]["name_norm_differs"] if row["cap"] == 1_000), None
    )
    lines.append(
        f"- **0.4 tokens** - {fmt_int(tokens['n_unique_tokens'])} unique name tokens over "
        f"{fmt_int(tokens['n_entities_in_scope'])} target entities. Of the "
        f"{_plural(differing_size, 'true pair')} whose names differ, "
        + (
            f"{usable['pct_of_subset']:.2f}% share at least one token with a posting list of "
            "<= 1000, so a token blocker could reach them."
            if usable
            else "none share a usable token."
        )
    )
    if zero.get("n_zero_match_entities"):
        kinds = zero["by_candidate_kind"]
        lines.append(
            f"- **0.5 zero-match** - of {fmt_int(zero['n_zero_match_entities'])} S1 entities with "
            f"no true match, {fmt_int(kinds['exact_name_norm']['n_entities'])} "
            f"({kinds['exact_name_norm']['pct_of_zero_match']:.2f}%) would still receive exact-name "
            "candidates: pure false positives."
        )
    lines.append("")

    # ---- 0.2 -------------------------------------------------------------
    lines.extend(_render_coverage(coverage))

    # ---- 0.3 -------------------------------------------------------------
    lines.append("## Phase 0.3 - address token overlap")
    lines.append("")
    lines.extend(
        _table(
            ["metric", "n", "mean", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"],
            [
                [
                    metric,
                    fmt_int(stats.get("n", 0)),
                    f"{stats.get('mean', 0):.4f}",
                    *[f"{stats.get(key, 0):.4f}" for key in ("p10", "p25", "p50", "p75", "p90", "p95", "p99")],
                    f"{stats.get('max', 0):.4f}",
                ]
                for metric, stats in (
                    ("source1 address tokens", all_overlap["source1_address_tokens"]),
                    ("target address tokens", all_overlap["target_address_tokens"]),
                    ("shared address tokens", all_overlap["shared_address_tokens"]),
                    ("Jaccard (both present)", all_overlap["jaccard"]),
                    ("overlap coefficient (both present)", all_overlap["overlap_coefficient"]),
                )
            ],
        )
    )
    lines.append("")
    lines.append("Shared address tokens (all true pairs):")
    lines.append("")
    lines.extend(
        _table(
            ["shared tokens", "pairs"],
            [[bucket, fmt_int(count)] for bucket, count in overlap["shared_token_histogram"].items()],
        )
    )
    lines.append("")
    lines.extend(
        _table(
            ["slice", "pairs", "both addresses", "zero shared", "zero shared %", ">=1 shared", ">=2 shared", ">=3 shared", "Jaccard >= 0.5", "overlap >= 0.8"],
            [
                [
                    label,
                    fmt_int(entry["counts"]["n_pairs"]),
                    fmt_int(entry["counts"]["n_both_addresses_present"]),
                    fmt_int(entry["counts"]["n_zero_shared_tokens"]),
                    f"{entry['pct']['zero_shared_tokens_of_comparable']:.2f}",
                    fmt_int(entry["counts"]["n_at_least_1_shared_token"]),
                    fmt_int(entry["counts"]["n_at_least_2_shared_tokens"]),
                    fmt_int(entry["counts"]["n_at_least_3_shared_tokens"]),
                    fmt_int(entry["counts"]["n_jaccard_at_least_0_5"]),
                    fmt_int(entry["counts"]["n_overlap_coefficient_at_least_0_8"]),
                ]
                for label, entry in (("all", all_overlap), *[(name, overlap["slices"][name]) for name in meta["sources"]])
            ],
        )
    )
    lines.append("")
    lines.append(
        "By name equality (all pairs) - when the names already agree the exact blocker has "
        "proposed the pair and the address is free confirmation; when they differ the address "
        "is the only thing left that could confirm it:"
    )
    lines.append("")
    lines.extend(
        _table(
            ["group", "pairs", "median Jaccard", "mean Jaccard", "zero shared %", "overlap >= 0.8"],
            [
                [
                    name,
                    fmt_int(entry["counts"]["n_pairs"]),
                    f"{entry['jaccard'].get('p50', 0):.4f}",
                    f"{entry['jaccard'].get('mean', 0):.4f}",
                    f"{entry['pct']['zero_shared_tokens_of_comparable']:.2f}",
                    fmt_int(entry["counts"]["n_overlap_coefficient_at_least_0_8"]),
                ]
                for name, entry in all_overlap["by_group"].items()
            ],
        )
    )
    lines.append("")
    lines.append(f"> {overlap['note']}")
    lines.append("")

    # ---- 0.4 -------------------------------------------------------------
    lines.append("## Phase 0.4 - token frequency and posting statistics")
    lines.append("")
    posting = tokens["posting_size"]
    lines.append(
        f"Combined scope: {fmt_int(tokens['n_entities_in_scope'])} target entities, "
        f"{fmt_int(tokens['n_unique_tokens'])} unique name tokens, "
        f"{fmt_int(tokens['n_token_occurrences'])} token occurrences "
        f"({tokens['postings_per_entity']} distinct tokens per entity on average). "
        "Per-source figures are in `blocking_statistics_report.json` under "
        "`token_frequency`."
    )
    lines.append("")
    lines.extend(
        _table(
            ["posting size", "value"],
            [
                ["median", fmt_int(posting["median"])],
                ["mean", f"{posting['mean']:.2f}"],
                ["p50", fmt_int(posting["p50"])],
                ["p75", fmt_int(posting["p75"])],
                ["p90", fmt_int(posting["p90"])],
                ["p95", fmt_int(posting["p95"])],
                ["p99", fmt_int(posting["p99"])],
                ["p99.9", fmt_int(posting["p99_9"])],
                ["max", fmt_int(posting["max"])],
            ],
        )
    )
    lines.append("")
    lines.append("Tokens exceeding a candidate-posting cap (each cap applied to the whole corpus):")
    lines.append("")
    lines.extend(
        _table(
            ["cap", "tokens over cap", "% of unique tokens", "postings over cap", "% of all postings"],
            [
                [fmt_int(row["cap"]), fmt_int(row["n_tokens_over_cap"]), f"{row['pct_tokens_over_cap']:.4f}", fmt_int(row["n_postings_over_cap"]), f"{row['pct_postings_over_cap']:.2f}"]
                for row in tokens["posting_caps"]
            ],
        )
    )
    lines.append("")
    lines.append("Most frequent tokens:")
    lines.append("")
    lines.extend(
        _table(
            ["rank", "token", "document frequency", "% of entities", "idf"],
            [
                [row["rank"], f"`{row['token']}`", fmt_int(row["document_frequency"]), f"{row['pct_of_entities']:.4f}", f"{row['idf']:.3f}" if row["idf"] is not None else ""]
                for row in tokens["top_tokens"][:20]
            ],
        )
    )
    lines.append("")
    lines.append("### Can a token blocker reach the pairs exact-name blocking misses?")
    lines.append("")
    differing = "name_norm_differs"
    lines.append(
        f"True pairs whose `name_norm` differs: {fmt_int(rarity['subset_sizes'][differing])}. "
        f"Pairs sharing at least one name token: "
        f"{fmt_int(rarity['has_shared_token_at_all'][differing]['n_pairs'])} "
        f"({rarity['has_shared_token_at_all'][differing]['pct_of_subset']:.2f}%)."
    )
    lines.append("")
    lines.extend(
        _table(
            ["max posting size allowed", "pairs with a usable token", "% of differing pairs", "pairs missed", "% missed"],
            [
                [fmt_int(row["cap"]), fmt_int(row["n_pairs_with_usable_token"]), f"{row['pct_of_subset']:.2f}", fmt_int(row["n_pairs_missed"]), f"{row['pct_missed']:.2f}"]
                for row in rarity["usability_by_cap"][differing]
            ],
        )
    )
    lines.append("")
    lines.extend(
        _table(
            ["subset", "pairs", f"usable token @ cap {fmt_int(CANDIDATE_VOLUME_CAP)}"],
            [
                [
                    name,
                    fmt_int(rarity["subset_sizes"][name]),
                    f"{next(row['pct_of_subset'] for row in rows if row['cap'] == CANDIDATE_VOLUME_CAP):.2f}%",
                ]
                for name, rows in rarity["usability_by_cap"].items()
            ],
        )
    )
    lines.append("")
    lines.append("Rarest shared name token (document frequency), over pairs that share one:")
    lines.append("")
    lines.extend(
        _table(
            ["subset", "pairs", "mean", "p25", "p50", "p75", "p90", "p99"],
            [
                [
                    name,
                    fmt_int(stats.get("n", 0)),
                    f"{stats.get('mean', 0):.1f}",
                    *[f"{stats.get(key, 0):.1f}" for key in ("p25", "p50", "p75", "p90", "p99")],
                ]
                for name, stats in rarity["rarest_shared_token_df"].items()
            ],
        )
    )
    lines.append("")
    volume = rarity["candidate_volume_at_cap"][differing]
    lines.append(
        f"Rough candidate volume at cap {fmt_int(CANDIDATE_VOLUME_CAP)} for the differing pairs: "
        f"{fmt_int(volume['estimated_candidates'])} candidate rows before deduplication "
        f"(mean {volume.get('mean_postings_per_pair', 0)} per pair, max "
        f"{fmt_int(volume.get('max_postings_per_pair', 0))})."
    )
    lines.append("")
    lines.append(f"> {rarity['note']}")
    lines.append("")
    lines.extend(_render_census(census, exact_index))

    # ---- 0.5 -------------------------------------------------------------
    lines.append("## Phase 0.5 - zero-match source1 entities")
    lines.append("")
    if not zero.get("n_zero_match_entities"):
        lines.append("No source1 entity has an empty ground truth in this run.")
    else:
        lines.append(
            f"Source1 entities with no true match: **{fmt_int(zero['n_zero_match_entities'])}** "
            f"({zero['pct_of_all_source1']:.2f}% of the "
            f"{fmt_int(meta['n_source1_entities'])} source1 entities in the ground truth)."
        )
        lines.append("")
        lines.extend(
            _table(
                ["candidate kind", "entities", "% of zero-match"],
                [
                    ["exact name_norm match", fmt_int(zero["by_candidate_kind"]["exact_name_norm"]["n_entities"]), f"{zero['by_candidate_kind']['exact_name_norm']['pct_of_zero_match']:.2f}"],
                    ["name_key only (spacing differs)", fmt_int(zero["by_candidate_kind"]["name_key_only"]["n_entities"]), f"{zero['by_candidate_kind']['name_key_only']['pct_of_zero_match']:.2f}"],
                    ["no candidate at all", fmt_int(zero["by_candidate_kind"]["no_candidate_at_all"]["n_entities"]), f"{zero['by_candidate_kind']['no_candidate_at_all']['pct_of_zero_match']:.2f}"],
                ],
            )
        )
        lines.append("")
        lines.append(
            "The first two rows are false-positive exposure: entities with NO true match that the "
            "exact-name blocker nevertheless proposes candidates for."
        )
        lines.append("")
        lines.extend(
            _table(
                ["scope", "entities with candidates", "%", "min", "p50", "p90", "p95", "p99", "max"],
                [
                    [
                        label,
                        fmt_int(entry["n_entities"]),
                        f"{entry['pct_of_zero_match']:.2f}",
                        fmt_int(entry["candidate_count"].get("min", 0)),
                        fmt_int(entry["candidate_count"].get("p50", 0)),
                        fmt_int(entry["candidate_count"].get("p90", 0)),
                        fmt_int(entry["candidate_count"].get("p95", 0)),
                        fmt_int(entry["candidate_count"].get("p99", 0)),
                        fmt_int(entry["candidate_count"].get("max", 0)),
                    ]
                    for label, entry in (
                        ("all sources (name_norm)", zero["exact_name_norm"]),
                        ("all sources (name_key)", zero["name_key_total"]),
                        *zero["per_source"].items(),
                    )
                ],
            )
        )
        lines.append("")
        lines.append("Candidates per zero-match entity (only entities with at least one):")
        lines.append("")
        lines.extend(
            _table(
                ["candidates", "entities"],
                [[bucket, fmt_int(count)] for bucket, count in zero["exact_name_norm"]["buckets"].items()],
            )
        )
        lines.append("")
        risk = zero["false_positive_risk"]
        lines.append(
            f"Address support for those candidates: {fmt_int(risk['n_candidate_pairs'])} candidate "
            f"pairs, of which {fmt_int(risk['n_pairs_both_addresses_present'])} have a non-empty "
            f"address on both sides. {fmt_int(risk['n_pairs_with_zero_address_overlap'])} of those "
            f"({risk['pct_pairs_with_zero_address_overlap']:.2f}%) share **no** address token - an "
            "address overlap rule would reject them outright; the rest need a similarity threshold."
        )
        lines.append("")
        lines.extend(
            _table(
                ["address Jaccard bucket", "candidate pairs"],
                [[bucket, fmt_int(count)] for bucket, count in risk["address_jaccard_histogram"].items()],
            )
        )
        lines.append("")
        lines.append(
            f"{fmt_int(risk['n_entities_with_any_address_support'])} zero-match entities have at "
            "least one candidate whose address overlaps by Jaccard >= 0.5, i.e. a plausible false "
            "positive that address support alone would not reject."
        )
        if zero["top_entities_by_candidate_count"]:
            lines.append("")
            lines.append("Zero-match entities with the most exact-name candidates:")
            lines.append("")
            lines.extend(
                _table(
                    ["source1 entity", "name_norm", "candidates", "S2", "S3", "best address Jaccard"],
                    [
                        [row["source1_entity_id"], row["name_norm"], fmt_int(row["n_candidates"]), fmt_int(row["n_candidates_s2"]), fmt_int(row["n_candidates_s3"]), row["best_address_jaccard"] if row["best_address_jaccard"] is not None else "-"]
                        for row in zero["top_entities_by_candidate_count"]
                    ],
                )
            )
    lines.append("")
    if zero.get("note"):
        lines.append(f"> {zero['note']}")
        lines.append("")
    lines.append("")
    lines.append(
        "No threshold is chosen here and no candidate file is written: this run only measures how "
        "much of the no-match population an exact-name blocker exposes."
    )
    lines.append("")
    lines.append("## Run details")
    lines.append("")
    lines.extend(
        _table(
            ["key", "value"],
            [
                ["split", meta["split"]],
                ["sources", ", ".join(meta["sources"])],
                ["workers", meta["workers"]],
                ["chunk pairs", fmt_int(meta["chunk_pairs"])],
                ["limit pairs", meta["limit_pairs"] if meta["limit_pairs"] is not None else "none"],
                ["exact blocker key field", meta["exact_blocker_key_field"]],
                ["source1 entities", fmt_int(meta["n_source1_entities"])],
                ["true pairs analysed", fmt_int(meta["n_true_pairs"])],
                ["shared tokens not in the frequency table", fmt_int(meta["n_unknown_df"])],
                ["elapsed", f"{meta['elapsed_minutes']:.2f} min"],
                ["prepared dir", meta["prepared_dir"]],
            ],
        )
    )
    lines.append("")
    lines.append(
        "Every figure above is from the single run recorded under `Run details`, over the split "
        "and sources named there. Nothing in this report is a blocker parameter: no cap, weight or "
        "threshold is chosen by this script."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def _write_coverage_csv(path: Path, report: dict) -> None:
    """Tidy Phase 0.2: one row per (section, scope, signal/threshold, metric).

    ``section=signal`` - one row per signal and reference threshold.
    ``section=combination`` - the joint table, one row per reached-by combination
    (including the ``none`` row, which is the irreducible residue).
    ``section=residue_category`` / ``residue_reason`` - what the residue is made of.
    ``section=coverage_by_category`` - how each name-difference category is covered.
    ``section=sensitivity`` - coverage against candidate cost at every threshold.
    ``section=per_source`` - S2/S3 breakdown.
    ``section=entity_outcome`` - the entity-weighted view.
    """
    coverage = report["signal_coverage"]
    rows: list[dict[str, Any]] = []

    def _add(section: str, scope: str, key: str, metric: str, value: Any) -> None:
        rows.append(
            {"section": section, "scope": scope, "key": key, "metric": metric, "value": value}
        )

    for signal, entry in coverage["signals"].items():
        for metric in ("n_pairs", "pct_of_pairs", "n_entities", "pct_of_entities_with_pairs"):
            _add("signal", "all", signal, metric, entry[metric])
        _add(
            "signal",
            "all",
            signal,
            "reference_threshold",
            entry["reference_threshold"] if entry["reference_threshold"] is not None else "",
        )
    for row in coverage["union"]["by_combination"]:
        for metric in ("n_pairs", "pct_of_pairs", "n_entities", "pct_of_entities_with_pairs"):
            _add("combination", "all", row["signals"], metric, row[metric])
    for label in ("all_pairs_reached", "some_pairs_reached", "no_pair_reached"):
        entry = coverage["union"]["entity_outcomes"][label]
        _add("entity_outcome", "all", label, "n_entities", entry["n_entities"])
        _add("entity_outcome", "all", label, "pct", entry["pct"])
    for source, entry in coverage["per_source"].items():
        _add("per_source", source, "union", "n_pairs", entry["union"]["n_pairs"])
        _add("per_source", source, "union", "pct_of_pairs", entry["union"]["pct_of_pairs"])
        _add("per_source", source, "residue", "n_pairs", entry["residue"]["n_pairs"])
        _add("per_source", source, "residue", "pct_of_pairs", entry["residue"]["pct_of_pairs"])
        for signal, signal_entry in entry["signals"].items():
            _add("per_source", source, signal, "n_pairs", signal_entry["n_pairs"])
            _add("per_source", source, signal, "pct_of_pairs", signal_entry["pct_of_pairs"])
    for signal, grid in coverage["sensitivity"].items():
        for row in grid:
            _add("sensitivity", signal, str(row["threshold"]), "n_pairs", row["n_pairs"])
            _add("sensitivity", signal, str(row["threshold"]), "pct_of_pairs", row["pct_of_pairs"])
            _add(
                "sensitivity",
                signal,
                str(row["threshold"]),
                "candidate_rows",
                row["candidate_cost"]["n_candidates"]
                if row["candidate_cost"]["n_candidates"] is not None
                else "",
            )
            _add(
                "sensitivity",
                signal,
                str(row["threshold"]),
                "candidate_cost_kind",
                row["candidate_cost"]["kind"],
            )
    residue = coverage["residue"]
    for metric in ("n_pairs", "pct_of_pairs", "n_entities"):
        _add("residue", "all", "none", metric, residue[metric])
    for name, entry in residue["by_name_difference_category"].items():
        _add("residue_category", "all", name, "n_pairs", entry["n_pairs"])
        _add("residue_category", "all", name, "pct_of_residue", entry["pct_of_residue"])
    for name, entry in residue["by_name_difference_reason"].items():
        _add("residue_reason", "all", name, "n_pairs", entry["n_pairs"])
        _add("residue_reason", "all", name, "pct_of_residue", entry["pct_of_residue"])
    for name, count in (residue.get("by_script_relation") or {}).items():
        _add("residue_script", "all", name, "n_pairs", count)
    for source, entry in residue["per_source"].items():
        _add("residue_source", source, "none", "n_pairs", entry["n_pairs"])
        _add("residue_source", source, "none", "pct_of_pairs", entry["pct_of_pairs"])
    for name, entry in coverage["coverage_by_name_difference_category"].items():
        for signal, signal_entry in entry["signals"].items():
            _add("coverage_by_category", name, signal, "n_pairs", signal_entry["n_pairs"])
            _add(
                "coverage_by_category", name, signal, "pct_of_category", signal_entry["pct_of_category"]
            )
        _add("coverage_by_category", name, "union", "n_pairs", entry["union"]["n_pairs"])
        _add("coverage_by_category", name, "union", "pct_of_category", entry["union"]["pct_of_category"])
        _add("coverage_by_category", name, NO_SIGNAL_LABEL, "n_pairs", entry["residue"]["n_pairs"])

    pd.DataFrame(
        rows, columns=["section", "scope", "key", "metric", "value"]
    ).to_csv(path, index=False, encoding="utf-8")


def _write_census_csv(path: Path, report: dict) -> None:
    """Tidy Phase 0.4 census: one row per (section, scope, key, metric).

    ``section=token_volume`` / ``exact_volume`` - the distributions.
    ``section=token_cap`` - volume and reachability at every cap.
    ``section=structural_zero`` - why an entity has no usable token.
    ``section=ratio`` - candidate-to-truth ratio.
    ``section=top_entity`` - the largest volumes, by name.
    ``section=reachability`` - entities reachable by neither cheap signal.
    ``section=posting_list`` - the largest exact-name posting lists.
    """
    census = report["candidate_census"]
    exact_index = report["exact_name_census"]
    token = census["token_signal"]
    rows: list[dict[str, Any]] = []

    def _add(section: str, scope: str, key: str, metric: str, value: Any) -> None:
        rows.append(
            {"section": section, "scope": scope, "key": key, "metric": metric, "value": value}
        )

    for scope, stats in (
        ("all_entities", token["estimated_candidates_all_entities"]),
        ("reachable_entities", token["estimated_candidates_reachable_entities"]),
    ):
        for statistic, value in stats.items():
            _add("token_volume", scope, str(token["cap"]), statistic, value)
    _add("token_volume", "all_entities", str(token["cap"]), "total_estimated_candidate_rows", token["total_estimated_candidate_rows"])
    for statistic, value in token["reachable_entities_p50_p90_p99_max"].items():
        _add("token_volume", "reachable_entities", str(token["cap"]), statistic, value)
    for row in token["by_cap"]:
        for statistic in (
            "n_entities_with_a_candidate",
            "pct_of_entities",
            "n_structural_zero_entities",
            "pct_structural_zero",
            "total_estimated_candidate_rows",
            "mean_postings_per_reached_entity",
            "max_postings_per_reached_entity",
        ):
            _add("token_cap", "all_entities", str(row["cap"]), statistic, row[statistic])
    zero = token["structural_zero_entities"]
    _add("structural_zero", "all_entities", str(token["cap"]), "n_entities", zero["n_entities"])
    _add("structural_zero", "all_entities", str(token["cap"]), "pct", zero["pct"])
    for reason, count in zero["by_reason"].items():
        _add("structural_zero", "all_entities", reason, "n_entities", count)
    ratio = token["candidate_to_truth_ratio"]
    for statistic in (
        "aggregate",
        "n_entities_with_analysed_pairs",
        "n_source1_entities_without_analysed_pairs",
        "n_analysed_true_pairs",
        "n_entities_below_one",
        "pct_entities_below_one",
        "n_entities_with_zero_candidates",
        "pct_entities_with_zero_candidates",
    ):
        _add("ratio", "entities_with_pairs", "candidate_to_truth", statistic, ratio[statistic])
    for statistic, value in ratio["per_entity"].items():
        _add("ratio", "entities_with_pairs", "candidate_to_truth", statistic, value)
    for row in token["top_entities_by_estimated_candidates"]:
        for statistic in (
            "name_norm",
            "n_name_tokens",
            "rarest_usable_token",
            "rarest_usable_token_df",
            "estimated_candidates",
            "analysed_true_pairs",
        ):
            _add("top_entity", "token_signal", row["source1_entity_id"], statistic, row[statistic])
    if "reachability" in census:
        for key, value in census["reachability"].items():
            if isinstance(value, (int, float)):
                _add("reachability", "all_entities", key, "n_entities", value)
            elif isinstance(value, dict):
                for statistic, inner in value.items():
                    if isinstance(inner, (int, float)):
                        _add("reachability", "all_entities", key, statistic, inner)
    exact_signal = census["exact_name_signal"]
    if exact_signal.get("available"):
        for scope, stats in (
            ("all_entities", exact_signal["candidates_all_entities"]),
            ("reachable_entities", exact_signal["candidates_reachable_entities"]),
        ):
            for statistic, value in stats.items():
                _add("exact_volume", scope, "exact_name", statistic, value)
        for statistic in (
            "n_entities_with_a_candidate",
            "pct_of_entities",
            "total_candidate_rows",
            "mean_per_entity",
        ):
            _add("exact_volume", "all_entities", "exact_name", statistic, exact_signal[statistic])
        for row in exact_signal["top_entities_by_exact_candidates"]:
            _add("top_entity", "exact_name", row["source1_entity_id"], "exact_candidates", row["exact_candidates"])
            _add("top_entity", "exact_name", row["source1_entity_id"], "name_norm", row["name_norm"])
    for source, entry in exact_index.get("per_source", {}).items():
        if not entry["available"]:
            _add("posting_list", source, "unavailable", "reason", entry["reason"])
            continue
        for statistic, value in entry["describe"].items():
            _add("posting_list", source, "index", statistic, value)
        for statistic, value in entry["duplicate_keys"].items():
            _add("posting_list", source, "index", statistic, value)
        for statistic, value in entry["source1_query"].items():
            if isinstance(value, (int, float)):
                _add("posting_list", source, "source1_query", statistic, value)
        for row in entry["largest_posting_lists"]:
            _add("posting_list", source, row["key"], "postings", row["postings"])

    pd.DataFrame(
        rows, columns=["section", "scope", "key", "metric", "value"]
    ).to_csv(path, index=False, encoding="utf-8")


def _write_address_csv(path: Path, report: dict) -> None:
    """Tidy: one row per (scope, group, metric, statistic)."""
    rows: list[dict[str, Any]] = []
    metrics = (
        ("source1_address_tokens", "tokens"),
        ("target_address_tokens", "tokens"),
        ("shared_address_tokens", "tokens"),
        ("jaccard", "ratio"),
        ("overlap_coefficient", "ratio"),
    )
    for scope, entry in report["address_overlap"]["slices"].items():
        groups = {"all_pairs": entry, **entry.get("by_group", {})}
        for group, group_entry in groups.items():
            for metric, unit in metrics:
                for statistic, value in group_entry.get(metric, {}).items():
                    rows.append(
                        {
                            "scope": scope,
                            "group": group,
                            "metric": metric,
                            "statistic": statistic,
                            "value": value,
                            "unit": unit,
                        }
                    )
            for key, value in group_entry.get("counts", {}).items():
                rows.append(
                    {
                        "scope": scope,
                        "group": group,
                        "metric": "counts",
                        "statistic": key,
                        "value": value,
                        "unit": "pairs",
                    }
                )
            for key, value in group_entry.get("pct", {}).items():
                rows.append(
                    {
                        "scope": scope,
                        "group": group,
                        "metric": "pct",
                        "statistic": key,
                        "value": value,
                        "unit": "percent",
                    }
                )
    for bucket, count in report["address_overlap"]["shared_token_histogram"].items():
        rows.append(
            {
                "scope": "all",
                "group": "all_pairs",
                "metric": "shared_token_histogram",
                "statistic": bucket,
                "value": count,
                "unit": "pairs",
            }
        )
    pd.DataFrame(rows, columns=["scope", "group", "metric", "statistic", "value", "unit"]).to_csv(
        path, index=False, encoding="utf-8"
    )


def _write_token_csv(path: Path, report: dict) -> None:
    """Tidy token frequency and posting-cap table.

    ``section=top_token``: ``key`` is the token, ``count`` its document frequency
    (entities containing it), ``share`` the percentage of in-scope entities.

    ``section=posting_cap``: ``key`` is the cap, ``count`` the number of unique
    tokens whose posting list is longer than the cap, ``share`` the percentage of
    all postings those tokens account for.
    """
    rows: list[dict[str, Any]] = []
    for scope, tokens in report["token_frequency"].items():
        for row in tokens.get("top_tokens", []):
            rows.append(
                {
                    "scope": scope,
                    "section": "top_token",
                    "key": row["token"],
                    "count": row["document_frequency"],
                    "share": row["pct_of_entities"],
                    "idf": row["idf"],
                }
            )
        for row in tokens.get("posting_caps", []):
            rows.append(
                {
                    "scope": scope,
                    "section": "posting_cap",
                    "key": row["cap"],
                    "count": row["n_tokens_over_cap"],
                    "share": row["pct_postings_over_cap"],
                    "idf": None,
                }
            )
    pd.DataFrame(rows, columns=["scope", "section", "key", "count", "share", "idf"]).to_csv(
        path, index=False, encoding="utf-8"
    )


def _write_zero_match_csv(path: Path, report: dict) -> None:
    """Tidy: ``section,key,count,share`` over the zero-match population."""
    rows: list[dict[str, Any]] = []
    zero = report["zero_match"]
    n_zero = zero.get("n_zero_match_entities", 0)
    rows.append(
        {
            "section": "summary",
            "key": "all_source1_entities",
            "count": report["meta"]["n_source1_entities"],
            "share": 100.0,
        }
    )
    rows.append(
        {
            "section": "summary",
            "key": "zero_match_entities",
            "count": n_zero,
            "share": zero.get("pct_of_all_source1", 0.0),
        }
    )
    if n_zero:
        for kind, entry in zero["by_candidate_kind"].items():
            rows.append(
                {
                    "section": "candidate_kind",
                    "key": kind,
                    "count": entry["n_entities"],
                    "share": entry["pct_of_zero_match"],
                }
            )
        for scope, entry in (("all", zero["exact_name_norm"]), *zero["per_source"].items()):
            rows.append(
                {
                    "section": "candidate_count",
                    "key": f"{scope}_entities_with_candidates",
                    "count": entry["n_entities"],
                    "share": entry["pct_of_zero_match"],
                }
            )
            for bucket, count in entry["buckets"].items():
                rows.append(
                    {
                        "section": "candidate_count_bucket",
                        "key": f"{scope}_{bucket}",
                        "count": count,
                        "share": _pct(count, entry["n_entities"]),
                    }
                )
            for statistic in ("min", "p50", "p90", "p95", "p99", "max"):
                value = entry["candidate_count"].get(statistic)
                if value is not None:
                    rows.append(
                        {
                            "section": "candidate_count_statistic",
                            "key": f"{scope}_{statistic}",
                            "count": value,
                            "share": "",
                        }
                    )
        risk = zero["false_positive_risk"]
        for key in (
            "n_candidate_pairs",
            "n_pairs_both_addresses_present",
            "n_pairs_with_zero_address_overlap",
            "n_pairs_with_an_empty_address",
            "n_entities_with_any_address_support",
        ):
            rows.append({"section": "false_positive_risk", "key": key, "count": risk[key], "share": ""})
        rows.append(
            {
                "section": "false_positive_risk",
                "key": "pct_pairs_with_zero_address_overlap",
                "count": "",
                "share": risk["pct_pairs_with_zero_address_overlap"],
            }
        )
        for bucket, count in risk["address_jaccard_histogram"].items():
            rows.append(
                {
                    "section": "address_jaccard_bucket",
                    "key": bucket,
                    "count": count,
                    "share": _pct(count, risk["n_pairs_both_addresses_present"]),
                }
            )
    pd.DataFrame(rows, columns=["section", "key", "count", "share"]).to_csv(
        path, index=False, encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 0.2-0.5: signal coverage and complementarity, address overlap, token "
            "frequency and the candidate census, zero-match statistics. Measurement only "
            "- builds no blocker and writes no candidate file."
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="default: <work_dir>/analysis")
    parser.add_argument(
        "--split",
        default="train",
        help="data split to analyse; only 'train' has a ground truth",
    )
    parser.add_argument(
        "--sources",
        default="source2,source3",
        help="comma-separated target sources to analyse",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="0 = auto (config, else the physical core count, clamped to the chunk count)",
    )
    parser.add_argument("--chunk-pairs", type=int, default=100_000, help="pairs per work chunk")
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help=(
            "analyse only the first N true pairs. Limits the PAIR phases (0.2-0.4); the "
            "corpus scans for Phase 0.4/0.5 still read every row."
        ),
    )
    parser.add_argument("--top-tokens", type=int, default=200, help="most frequent tokens written to the CSV")
    parser.add_argument(
        "--no-name-categories",
        action="store_true",
        help=(
            "skip the Phase 0.1 name-difference classification of every true pair. It "
            "roughly halves the per-pair pass; the coverage, residue and census figures "
            "are unaffected, only the residue cross-tab and the by-category table go."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse the checkpointed results of completed phases from "
            "<output-dir>/_ckpt/ instead of recomputing them. A checkpoint whose "
            "inputs no longer match (different sources, --limit-pairs, chunk size or "
            "prepared data) is ignored and recomputed."
        ),
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help=(
            "record a per-phase wall-clock breakdown in the report meta and the log. "
            "Off by default because the numbers are machine-dependent; turn it on for "
            "a profiling run to see which phase actually dominates."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Phase-level checkpointing (--resume) and per-phase timing
# ---------------------------------------------------------------------------
# A checkpoint is one ``.npz`` plus an atomic ``.done.json`` marker. The marker
# carries a fingerprint of everything the phase's result depends on, so a
# checkpoint written for a different split, source list, --limit-pairs, chunk size
# or prepared corpus is ignored rather than silently trusted. Nothing here runs
# unless --resume is passed, which keeps the default output byte-for-byte what the
# tests pin.
#
# Deliberately NOT checkpointed: the per-source corpus scan and the two censuses.
# The scan's state is ~500MB of per-pair target strings plus multi-million-entry
# token counters - the "enormous intermediate artifact" the brief warns against -
# for a single sequential pass; the censuses return nested report structures rather
# than arrays, and are seconds of work. The pair pass is checkpointed because its
# state is exactly eight flat numeric arrays. If a --timings run shows the scan
# dominating wall-clock, the answer is to parallelize it, not to checkpoint it.
CKPT_DIRNAME = "_ckpt"
CKPT_VERSION = 1


class _PhaseCheckpoint:
    """Resume support for phases whose state is compact and side-effect free."""

    def __init__(
        self, output_dir: Path, fingerprint: str, enabled: bool, log: logging.Logger
    ) -> None:
        self.dir = Path(output_dir) / CKPT_DIRNAME
        self.fingerprint = fingerprint
        self.enabled = enabled
        self.log = log
        # Names of phases actually restored, so the report can distinguish
        # "--resume was passed" from "--resume saved work".
        self.restored: list[str] = []

    def _payload_path(self, name: str) -> Path:
        return self.dir / f"{name}.npz"

    def _marker_path(self, name: str) -> Path:
        return self.dir / f"{name}.done.json"

    def load(self, name: str) -> Optional[dict[str, np.ndarray]]:
        """Return a completed phase's arrays, or ``None`` when it must be recomputed."""
        if not self.enabled:
            return None
        marker = self._marker_path(name)
        if not marker.exists() or not self._payload_path(name).exists():
            return None
        try:
            meta = read_json(marker)
        except Exception as exc:
            self.log.warning("checkpoint %s has an unreadable marker (%s); recomputing", name, exc)
            return None
        if meta.get("version") != CKPT_VERSION or meta.get("fingerprint") != self.fingerprint:
            self.log.info("checkpoint %s does not match this run's inputs; recomputing", name)
            return None
        try:
            with np.load(self._payload_path(name), allow_pickle=False) as data:
                arrays = {key: data[key] for key in data.files}
        except Exception as exc:
            self.log.warning("checkpoint %s could not be read (%s); recomputing", name, exc)
            return None
        self.restored.append(name)
        self.log.info("resumed phase %r from checkpoint", name)
        return arrays

    def save(self, name: str, **arrays: np.ndarray) -> None:
        """Write a completed phase's arrays, then the marker that blesses them.

        The marker goes last and atomically, so an interruption can leave a stray
        ``.npz`` but never a marker pointing at a half-written payload. A scratch
        directory that cannot be written degrades to "no checkpoint", not a crash.
        """
        if not self.enabled:
            return
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self._payload_path(name).with_suffix(".npz.tmp")
            with open(tmp, "wb") as handle:
                np.savez(handle, **arrays)
            os.replace(tmp, self._payload_path(name))
            write_json(
                self._marker_path(name),
                {
                    "version": CKPT_VERSION,
                    "fingerprint": self.fingerprint,
                    "arrays": sorted(arrays),
                },
            )
        except OSError as exc:
            self.log.warning("could not write checkpoint %s (%s); continuing without it", name, exc)


def checkpoint_fingerprint(
    args: argparse.Namespace,
    sources: Sequence[str],
    config: dict,
    n_pairs: int,
) -> str:
    """Fingerprint of every input a checkpointed phase's result depends on.

    Anything that changes the numbers - the pair population, the chunking the
    per-pair arrays are built with, whether the name cross-tab ran, or which
    prepared corpus is behind them - must appear here, or a stale checkpoint gets
    silently reused and the report becomes unreproducible.
    """
    material = "|".join(
        str(part)
        for part in (
            CKPT_VERSION,
            args.split,
            ",".join(sources),
            args.limit_pairs,
            args.chunk_pairs,
            args.no_name_categories,
            n_pairs,
            config["resolved"]["prepared_dir"],
            REFERENCE_TOKEN_CAP,
        )
    )
    return f"{stable_hash64(material):016x}"


class _PhaseTimer:
    """Per-phase wall-clock, so a profiling run says where the time actually went."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.seconds: dict[str, float] = {}
        self._name: Optional[str] = None
        self._start: float = 0.0

    def start(self, name: str) -> None:
        if self.enabled:
            self._name, self._start = name, time.time()

    def stop(self) -> None:
        if self.enabled and self._name is not None:
            self.seconds[self._name] = round(time.time() - self._start, 3)
            self._name = None

    def summary(self) -> dict[str, float]:
        """Phases ordered slowest-first, which is the order that matters."""
        return dict(sorted(self.seconds.items(), key=lambda item: -item[1]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, overrides={"data_root": args.data_root, "work_dir": args.work_dir})
    log = setup_logging(
        LOG_NAME,
        log_dir=config["resolved"]["log_dir"],
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    set_seed(config.get("project", {}).get("seed", 42))

    if args.split != "train":
        raise SystemExit(
            f"--split {args.split!r} is not supported: the ground truth exists only for "
            "'train', and every phase here is defined against it"
        )
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    unknown = [item for item in sources if item not in SOURCE_CODES]
    if unknown:
        raise SystemExit(f"unknown source(s): {unknown}; expected source2/source3")

    log.info("=" * 78)
    log.info("Phase 0.2-0.5: blocking statistics")
    log.info(describe_environment(config))
    hardware = detect_hardware()
    for line in format_hardware_report(hardware).splitlines():
        log.info("  %s", line)
    log.info("=" * 78)
    started = time.time()

    # Resolved here rather than at the end so the checkpoint directory can live
    # under the caller's --output-dir, which is what makes a resume land on the
    # same run.
    output_dir = (
        Path(args.output_dir) if args.output_dir else Path(config["resolved"]["work_dir"]) / "analysis"
    )

    # -- 1. ground truth -> true pairs (no cross-product) -------------------
    ground_truth = load_ground_truth(config, log=log)
    n_source1 = ground_truth.n_entities
    lengths = ground_truth.lengths()
    owners = np.repeat(np.arange(n_source1, dtype=np.int64), lengths)
    target_codes = ground_truth.codes
    source_of_pair = (target_codes // 10**10).astype(np.int8)
    log_memory(log, "after ground truth")

    keep = np.isin(source_of_pair, [SOURCE_CODES[source] for source in sources])
    keep_indices = np.flatnonzero(keep)
    if args.limit_pairs is not None:
        keep_indices = keep_indices[: args.limit_pairs]
    owners = owners[keep_indices]
    target_codes = target_codes[keep_indices]
    source_of_pair = source_of_pair[keep_indices]
    n_pairs = len(keep_indices)
    log.info(
        "true pairs to analyse: %s (of %s in the ground truth)",
        fmt_int(n_pairs),
        fmt_int(len(ground_truth.codes)),
    )
    if n_pairs == 0:
        raise SystemExit("no ground-truth pair matched the requested sources")

    timer = _PhaseTimer(args.timings)
    ckpt = _PhaseCheckpoint(
        output_dir=output_dir,
        fingerprint=checkpoint_fingerprint(args, sources, config, n_pairs),
        enabled=args.resume,
        log=log,
    )
    if args.resume:
        log.info("resume enabled: completed phases will be reused from %s", ckpt.dir)

    # -- 2. source1 attributes, and the zero-match population ---------------
    s1 = _resolve_s1_columns(config, ground_truth, (NAME_NORM, NAME_KEY, ADDRESS_NORM), log)
    log_memory(log, "after source1 resolution")

    zero_positions = np.flatnonzero(lengths == 0)
    log.info(
        "source1 entities with no true match: %s of %s",
        fmt_int(len(zero_positions)),
        fmt_int(n_source1),
    )
    probe = _ZeroMatchProbe(
        entity_ids=ground_truth.entity_ids[zero_positions],
        names=s1.get(NAME_NORM, np.empty(0, dtype=object))[zero_positions],
        keys=(s1[NAME_KEY][zero_positions] if NAME_KEY in s1 else None),
        addresses=s1.get(ADDRESS_NORM, np.empty(0, dtype=object))[zero_positions],
        sources=sources,
    )

    # The full per-entity source1 columns stay alive until the census: they cost
    # object-array pointers to strings that are already in memory (~18MB each at
    # 2.2M rows), and the census needs every entity, not just the matched ones.
    all_names = s1.get(NAME_NORM, np.full(n_source1, "", dtype=object))
    names_a = all_names[owners]
    keys_a = s1.get(NAME_KEY, np.full(n_source1, "", dtype=object))[owners]
    addrs_a = s1.get(ADDRESS_NORM, np.full(n_source1, "", dtype=object))[owners]
    pairs_per_entity = np.bincount(owners, minlength=n_source1)
    del s1
    log_memory(log, "after per-pair source1 arrays")

    # -- 3. one streaming pass per target source ----------------------------
    token_counters: dict[str, Counter] = {}
    rows_per_source: dict[str, int] = {}
    needed_by_source: dict[int, np.ndarray] = {}
    resolved_by_source: dict[int, dict[str, Any]] = {}
    timer.start("corpus_scan")
    for source in sources:
        code = SOURCE_CODES[source]
        needed_by_source[code] = np.unique(target_codes[source_of_pair == code])
        token_counters[source] = Counter()
        resolved, rows_seen = _scan_target_source(
            config=config,
            source=source,
            needed_codes=needed_by_source[code],
            token_counter=token_counters[source],
            probe=probe,
            log=log,
        )
        resolved_by_source[code] = resolved
        rows_per_source[source] = rows_seen
        log_memory(log, f"after {source} scan")
    timer.stop()

    # -- 4. per-pair target attributes -------------------------------------
    names_b = np.full(n_pairs, "", dtype=object)
    keys_b = np.full(n_pairs, "", dtype=object)
    addrs_b = np.full(n_pairs, "", dtype=object)
    for code, resolved in resolved_by_source.items():
        mask = source_of_pair == code
        if not mask.any():
            continue
        needed = needed_by_source[code]
        slots = np.searchsorted(needed, target_codes[mask])
        np.clip(slots, 0, max(len(needed) - 1, 0), out=slots)
        names_b[mask] = resolved.get(NAME_NORM, np.full(len(needed), "", dtype=object))[slots]
        keys_b[mask] = resolved.get(NAME_KEY, np.full(len(needed), "", dtype=object))[slots]
        addrs_b[mask] = resolved.get(ADDRESS_NORM, np.full(len(needed), "", dtype=object))[slots]
    del resolved_by_source
    log_memory(log, "after per-pair target arrays")

    # -- 6. Phase 0.4: token frequency table (the lookup the pair pass needs)
    timer.start("token_frequency")
    per_scope, df_hashes, df_values = _token_frequency(
        token_counters, rows_per_source, args.top_tokens, log
    )
    del token_counters
    timer.stop()
    log_memory(log, "after token frequency table")

    # -- 7. Phases 0.3 / 0.4: sharded per-pair statistics ------------------
    # int16 for the token counts and int64 for the rarest-token document
    # frequency: ~160MB at 7.6M pairs, in place of a 7.6M-row Python object.
    name_matrix = np.zeros((n_pairs, 3), dtype=np.int16)
    addr_matrix = np.zeros((n_pairs, 3), dtype=np.int16)
    name_equal = np.zeros(n_pairs, dtype=bool)
    min_df = np.zeros(n_pairs, dtype=np.int64)
    char_sim = np.zeros(n_pairs, dtype=np.float32)
    categories = np.full(n_pairs, -1, dtype=np.int16)
    reasons = np.full(n_pairs, -1, dtype=np.int16)
    classify_names = not args.no_name_categories

    n_chunks = (n_pairs + args.chunk_pairs - 1) // args.chunk_pairs
    workers = _resolve_workers(
        args.workers, config.get("compute", {}).get("num_workers", 0), n_chunks, log
    )

    # Size the queue against a real RAM budget. The pool keeps several chunks in
    # flight to stay fed, and every queued chunk holds its pair strings as python
    # objects - at the core counts worth using here, that transient payload rather
    # than the report arrays is what decides peak RSS.
    bytes_per_pair = _estimate_payload_bytes_per_pair(
        names_a, names_b, keys_a, keys_b, addrs_a, addrs_b
    )
    window_size, chunk_pairs = plan_inflight_window(
        workers=workers,
        chunk_pairs=args.chunk_pairs,
        bytes_per_pair=bytes_per_pair,
        budget_bytes=_payload_budget_bytes(config, hardware),
        logger=log,
        label="pair chunks",
    )
    progress_step = max(1, chunk_pairs * 10)
    processed = 0
    n_unknown_df = 0

    def consume(block: np.ndarray, result: dict[str, np.ndarray]) -> None:
        """Fold one chunk of pair statistics into the per-pair arrays."""
        nonlocal processed, n_unknown_df
        chunk_names = result["name_counts"]
        shared_hashes = result["shared_hashes"]
        name_matrix[block] = chunk_names
        addr_matrix[block] = result["addr_counts"]
        name_equal[block] = result["name_equal"].astype(bool)
        char_sim[block] = result["char_sim"]
        if classify_names:
            categories[block] = result["category"]
            reasons[block] = result["reason"]
        if shared_hashes.size and df_hashes.size:
            # The workers return token HASHES, so every shared token of the whole
            # pair set is resolved here in one vectorized searchsorted instead of
            # a Python-level dictionary lookup per token.
            slots = np.searchsorted(df_hashes, shared_hashes)
            np.clip(slots, 0, df_hashes.size - 1, out=slots)
            matched = df_hashes[slots] == shared_hashes
            n_unknown_df += int((~matched).sum())
            values = np.where(matched, df_values[slots], np.int64(UNKNOWN_DF))
            counts = chunk_names[:, 2].astype(np.int64)
            nonempty = counts > 0
            if nonempty.any():
                # The shared hashes of pair i occupy [end - count, end), so
                # reduceat over those starts gives the rarest shared token of each
                # pair. Empty pairs contribute no elements, which makes the starts
                # of consecutive non-empty pairs adjacent - exactly what reduceat
                # needs.
                ends = np.cumsum(counts)
                starts = ends[nonempty] - counts[nonempty]
                min_df[block[nonempty]] = np.minimum.reduceat(values, starts)
        processed += len(block)
        if processed % progress_step < chunk_pairs:
            elapsed = time.time() - pair_started
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta = (n_pairs - processed) / rate if rate > 0 else float("inf")
            log.info(
                "  pair statistics: %s / %s (%.1f%%) | %s workers | %.0f pairs/s | ETA %.1f min | peak RSS %s",
                fmt_int(processed),
                fmt_int(n_pairs),
                100.0 * processed / max(n_pairs, 1),
                fmt_int(workers),
                rate,
                eta / 60.0,
                human_bytes(peak_rss_bytes() or current_rss_bytes() or 0),
            )

    timer.start("pair_statistics")
    pair_started = time.time()
    cached_pairs = ckpt.load("pair_statistics")
    if cached_pairs is not None:
        # Restore instead of recomputing. Every array here is an output of the
        # pair pass and nothing else, so a hit is a pure time saving.
        name_matrix[:] = cached_pairs["name_matrix"]
        addr_matrix[:] = cached_pairs["addr_matrix"]
        name_equal[:] = cached_pairs["name_equal"].astype(bool)
        char_sim[:] = cached_pairs["char_sim"]
        min_df[:] = cached_pairs["min_df"]
        if classify_names:
            categories[:] = cached_pairs["category"]
            reasons[:] = cached_pairs["reason"]
        n_unknown_df = int(cached_pairs["n_unknown_df"])
        processed = n_pairs
    else:
        chunks = _iter_pair_chunks(
            names_a, names_b, keys_a, keys_b, addrs_a, addrs_b, n_pairs, chunk_pairs
        )
        if workers == 1:
            log.info("running single-process (workers=1)")
            for block, payload in chunks:
                consume(block, _pair_statistics(payload, classify_names))
        else:
            # ProcessPoolExecutor.map() submits EVERY chunk before the first one is
            # consumed, which would queue the whole pair set in memory - the exact
            # thing chunking exists to avoid. A bounded sliding window keeps a fixed
            # number of chunks in flight - sized against a RAM budget, not a bare
            # multiplier - and consumes strictly in submission order, which also
            # keeps the run reproducible.
            window: deque = deque()
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for block, payload in chunks:
                    window.append((block, pool.submit(_pair_statistics, payload, classify_names)))
                    if len(window) >= window_size:
                        done_block, future = window.popleft()
                        consume(done_block, future.result())
                while window:
                    done_block, future = window.popleft()
                    consume(done_block, future.result())
        ckpt.save(
            "pair_statistics",
            name_matrix=name_matrix,
            addr_matrix=addr_matrix,
            name_equal=name_equal,
            char_sim=char_sim,
            min_df=min_df,
            category=categories,
            reason=reasons,
            n_unknown_df=np.asarray(n_unknown_df, dtype=np.int64),
        )
    timer.stop()
    log.info(
        "pair statistics done in %.1f min (%s pairs, %s unknown token frequencies)",
        (time.time() - pair_started) / 60.0,
        fmt_int(processed),
        fmt_int(n_unknown_df),
    )
    # The per-pair name and key strings stay alive through Phase 0.2 (the residue
    # examples name the pairs) and the name-equality mask through Phase 0.3.
    del keys_a, keys_b, addrs_a, addrs_b
    log_memory(log, "after pair statistics")

    # -- 8. Phase 0.2 ------------------------------------------------------
    timer.start("signal_coverage")
    coverage_report = _signal_coverage(
        name_equal=name_equal,
        min_df=min_df,
        char_sim=char_sim,
        addr_counts=addr_matrix,
        categories=categories,
        reasons=reasons,
        categories_available=classify_names,
        source_of_pair=source_of_pair,
        owners=owners,
        entity_ids=ground_truth.entity_ids,
        target_codes=target_codes,
        names_a=names_a,
        names_b=names_b,
        n_entities=n_source1,
        sources=sources,
        log=log,
    )
    # min_df survives: Phase 0.4's rarity table is defined on it.
    del char_sim, categories, reasons, names_a, names_b, owners
    timer.stop()
    log_memory(log, "after signal coverage")

    # -- 9. Phase 0.3 ------------------------------------------------------
    overlap_report = _address_overlap(addr_matrix, source_of_pair, name_equal, sources)
    del addr_matrix
    log.info("0.3 address: distributions computed")
    log_memory(log, "after address overlap")

    # -- 10. Phase 0.4 rarity ----------------------------------------------
    rarity_report = _pair_rarity(name_matrix, name_equal, min_df, source_of_pair, sources, log)
    del name_matrix, name_equal, min_df
    log_memory(log, "after pair rarity")

    # -- 11. Phase 0.4 census (all source1 entities, not just the matched) --
    timer.start("exact_name_census")
    exact_index_report, exact_candidates = _exact_name_census(
        config=config,
        split=args.split,
        sources=sources,
        s1_names=all_names,
        log=log,
    )
    timer.stop()
    log_memory(log, "after exact-name census")
    timer.start("candidate_census")
    census_report = _candidate_census(
        names=all_names,
        entity_ids=ground_truth.entity_ids,
        df_hashes=df_hashes,
        df_values=df_values,
        pairs_per_entity=pairs_per_entity,
        exact_candidates=exact_candidates,
        log=log,
    )
    del all_names, exact_candidates, df_hashes, df_values, pairs_per_entity
    timer.stop()
    log_memory(log, "after candidate census")

    # -- 12. Phase 0.5 -----------------------------------------------------
    zero_report = probe.summary()
    if zero_report.get("n_zero_match_entities"):
        zero_report["pct_of_all_source1"] = _pct(
            zero_report["n_zero_match_entities"], n_source1
        )
    log.info(
        "0.5 zero-match: %s entities, %s with exact-name candidates",
        fmt_int(zero_report.get("n_zero_match_entities", 0)),
        fmt_int(zero_report.get("exact_name_norm", {}).get("n_entities", 0)),
    )

    # -- 13. report + outputs ---------------------------------------------
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": args.split,
            "sources": sources,
            "workers": workers,
            # ``chunk_pairs`` is what actually ran; it is reduced below the request
            # only when the RAM budget cannot hold the queue (see the log line).
            "chunk_pairs": chunk_pairs,
            "chunk_pairs_requested": args.chunk_pairs,
            "in_flight_window": window_size,
            "limit_pairs": args.limit_pairs,
            "top_tokens": args.top_tokens,
            "name_categories": classify_names,
            # Asking to resume and actually resuming are different facts, and a
            # report that conflates them cannot answer "did I pay for this phase
            # twice?" - so both are recorded.
            "resume_requested": bool(args.resume),
            "resumed_phases": sorted(ckpt.restored),
            "n_source1_entities": int(n_source1),
            "n_true_pairs": int(n_pairs),
            "n_target_entities": int(sum(rows_per_source.values())),
            "target_rows_per_source": rows_per_source,
            "n_unknown_df": int(n_unknown_df),
            "candidate_census_cap": REFERENCE_TOKEN_CAP,
            # The field the shipped exact blocker keys on, and the field the
            # signals are defined against.
            "exact_blocker_key_field": NAME_NORM,
            "signal_key_field": NAME_KEY,
            "elapsed_minutes": round((time.time() - started) / 60.0, 3),
            "peak_rss": human_bytes(peak_rss_bytes() or current_rss_bytes() or 0),
            "prepared_dir": str(config["resolved"]["prepared_dir"]),
            # Machine-dependent, so recorded only when asked for: --timings is what
            # turns "the run is slow" into "this phase is slow".
            "hardware": hardware if args.timings else None,
            "phase_seconds": timer.summary() if args.timings else None,
        },
        "signal_coverage": coverage_report,
        "address_overlap": overlap_report,
        "token_frequency": per_scope,
        "pair_rarity": rarity_report,
        "candidate_census": census_report,
        "exact_name_census": exact_index_report,
        "zero_match": zero_report,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "blocking_statistics_report.json"
    md_path = output_dir / "blocking_statistics_report.md"
    write_json(json_path, report)
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    _write_coverage_csv(output_dir / "signal_coverage.csv", report)
    _write_census_csv(output_dir / "candidate_census.csv", report)
    _write_address_csv(output_dir / "address_overlap.csv", report)
    _write_token_csv(output_dir / "token_frequency.csv", report)
    _write_zero_match_csv(output_dir / "zero_match_statistics.csv", report)

    log.info("wrote %s", json_path)
    log.info("wrote %s", md_path)
    log.info("wrote 5 CSVs under %s", output_dir)
    log.info("total elapsed %.2f min", (time.time() - started) / 60.0)
    return 0
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
