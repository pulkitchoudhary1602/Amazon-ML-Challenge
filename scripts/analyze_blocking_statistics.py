#!/usr/bin/env python
"""Phase 0.2-0.5: measurements that decide what the blocker should be.

This is a MEASUREMENT script. It builds no blocker, trains no model, writes no
candidate file and changes nothing in the pipeline - it reads the prepared
tables and the ground truth and writes statistics. Four questions are answered,
one per phase:

* **0.2 country agreement** - among TRUE pairs, do the two sides agree on
  ``country_norm``? How often is a country missing? (Measurement only: nothing
  here says country may be used as a hard blocking rule.)
* **0.3 address overlap** - among TRUE pairs, how much do the normalized
  addresses overlap (shared tokens, Jaccard, overlap coefficient)? Is there
  enough signal to score or block on?
* **0.4 token frequency** - how large are the posting lists a token blocker
  would touch, how many tokens exceed a candidate cap, and - the question that
  actually matters - do the true pairs whose names DIFFER still share at least
  one token rare enough to be worth blocking on?
* **0.5 zero-match entities** - the S1 entities with no true match at all. How
  many of them would an exact-name blocker propose candidates for anyway? Those
  are pure false positives, so this measures how dangerous the population is.

Design
------
``iter_prepared`` streams the normalized tables, ``GroundTruth`` supplies the
true pairs, ``np.searchsorted`` resolves ids to attributes, and the per-pair
string work is sharded across worker processes (pure CPU work - there is no GPU
path worth using for token comparison). There is no cross-product anywhere:
Phases 0.2-0.4 walk the O(n_true_pairs) pair list, and Phase 0.5 walks the
O(n_target) corpus once per source.

The whole run is THREE streaming passes - source1, then source2, then source3 -
because one pass serves every phase: the token frequency counts, the zero-match
probe and the resolution of the matched targets' attributes all consume the same
chunk before it is discarded.

Memory
------
Peak RSS is dominated by two things (roughly 3-6GB on the full training set):
the token -> document-frequency counters (one per requested source, plus the
transient union used for the combined scope), and the resolved name/address
strings for the S1 entities and the matched targets. The counters hold COUNTS
only - never token -> entities, which would be ~100x larger. Nothing scales with
n_S1 x n_target. ``--sources source2`` roughly halves the peak.

Outputs (``<work_dir>/analysis`` unless ``--output-dir`` is given):

* ``blocking_statistics_report.json``  - all counts, machine-readable
* ``blocking_statistics_report.md``    - human-readable summary
* ``country_agreement.csv``            - S1 country | target country | pairs | %
* ``address_overlap.csv``              - tidy: scope, group, metric, statistic, value, unit
* ``token_frequency.csv``              - tidy: section=top_token|posting_cap, key, count, share
* ``zero_match_statistics.csv``        - tidy: section, key, count, share

No per-pair output is written - 7.6M rows would be a large artifact for no gain.

Usage::

    # HPC, full run
    python scripts/analyze_blocking_statistics.py --workers 16 --chunk-pairs 200000

    # local smoke test on a tiny synthetic fixture
    python scripts/analyze_blocking_statistics.py --config /tmp/smoke.yaml --split train

NOTE: the full run reads the whole training ground truth and the S2/S3 name,
address and country columns and is an HPC job. Do not run it on a laptop.
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import logging
import math
import multiprocessing as mp
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

from src.data_loader import (  # noqa: E402
    describe_environment,
    iter_prepared,
    load_config,
    load_ground_truth,
    prepared_path,
)
from src.normalization import ADDRESS_NORM, COUNTRY_NORM, NAME_KEY, NAME_NORM  # noqa: E402
from src.utils import (  # noqa: E402
    encode_entity_ids,
    fmt_int,
    log_memory,
    setup_logging,
    set_seed,
    stable_hash64,
    write_json,
)

LOG_NAME = "analyze_blocking_statistics"

SOURCE_CODES = {"source2": 2, "source3": 3}

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

# Above this many distinct values the country x country pair table is skipped:
# the table is O(V^2) and a runaway V (unexpected column contents) would allocate
# absurdly. Real country vocabularies are in the low hundreds.
MAX_COUNTRY_VOCABULARY = 2_000


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


# ---------------------------------------------------------------------------
# Country vocabulary
# ---------------------------------------------------------------------------
class _CountryVocabulary:
    """Shared id -> code interning table for ``country_norm``.

    Both sides of every pair must be coded in the SAME space for the agreement
    table to mean anything, so one vocabulary is threaded through the source1
    pass and both target passes. Code 0 is reserved for "no country". Nothing
    here knows or cares which countries exist, which keeps the analysis generic.
    """

    def __init__(self) -> None:
        self.names: list[str] = [""]  # index 0 = missing
        self.codes: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.names)

    def intern(self, values: np.ndarray) -> np.ndarray:
        """Code a chunk of country strings, extending the vocabulary as needed.

        Only the chunk's DISTINCT values (a few hundred) cross into Python; the
        per-row mapping stays a vectorized gather.
        """
        if not len(values):
            return np.zeros(0, dtype=np.int32)
        cleaned = pd.Series(values, dtype=object).fillna("").to_numpy(dtype=object)
        codes, uniques = pd.factorize(cleaned, sort=False)
        lookup = np.zeros(len(uniques), dtype=np.int32)
        for index, value in enumerate(uniques):
            text = str(value)
            if not text:
                lookup[index] = 0
                continue
            code = self.codes.get(text)
            if code is None:
                code = len(self.names)
                self.names.append(text)
                self.codes[text] = code
            lookup[index] = code
        codes = np.where(codes < 0, 0, codes)
        return lookup[codes]


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
    vocabulary: _CountryVocabulary,
    log: logging.Logger,
) -> dict[str, Any]:
    """Resolve columns for every ground-truth source1 entity, by GT row position.

    Indexing by ground-truth position makes the per-pair arrays a single gather
    (``values[owners]``) instead of a dictionary lookup per pair.

    Returns:
        ``{column: object array}`` for the string columns, plus ``"country"`` as
        int32 codes into ``vocabulary``.
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
    countries = np.zeros(ground_truth.n_entities, dtype=np.int32)
    found = 0
    for chunk in iter_prepared(config, "train", "source1", columns=_unique([entity_column, *wanted])):
        positions = ground_truth.positions_of(chunk[entity_column])
        keep = positions >= 0
        if not keep.any():
            continue
        positions = positions[keep]
        for column in wanted:
            out[column][positions] = _string_column(chunk, column)[keep]
        if COUNTRY_NORM in wanted:
            countries[positions] = vocabulary.intern(_string_column(chunk, COUNTRY_NORM)[keep])
        found += int(keep.sum())
    log.info(
        "source1: resolved %s of %s ground-truth entities",
        fmt_int(found),
        fmt_int(ground_truth.n_entities),
    )
    out["country"] = countries
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
    vocabulary: _CountryVocabulary,
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
       for Phases 0.2 and 0.3.

    Returns:
        ``(resolved, rows_seen)`` - the resolved attribute arrays (sized by the
        number of needed codes, empty when this source contributes no pair) and
        the number of rows scanned, which is the entity count Phase 0.4 reports
        frequencies against.
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
    has_country = COUNTRY_NORM in header
    if not has_key and probe.n_zero:
        log.warning(
            "%s has no %s column; the zero-match name_key counts are unavailable",
            source,
            NAME_KEY,
        )

    columns = [entity_column, NAME_NORM, *wanted]
    if has_key:
        columns.append(NAME_KEY)
    if has_country:
        columns.append(COUNTRY_NORM)
    columns = _unique(columns)

    use_resolution = len(needed_codes) > 0
    if use_resolution:
        for column in wanted:
            resolved[column] = np.full(len(needed_codes), "", dtype=object)
        resolved["country"] = np.zeros(len(needed_codes), dtype=np.int32)

    rows_seen = 0
    resolved_rows = 0
    report_every = max(1, config.get("io", {}).get("chunksize", 500_000))
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
                if has_country:
                    resolved["country"][slots] = vocabulary.intern(
                        _string_column(chunk, COUNTRY_NORM)[hit]
                    )
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
    payload: tuple[list[str], list[str], list[str], list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-pair name and address token statistics for one chunk of true pairs.

    Runs in a worker process. Returns:

    * ``name_counts``  ``int32[n, 3]`` - distinct S1 tokens, distinct target
      tokens, shared tokens.
    * ``addr_counts``  ``int32[n, 3]`` - the same for the normalized addresses.
    * ``name_equal``   ``uint8[n]`` - 1 where ``name_norm`` is byte-identical,
      i.e. exactly the pairs the exact-name blocker can see.
    * ``shared_hashes`` ``uint64[sum(shared)]`` - hashes of the shared NAME
      tokens, packed per pair. Hashes rather than strings so the parent can look
      every shared token up in one vectorized ``searchsorted``; the parent owns
      the frequency table and the workers never need a copy of it.
    """
    names_a, names_b, addrs_a, addrs_b = payload
    n = len(names_a)
    name_counts = np.zeros((n, 3), dtype=np.int32)
    addr_counts = np.zeros((n, 3), dtype=np.int32)
    name_equal = np.zeros(n, dtype=np.uint8)
    shared_parts: list[np.ndarray] = []

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

        address_a = addrs_a[row]
        address_b = addrs_b[row]
        addr_tokens_a = _token_set(address_a)
        addr_tokens_b = _token_set(address_b)
        addr_counts[row, 0] = len(addr_tokens_a)
        addr_counts[row, 1] = len(addr_tokens_b)
        addr_counts[row, 2] = len(addr_tokens_a & addr_tokens_b)

    shared_hashes = np.concatenate(shared_parts) if shared_parts else np.empty(0, dtype=np.uint64)
    return name_counts, addr_counts, name_equal, shared_hashes


def _iter_pair_chunks(
    names_a: np.ndarray,
    names_b: np.ndarray,
    addrs_a: np.ndarray,
    addrs_b: np.ndarray,
    n_pairs: int,
    chunk_pairs: int,
) -> Iterator[tuple[np.ndarray, tuple[list[str], list[str], list[str], list[str]]]]:
    """Yield ``(indices, payload)`` for each chunk of pairs.

    A fresh ``arange`` per chunk rather than one materialized index array: the
    caller must not hold 7.6M int64 just to slice what it already has.
    """
    for start in range(0, n_pairs, chunk_pairs):
        block = np.arange(start, min(start + chunk_pairs, n_pairs), dtype=np.int64)
        yield block, (
            names_a[block].tolist(),
            names_b[block].tolist(),
            addrs_a[block].tolist(),
            addrs_b[block].tolist(),
        )


def _resolve_workers(requested: int, configured: int, n_chunks: int, log: logging.Logger) -> int:
    """Decide the worker count: CLI > config > auto, clamped to the chunk count."""
    if requested > 0:
        workers = requested
    elif configured > 0:
        workers = configured
    else:
        workers = min(mp.cpu_count(), 8)
    workers = max(1, min(workers, max(n_chunks, 1)))
    log.info("pair workers: %d (cpu_count=%d)", workers, mp.cpu_count())
    return workers


# ---------------------------------------------------------------------------
# Phase 0.2: country agreement
# ---------------------------------------------------------------------------
def _country_agreement(
    country_a: np.ndarray,
    country_b: np.ndarray,
    source_of_pair: np.ndarray,
    owners: np.ndarray,
    n_entities: int,
    vocabulary: _CountryVocabulary,
    sources: Sequence[str],
    log: logging.Logger,
) -> dict[str, Any]:
    """Agreement between the two sides of every true pair, plus the entity view."""
    has_a = country_a != 0
    has_b = country_b != 0
    both = has_a & has_b
    agree = both & (country_a == country_b)
    disagree = both & (country_a != country_b)
    missing_either = ~both

    def _slice(mask: Optional[np.ndarray]) -> dict[str, Any]:
        n_pairs = int(len(country_a)) if mask is None else int(mask.sum())
        n_agree = int((agree if mask is None else (agree & mask)).sum())
        n_disagree = int((disagree if mask is None else (disagree & mask)).sum())
        n_missing_either = int((missing_either if mask is None else (missing_either & mask)).sum())
        n_missing_both = int((_combine(~has_a & ~has_b, mask)).sum())
        n_missing_a = int((_combine(~has_a & has_b, mask)).sum())
        n_missing_b = int((_combine(has_a & ~has_b, mask)).sum())
        return {
            "n_true_pairs": n_pairs,
            "n_agree": n_agree,
            "n_disagree": n_disagree,
            "pct_agree_of_all": _pct(n_agree, n_pairs),
            "pct_agree_of_comparable": _pct(n_agree, n_agree + n_disagree),
            "pct_disagree_of_all": _pct(n_disagree, n_pairs),
            "pct_disagree_of_comparable": _pct(n_disagree, n_agree + n_disagree),
            "n_missing_on_either_side": n_missing_either,
            "pct_missing_on_either_side": _pct(n_missing_either, n_pairs),
            "n_missing_both_sides": n_missing_both,
            "n_missing_source1_only": n_missing_a,
            "n_missing_target_only": n_missing_b,
        }

    per_source = {
        source: _slice(source_of_pair == SOURCE_CODES[source]) for source in sources
    }

    # -- entity level ------------------------------------------------------
    # Counted over the pairs ACTUALLY ANALYSED, not over ground_truth.lengths():
    # with --sources source2 the entity has pairs this run never looked at, and
    # comparing against the full length would call every such entity a
    # disagreement.
    pairs_per_entity = np.bincount(owners, minlength=n_entities)
    entity_agree = np.zeros(n_entities, dtype=np.int64)
    entity_disagree = np.zeros(n_entities, dtype=np.int64)
    entity_both = np.zeros(n_entities, dtype=np.int64)
    np.add.at(entity_agree, owners, agree.astype(np.int64))
    np.add.at(entity_disagree, owners, disagree.astype(np.int64))
    np.add.at(entity_both, owners, both.astype(np.int64))
    with_matches = pairs_per_entity > 0
    n_with_matches = int(with_matches.sum())
    all_agree = with_matches & (entity_agree == pairs_per_entity)
    any_disagree = with_matches & (entity_disagree > 0)
    no_usable = with_matches & (entity_both == 0)
    partial = with_matches & ~all_agree & ~any_disagree & (entity_both > 0)

    # -- pair frequency table ---------------------------------------------
    n_vocab = len(vocabulary)
    table_rows: list[dict[str, Any]] = []
    table_skipped = ""
    if n_vocab <= MAX_COUNTRY_VOCABULARY:
        for scope in (*sources, "all"):
            mask = None if scope == "all" else (source_of_pair == SOURCE_CODES[scope])
            a_codes = country_a if mask is None else country_a[mask]
            b_codes = country_b if mask is None else country_b[mask]
            flat = a_codes.astype(np.int64) * n_vocab + b_codes.astype(np.int64)
            counts = np.bincount(flat, minlength=n_vocab * n_vocab)
            total = int(counts.sum())
            for index in np.flatnonzero(counts).tolist():
                table_rows.append(
                    {
                        "target_source": scope,
                        "source1_country": vocabulary.names[index // n_vocab],
                        "target_country": vocabulary.names[index % n_vocab],
                        "n_pairs": int(counts[index]),
                        "pct_of_pairs": _pct(int(counts[index]), total),
                    }
                )
        table_rows.sort(key=lambda row: (row["target_source"], -row["n_pairs"]))
    else:
        table_skipped = (
            f"country vocabulary has {fmt_int(n_vocab)} distinct values, above the "
            f"{fmt_int(MAX_COUNTRY_VOCABULARY)} cap - the O(V^2) pair table is omitted"
        )
        log.warning(table_skipped)

    return {
        "all": _slice(None),
        "per_source": per_source,
        "entity_level": {
            "n_source1_entities_with_analysed_pairs": n_with_matches,
            "all_pairs_agree": {
                "n_entities": int(all_agree.sum()),
                "pct": _pct(int(all_agree.sum()), n_with_matches),
            },
            "any_pair_disagrees": {
                "n_entities": int(any_disagree.sum()),
                "pct": _pct(int(any_disagree.sum()), n_with_matches),
            },
            "no_usable_country_information": {
                "n_entities": int(no_usable.sum()),
                "pct": _pct(int(no_usable.sum()), n_with_matches),
                "definition": (
                    "every analysed true pair of the entity has a missing country on at "
                    "least one side, so no pair could be judged"
                ),
            },
            "partially_comparable_never_disagreeing": {
                "n_entities": int(partial.sum()),
                "pct": _pct(int(partial.sum()), n_with_matches),
                "definition": (
                    "some analysed pairs comparable and agreeing, the rest missing a country"
                ),
            },
            "n_country_values": max(n_vocab - 1, 0),
        },
        "pair_table": table_rows,
        "pair_table_skipped": table_skipped,
        "note": (
            "Measurement only. A high agreement rate does NOT establish that country may "
            "be used as a hard blocking rule: the pairs measured here are exactly the ones "
            "that ARE matches, so agreement is the posterior, not the selectivity. "
            "Candidate reduction has to be measured on non-matches."
        ),
    }


# ---------------------------------------------------------------------------
# Phase 0.3: address overlap
# ---------------------------------------------------------------------------
def _address_overlap(
    addr_counts: np.ndarray,
    source_of_pair: np.ndarray,
    country_relation: np.ndarray,
    sources: Sequence[str],
) -> dict[str, Any]:
    """Per-pair address token overlap, sliced by source and by country agreement.

    ``country_relation`` is the cheap int8 summary of Phase 0.2 - 0 = not
    comparable (a country is missing on one side), 1 = the two countries agree,
    2 = they disagree - so the country cross-tab does not need the two int32
    country arrays kept alive through the pair pass.
    """
    tokens_a = addr_counts[:, 0].astype(np.int64)
    tokens_b = addr_counts[:, 1].astype(np.int64)
    shared = addr_counts[:, 2].astype(np.int64)
    n = len(shared)
    both_present = (tokens_a > 0) & (tokens_b > 0)
    union = tokens_a + tokens_b - shared
    jaccard = np.divide(shared, union, out=np.zeros(n, dtype=np.float64), where=union > 0)
    minimum = np.minimum(tokens_a, tokens_b)
    overlap = np.divide(shared, minimum, out=np.zeros(n, dtype=np.float64), where=minimum > 0)

    groups: dict[str, Optional[np.ndarray]] = {
        "both_addresses_present": both_present,
        "country_agree": country_relation == 1,
        "country_disagree": country_relation == 2,
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
# Markdown report
# ---------------------------------------------------------------------------
def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _render_markdown(report: dict) -> str:
    """Human-readable summary: the four answers first, then the detail."""
    meta = report["meta"]
    country = report["country_agreement"]
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
        f"- **0.2 country** - {country['all']['pct_agree_of_comparable']:.2f}% of comparable "
        f"true pairs agree on country; {_plural(country['all']['n_disagree'], 'pair')} disagree."
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
    lines.append("## Phase 0.2 - country agreement")
    lines.append("")
    lines.append(
        f"All true pairs analysed: {fmt_int(country['all']['n_true_pairs'])}. Comparable (both "
        f"sides carry a country): {fmt_int(country['all']['n_agree'] + country['all']['n_disagree'])}."
    )
    lines.append("")
    lines.extend(
        _table(
            ["slice", "true pairs", "agree", "agree %", "disagree", "disagree %", "missing (either side)", "missing %"],
            [
                [
                    label,
                    fmt_int(entry["n_true_pairs"]),
                    fmt_int(entry["n_agree"]),
                    f"{entry['pct_agree_of_comparable']:.2f}",
                    fmt_int(entry["n_disagree"]),
                    f"{entry['pct_disagree_of_comparable']:.2f}",
                    fmt_int(entry["n_missing_on_either_side"]),
                    f"{entry['pct_missing_on_either_side']:.2f}",
                ]
                for label, entry in (("all", country["all"]), *country["per_source"].items())
            ],
        )
    )
    lines.append("")
    lines.append(
        "Missing-country detail (all pairs): both sides missing "
        f"{fmt_int(country['all']['n_missing_both_sides'])}, source1 only "
        f"{fmt_int(country['all']['n_missing_source1_only'])}, target only "
        f"{fmt_int(country['all']['n_missing_target_only'])}."
    )
    lines.append("")
    entity = country["entity_level"]
    lines.append(
        "Per source1 entity (over the pairs analysed in this run): "
        f"{fmt_int(entity['n_source1_entities_with_analysed_pairs'])} entities have at least one."
    )
    lines.append("")
    lines.extend(
        _table(
            ["entity outcome", "entities", "%"],
            [
                ["all pairs agree", fmt_int(entity["all_pairs_agree"]["n_entities"]), f"{entity['all_pairs_agree']['pct']:.2f}"],
                ["at least one disagreement", fmt_int(entity["any_pair_disagrees"]["n_entities"]), f"{entity['any_pair_disagrees']['pct']:.2f}"],
                ["no usable country info", fmt_int(entity["no_usable_country_information"]["n_entities"]), f"{entity['no_usable_country_information']['pct']:.2f}"],
                ["partially comparable, never disagreeing", fmt_int(entity["partially_comparable_never_disagreeing"]["n_entities"]), f"{entity['partially_comparable_never_disagreeing']['pct']:.2f}"],
            ],
        )
    )
    lines.append("")
    lines.append("Most frequent country combinations (see `country_agreement.csv` for all rows):")
    lines.append("")
    top_pairs = sorted(
        (row for row in country["pair_table"] if row["target_source"] == "all"),
        key=lambda row: -row["n_pairs"],
    )[:15]
    lines.extend(
        _table(
            ["source1 country", "target country", "true pairs", "% of pairs"],
            [
                [row["source1_country"] or "(missing)", row["target_country"] or "(missing)", fmt_int(row["n_pairs"]), f"{row['pct_of_pairs']:.4f}"]
                for row in top_pairs
            ],
        )
    )
    if country["pair_table_skipped"]:
        lines.append("")
        lines.append(f"NOTE: {country['pair_table_skipped']}")
    lines.append("")
    lines.append(f"> {country['note']}")
    lines.append("")

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
    lines.append("By country agreement (all pairs):")
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
def _write_country_csv(path: Path, report: dict) -> None:
    """The requested frequency table: S1 country | target country | pairs | %."""
    frame = pd.DataFrame(
        report["country_agreement"]["pair_table"],
        columns=["target_source", "source1_country", "target_country", "n_pairs", "pct_of_pairs"],
    )
    frame.to_csv(path, index=False, encoding="utf-8")


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
            "Phase 0.2-0.5: country agreement, address overlap, token frequency and "
            "zero-match statistics. Measurement only - builds no blocker."
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
    parser.add_argument("--workers", type=int, default=0, help="0 = auto (config, else min(cpu_count, 8))")
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
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


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
    log.info("=" * 78)
    started = time.time()

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

    # -- 2. source1 attributes, and the zero-match population ---------------
    vocabulary = _CountryVocabulary()
    s1 = _resolve_s1_columns(
        config, ground_truth, (NAME_NORM, NAME_KEY, ADDRESS_NORM, COUNTRY_NORM), vocabulary, log
    )
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

    names_a = s1.get(NAME_NORM, np.full(n_source1, "", dtype=object))[owners]
    addrs_a = s1.get(ADDRESS_NORM, np.full(n_source1, "", dtype=object))[owners]
    country_a = s1["country"][owners]
    del s1
    log_memory(log, "after per-pair source1 arrays")

    # -- 3. one streaming pass per target source ----------------------------
    token_counters: dict[str, Counter] = {}
    rows_per_source: dict[str, int] = {}
    needed_by_source: dict[int, np.ndarray] = {}
    resolved_by_source: dict[int, dict[str, Any]] = {}
    for source in sources:
        code = SOURCE_CODES[source]
        needed_by_source[code] = np.unique(target_codes[source_of_pair == code])
        token_counters[source] = Counter()
        resolved, rows_seen = _scan_target_source(
            config=config,
            source=source,
            needed_codes=needed_by_source[code],
            vocabulary=vocabulary,
            token_counter=token_counters[source],
            probe=probe,
            log=log,
        )
        resolved_by_source[code] = resolved
        rows_per_source[source] = rows_seen
        log_memory(log, f"after {source} scan")

    # -- 4. per-pair target attributes -------------------------------------
    names_b = np.full(n_pairs, "", dtype=object)
    addrs_b = np.full(n_pairs, "", dtype=object)
    country_b = np.zeros(n_pairs, dtype=np.int32)
    for code, resolved in resolved_by_source.items():
        mask = source_of_pair == code
        if not mask.any():
            continue
        needed = needed_by_source[code]
        slots = np.searchsorted(needed, target_codes[mask])
        np.clip(slots, 0, max(len(needed) - 1, 0), out=slots)
        names_b[mask] = resolved.get(NAME_NORM, np.full(len(needed), "", dtype=object))[slots]
        addrs_b[mask] = resolved.get(ADDRESS_NORM, np.full(len(needed), "", dtype=object))[slots]
        country_b[mask] = resolved.get("country", np.zeros(len(needed), dtype=np.int32))[slots]
    del resolved_by_source
    log_memory(log, "after per-pair target arrays")

    # -- 5. Phase 0.2 ------------------------------------------------------
    country_report = _country_agreement(
        country_a, country_b, source_of_pair, owners, n_source1, vocabulary, sources, log
    )
    log.info(
        "0.2 country: %.2f%% of comparable true pairs agree",
        country_report["all"]["pct_agree_of_comparable"],
    )
    # Keep the country comparison as a 1-byte-per-pair code so Phase 0.3 can slice
    # by country agreement after the int32 country arrays are gone.
    comparable = (country_a != 0) & (country_b != 0)
    country_relation = np.zeros(n_pairs, dtype=np.int8)
    country_relation[comparable] = np.where(
        country_a[comparable] == country_b[comparable], np.int8(1), np.int8(2)
    )
    del country_a, country_b, owners
    log_memory(log, "after country agreement")

    # -- 6. Phase 0.4: token frequency table (the lookup the pair pass needs)
    per_scope, df_hashes, df_values = _token_frequency(
        token_counters, rows_per_source, args.top_tokens, log
    )
    del token_counters
    log_memory(log, "after token frequency table")

    # -- 7. Phases 0.3 / 0.4: sharded per-pair statistics ------------------
    # int16 for the token counts and int64 for the rarest-token document
    # frequency: ~160MB at 7.6M pairs, in place of a 7.6M-row Python object.
    name_matrix = np.zeros((n_pairs, 3), dtype=np.int16)
    addr_matrix = np.zeros((n_pairs, 3), dtype=np.int16)
    name_equal = np.zeros(n_pairs, dtype=bool)
    min_df = np.zeros(n_pairs, dtype=np.int64)

    n_chunks = (n_pairs + args.chunk_pairs - 1) // args.chunk_pairs
    workers = _resolve_workers(
        args.workers, config.get("compute", {}).get("num_workers", 0), n_chunks, log
    )
    progress_step = max(1, args.chunk_pairs * 10)
    processed = 0
    n_unknown_df = 0

    def consume(block: np.ndarray, result: tuple) -> None:
        """Fold one chunk of pair statistics into the per-pair arrays."""
        nonlocal processed, n_unknown_df
        chunk_names, chunk_addrs, chunk_equal, shared_hashes = result
        name_matrix[block] = chunk_names
        addr_matrix[block] = chunk_addrs
        name_equal[block] = chunk_equal.astype(bool)
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
        if processed % progress_step < args.chunk_pairs:
            log.info("  pair statistics: %s / %s", fmt_int(processed), fmt_int(n_pairs))

    chunks = _iter_pair_chunks(names_a, names_b, addrs_a, addrs_b, n_pairs, args.chunk_pairs)
    pair_started = time.time()
    if workers == 1:
        log.info("running single-process (workers=1)")
        for block, payload in chunks:
            consume(block, _pair_statistics(payload))
    else:
        # ProcessPoolExecutor.map() submits EVERY chunk before the first one is
        # consumed, which would queue the whole pair set in memory - the exact
        # thing chunking exists to avoid. A bounded sliding window keeps at most
        # 4 x workers chunks in flight and consumes strictly in submission order,
        # which also keeps the run reproducible.
        window: deque = deque()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for block, payload in chunks:
                window.append((block, pool.submit(_pair_statistics, payload)))
                if len(window) >= workers * 4:
                    done_block, future = window.popleft()
                    consume(done_block, future.result())
            while window:
                done_block, future = window.popleft()
                consume(done_block, future.result())
    log.info(
        "pair statistics done in %.1f min (%s pairs, %s unknown token frequencies)",
        (time.time() - pair_started) / 60.0,
        fmt_int(processed),
        fmt_int(n_unknown_df),
    )
    del names_a, names_b, addrs_a, addrs_b
    log_memory(log, "after pair statistics")

    # -- 8. Phase 0.3 ------------------------------------------------------
    overlap_report = _address_overlap(addr_matrix, source_of_pair, country_relation, sources)
    del addr_matrix, country_relation
    log.info("0.3 address: distributions computed")
    log_memory(log, "after address overlap")

    # -- 9. Phase 0.4 rarity -----------------------------------------------
    rarity_report = _pair_rarity(name_matrix, name_equal, min_df, source_of_pair, sources, log)
    del name_matrix, name_equal, min_df
    log_memory(log, "after pair rarity")

    # -- 10. Phase 0.5 -----------------------------------------------------
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

    # -- 11. report + outputs ---------------------------------------------
    report = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": args.split,
            "sources": sources,
            "workers": workers,
            "chunk_pairs": args.chunk_pairs,
            "limit_pairs": args.limit_pairs,
            "top_tokens": args.top_tokens,
            "n_source1_entities": int(n_source1),
            "n_true_pairs": int(n_pairs),
            "n_target_entities": int(sum(rows_per_source.values())),
            "target_rows_per_source": rows_per_source,
            "n_unknown_df": int(n_unknown_df),
            "country_vocabulary_size": max(len(vocabulary) - 1, 0),
            # The field the shipped exact blocker keys on. Every "pairs exact-name
            # blocking misses" figure below is defined against it.
            "exact_blocker_key_field": NAME_NORM,
            "elapsed_minutes": round((time.time() - started) / 60.0, 3),
            "prepared_dir": str(config["resolved"]["prepared_dir"]),
        },
        "country_agreement": country_report,
        "address_overlap": overlap_report,
        "token_frequency": per_scope,
        "pair_rarity": rarity_report,
        "zero_match": zero_report,
    }

    output_dir = Path(args.output_dir) if args.output_dir else Path(config["resolved"]["work_dir"]) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "blocking_statistics_report.json"
    md_path = output_dir / "blocking_statistics_report.md"
    write_json(json_path, report)
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    _write_country_csv(output_dir / "country_agreement.csv", report)
    _write_address_csv(output_dir / "address_overlap.csv", report)
    _write_token_csv(output_dir / "token_frequency.csv", report)
    _write_zero_match_csv(output_dir / "zero_match_statistics.csv", report)

    log.info("wrote %s", json_path)
    log.info("wrote %s", md_path)
    log.info("wrote 4 CSVs under %s", output_dir)
    log.info("total elapsed %.2f min", (time.time() - started) / 60.0)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
