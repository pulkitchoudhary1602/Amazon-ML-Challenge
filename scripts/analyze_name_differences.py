#!/usr/bin/env python
"""Phase 0.1: why do the names of TRUE matches differ?

The question this answers
------------------------
"What proportion of true matches would exact-name blocking miss, and what are
the dominant reasons?"

The exact-name blocker (``blocking.exact_name``, ``key: name_norm``) proposes a
pair only when the two normalized names are byte-identical. Every true pair whose
names differ is therefore invisible to it. This script walks the ground truth,
takes every true pair, and classifies *why* the two names differ - so the miss
rate is not just measured but explained, and the explanation says which
normalization or blocker change would recover the most pairs.

Scope
-----
Only pairs that are already in the ground truth are examined. There is no
cross-product anywhere: the script never compares an S1 record against a target
it is not truly matched to, so the workload is O(n_true_pairs), not
O(n_S1 x n_target).

Everything is built on the existing streaming stack - ``iter_prepared`` for the
normalized tables, ``GroundTruth`` for the pairs, ``np.searchsorted`` for
id -> name resolution - and the string comparison stage is sharded across worker
processes because it is pure CPU work with no GPU path worth using.

Outputs (written to ``<work_dir>/analysis`` unless ``--output-dir`` is given):

* ``name_difference_report.json``  - machine-readable counts and percentages
* ``name_difference_report.md``    - human-readable summary
* ``name_difference_examples.tsv`` - representative pair per category

Usage::

    # HPC, full run
    python scripts/analyze_name_differences.py --workers 16

    # local smoke test on a tiny synthetic fixture
    python scripts/analyze_name_differences.py --config /tmp/smoke.yaml --limit-pairs 50

NOTE: the full run reads the whole training ground truth and the S2/S3 name
columns and is an HPC job. Do not run it on a laptop.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import multiprocessing as mp
import sys
import time
import unicodedata
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
from src.normalization import COUNTRY_NORM, NAME_NORM  # noqa: E402
from src.utils import (  # noqa: E402
    decode_entity_id,
    encode_entity_id,
    fmt_int,
    log_memory,
    setup_logging,
    set_seed,
    write_json,
)

LOG_NAME = "analyze_name_differences"

SOURCE_CODES = {"source2": 2, "source3": 3}

# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------
# Mutually exclusive, applied in this precedence order. The order is the whole
# design: the cheap structural explanations are settled before any similarity
# number is computed, so a pair is only called a "typo" when it survived the
# checks that would have explained it more precisely.
CATEGORIES = [
    "exact_normalized_match",   # name_norm identical - exact blocking catches it
    "separator_only",           # differs only in spacing: name_key still matches
    "token_word_order",         # same tokens, different order
    "token_subset_superset",    # one name's tokens are a strict subset of the other's
    "typo_small_edit",          # <=15% normalized edit distance
    "transliteration_script",   # the two names are written in different scripts
    "substantially_different",  # no token overlap and >=60% edit distance, confidently unrelated
    "unknown_other",            # heuristic could not establish the cause
]
CATEGORY_INDEX = {name: i for i, name in enumerate(CATEGORIES)}

# Why a pair landed in unknown_other. Reported so the residual bucket is not a
# black box: it says what the heuristic was missing, not just that it failed.
UNKNOWN_REASONS = [
    "not_applicable",
    "missing_name",                    # one side normalized to an empty string
    "partial_token_overlap",           # shares some tokens but is neither subset nor reorder
    "shared_character_content",        # large edit distance yet many shared trigrams
    "no_shared_content",               # nothing shared, but similarity too low to be confident
    "moderate_similarity",             # in between: not a small edit, not clearly unrelated
]
REASON_INDEX = {name: i for i, name in enumerate(UNKNOWN_REASONS)}

# Thresholds. Deliberately conservative: the typo band is narrow and the
# "substantially different" band requires BOTH a large edit distance and almost
# no shared character trigrams. Anything in the middle stays unknown rather than
# being guessed at.
TYPO_MAX_NORM_EDIT = 0.15
DIFFERENT_MIN_NORM_EDIT = 0.60
DIFFERENT_MAX_TRIGRAM_JACCARD = 0.25

TOKEN_HISTOGRAM_MAX = 12  # token counts are bucketed at "12+"


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _dominant_script(text: str) -> str:
    """Most common Unicode script among the letters/marks of ``text``.

    Derived from ``unicodedata.name`` so no extra dependency is needed: the first
    word of a character name is its script ("DEVANAGARI SIGN VIRAMA" ->
    DEVANAGARI, "LATIN SMALL LETTER A" -> LATIN). Digits, punctuation and symbols
    are skipped, so "12 Main St" reports LATIN rather than DIGIT.

    Returns "" when the string carries no letters at all.
    """
    counts: Counter[str] = Counter()
    for char in text:
        if unicodedata.category(char)[0] not in ("L", "M"):
            continue
        name = unicodedata.name(char, "")
        if name:
            counts[name.split(" ", 1)[0]] += 1
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _bounded_levenshtein(a: str, b: str, cap: int) -> int:
    """Levenshtein distance, abandoned as soon as it provably exceeds ``cap``.

    Returns ``cap + 1`` to mean "more than cap". The band keeps the DP at
    O(len * cap) instead of O(len^2), which matters because this is the only
    super-linear step in the classifier and it runs over millions of pairs.
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        row_min = i
        for j, char_b in enumerate(b, 1):
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (0 if char_a == char_b else 1),
            )
            current.append(value)
            if value < row_min:
                row_min = value
        if row_min > cap:
            return cap + 1
        previous = current
    return previous[-1]


def _trigram_jaccard(a: str, b: str) -> float:
    """Jaccard overlap of character trigrams; character sets for short strings."""
    if len(a) < 3 or len(b) < 3:
        set_a, set_b = set(a), set(b)
    else:
        set_a = {a[i : i + 3] for i in range(len(a) - 2)}
        set_b = {b[i : i + 3] for i in range(len(b) - 2)}
    if not set_a or not set_b:
        return 0.0
    shared = len(set_a & set_b)
    union = len(set_a) + len(set_b) - shared
    return shared / union if union else 0.0


def _token_jaccard(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    shared = len(tokens_a & tokens_b)
    union = len(tokens_a) + len(tokens_b) - shared
    return shared / union if union else 0.0


# ---------------------------------------------------------------------------
# The classifier
# ---------------------------------------------------------------------------
def classify_pair(name_a: str, name_b: str) -> tuple[int, int, int, int]:
    """Classify why two normalized names differ.

    Args:
        name_a: ``name_norm`` of the source1 entity.
        name_b: ``name_norm`` of the matched target entity.

    Returns:
        ``(category, reason, n_tokens_a, n_tokens_b)`` as integers indexing
        :data:`CATEGORIES` and :data:`UNKNOWN_REASONS`.
    """
    tokens_a_list = name_a.split()
    tokens_b_list = name_b.split()
    n_tokens_a = len(tokens_a_list)
    n_tokens_b = len(tokens_b_list)

    def result(category: str, reason: str = "not_applicable") -> tuple[int, int, int, int]:
        return CATEGORY_INDEX[category], REASON_INDEX[reason], n_tokens_a, n_tokens_b

    # 1. Exact normalized-name match.
    if name_a and name_a == name_b:
        return result("exact_normalized_match")

    # 2. Missing name on one side: nothing can be established from the text.
    if not name_a or not name_b:
        return result("unknown_other", "missing_name")

    # 3. Differs only by separators. name_key would still match these, so they
    #    are "missed by exact_name on name_norm" but recoverable for free by
    #    keying the same index on name_key.
    if name_a.replace(" ", "") == name_b.replace(" ", ""):
        return result("separator_only")

    # 4. Same tokens in a different order.
    if Counter(tokens_a_list) == Counter(tokens_b_list):
        return result("token_word_order")

    # 5. Strict token subset/superset (an extra word, a dropped suffix, ...).
    set_a, set_b = set(tokens_a_list), set(tokens_b_list)
    if set_a < set_b or set_b < set_a:
        return result("token_subset_superset")

    # 6. Different script: one side romanized, the other in a native script.
    #    Checked before any distance metric, because a transliteration has a
    #    large edit distance and would otherwise be misread as "unrelated".
    script_a = _dominant_script(name_a)
    script_b = _dominant_script(name_b)
    if script_a and script_b and script_a != script_b:
        return result("transliteration_script")

    # 7+. Same script, no token structure explains it: fall back to edit distance.
    longest = max(len(name_a), len(name_b))
    cap = max(3, int(DIFFERENT_MIN_NORM_EDIT * longest) + 1)
    distance = _bounded_levenshtein(name_a, name_b, cap)

    if distance <= TYPO_MAX_NORM_EDIT * longest:
        return result("typo_small_edit")

    # A shared whole word blocks the "confidently unrelated" label even when the
    # strings are far apart: "acme solutions" vs "zenith solutions" is not the
    # same name, but the shared token is evidence we cannot dismiss, so it stays
    # unknown rather than being asserted as a different name.
    if _token_jaccard(set_a, set_b) > 0:
        return result("unknown_other", "partial_token_overlap")

    if distance > DIFFERENT_MIN_NORM_EDIT * longest:
        if _trigram_jaccard(name_a, name_b) <= DIFFERENT_MAX_TRIGRAM_JACCARD:
            return result("substantially_different")
        return result("unknown_other", "shared_character_content")

    # The middle band. Refuse to name a cause.
    if _trigram_jaccard(name_a, name_b) > DIFFERENT_MAX_TRIGRAM_JACCARD:
        return result("unknown_other", "shared_character_content")
    return result("unknown_other", "no_shared_content")


def _classify_chunk(payload: tuple[list[str], list[str]]) -> np.ndarray:
    """Worker entry point: classify one chunk of pairs.

    Returns an ``int32[n, 4]`` array of ``(category, reason, tokens_a, tokens_b)``.
    A dense array rather than a list of tuples keeps the inter-process payload
    small, which matters when millions of pairs cross the boundary.
    """
    names_a, names_b = payload
    out = np.empty((len(names_a), 4), dtype=np.int32)
    for i in range(len(names_a)):
        out[i, 0], out[i, 1], out[i, 2], out[i, 3] = classify_pair(names_a[i], names_b[i])
    return out


# ---------------------------------------------------------------------------
# Loading
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


def _string_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    """A column as an object array of plain ``str``, missing values as "".

    Going through pandas' ``string`` dtype rather than ``fillna("").astype(str)``
    matters: a column that is entirely empty is read as float64, and ``astype(str)``
    on that would silently turn every missing name into the string "0.0".
    """
    return frame[column].astype("string").fillna("").to_numpy(dtype=object)


def _country_code(value: str, vocabulary: dict[str, int]) -> int:
    """Intern a country string into a small int; 0 means "no country"."""
    if not value:
        return 0
    code = vocabulary.get(value)
    if code is None:
        code = len(vocabulary) + 1
        vocabulary[value] = code
    return code


def _load_target_names(
    config: dict,
    source: str,
    needed_codes: np.ndarray,
    country_vocabulary: dict[str, int],
    log: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the names of the target entities that appear in the ground truth.

    Streams the prepared table once and keeps only the rows whose entity id is in
    ``needed_codes``. The file is never fully loaded, and the output arrays are
    sized by the number of *matched* targets, not by the size of the source.

    Args:
        needed_codes: sorted int64 codes of the target entities to keep.
        country_vocabulary: shared id -> code interning table, updated in place.

    Returns:
        ``(names, country_codes)``; ``names`` is an object array parallel to
        ``needed_codes`` (empty string where the entity was not found).
    """
    entity_column = config.get("columns", {}).get("entity_id", "entity_id")
    if not len(needed_codes):
        return np.empty(0, dtype=object), np.zeros(0, dtype=np.int32)
    path = prepared_path(config, "train", source)
    header = _prepared_header(path)
    columns = [entity_column, NAME_NORM]
    if COUNTRY_NORM in header:
        columns.append(COUNTRY_NORM)
    else:
        log.warning("%s has no %s column; country breakdown will be omitted", path.name, COUNTRY_NORM)

    names = np.full(len(needed_codes), "", dtype=object)
    countries = np.zeros(len(needed_codes), dtype=np.int32)
    found = 0
    for chunk in iter_prepared(config, "train", source, columns=columns):
        codes = chunk[entity_column].map(encode_entity_id).to_numpy(dtype=np.int64)
        slots = np.searchsorted(needed_codes, codes)
        np.clip(slots, 0, max(len(needed_codes) - 1, 0), out=slots)
        hit = needed_codes[slots] == codes
        if not hit.any():
            continue
        slots = slots[hit]
        names[slots] = _string_column(chunk, NAME_NORM)[hit]
        if COUNTRY_NORM in columns:
            values = _string_column(chunk, COUNTRY_NORM)[hit]
            countries[slots] = np.fromiter(
                (_country_code(str(v), country_vocabulary) for v in values),
                dtype=np.int32,
                count=len(values),
            )
        found += int(hit.sum())
    log.info("%s: resolved %s of %s needed target names", source, fmt_int(found), fmt_int(len(needed_codes)))
    return names, countries


def _load_s1_names(
    config: dict,
    ground_truth,
    country_vocabulary: dict[str, int],
    log: logging.Logger,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve ``name_norm`` for every source1 entity in the ground truth.

    Indexed by ground-truth row position, so the pair arrays can gather from it
    directly.

    Returns:
        ``(names, country_codes)``, both length ``ground_truth.n_entities``.
    """
    entity_column = config.get("columns", {}).get("entity_id", "entity_id")
    path = prepared_path(config, "train", "source1")
    header = _prepared_header(path)
    columns = [entity_column, NAME_NORM]
    if COUNTRY_NORM in header:
        columns.append(COUNTRY_NORM)
    else:
        log.warning("%s has no %s column; country breakdown will be omitted", path.name, COUNTRY_NORM)

    names = np.full(ground_truth.n_entities, "", dtype=object)
    countries = np.zeros(ground_truth.n_entities, dtype=np.int32)
    found = 0
    for chunk in iter_prepared(config, "train", "source1", columns=columns):
        positions = ground_truth.positions_of(chunk[entity_column])
        keep = positions >= 0
        if not keep.any():
            continue
        positions = positions[keep]
        names[positions] = _string_column(chunk, NAME_NORM)[keep]
        if COUNTRY_NORM in columns:
            values = _string_column(chunk, COUNTRY_NORM)[keep]
            countries[positions] = np.fromiter(
                (_country_code(str(v), country_vocabulary) for v in values),
                dtype=np.int32,
                count=len(values),
            )
        found += int(keep.sum())
    log.info("source1: resolved %s of %s ground-truth entities", fmt_int(found), fmt_int(ground_truth.n_entities))
    return names, countries


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
class _Counts:
    """Per-slice category/reason counters plus the token and country tables."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.categories = np.zeros(len(CATEGORIES), dtype=np.int64)
        self.reasons = np.zeros(len(UNKNOWN_REASONS), dtype=np.int64)
        self.n_pairs = 0
        self.n_exact_key_equal = 0
        self.n_missing_name = 0
        self.n_country_mismatch = 0
        self.token_hist_a = np.zeros(TOKEN_HISTOGRAM_MAX + 1, dtype=np.int64)
        self.token_hist_b = np.zeros(TOKEN_HISTOGRAM_MAX + 1, dtype=np.int64)
        self.token_delta = Counter()

    def add_chunk(
        self,
        result: np.ndarray,
        key_equal: np.ndarray,
        country_a: np.ndarray,
        country_b: np.ndarray,
    ) -> None:
        categories = result[:, 0]
        reasons = result[:, 1]
        self.n_pairs += len(categories)
        self.categories += np.bincount(categories, minlength=len(CATEGORIES))
        unknown = categories == CATEGORY_INDEX["unknown_other"]
        if unknown.any():
            self.reasons += np.bincount(reasons[unknown], minlength=len(UNKNOWN_REASONS))
        self.n_exact_key_equal += int(key_equal.sum())
        self.n_missing_name += int((reasons == REASON_INDEX["missing_name"]).sum())
        if len(country_a):
            self.n_country_mismatch += int((country_a != country_b).sum())

        tokens_a = np.minimum(result[:, 2], TOKEN_HISTOGRAM_MAX)
        tokens_b = np.minimum(result[:, 3], TOKEN_HISTOGRAM_MAX)
        self.token_hist_a += np.bincount(tokens_a, minlength=TOKEN_HISTOGRAM_MAX + 1)
        self.token_hist_b += np.bincount(tokens_b, minlength=TOKEN_HISTOGRAM_MAX + 1)
        delta = np.clip(result[:, 3] - result[:, 2], -TOKEN_HISTOGRAM_MAX, TOKEN_HISTOGRAM_MAX)
        values, counts = np.unique(delta, return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            self.token_delta[int(value)] += int(count)

    # -- rendering ----------------------------------------------------------
    def category_table(self) -> dict[str, dict[str, Any]]:
        differing = self.n_pairs - int(self.categories[CATEGORY_INDEX["exact_normalized_match"]])
        table = {}
        for index, name in enumerate(CATEGORIES):
            count = int(self.categories[index])
            table[name] = {
                "count": count,
                "pct_of_pairs": _pct(count, self.n_pairs),
                "pct_of_differing": _pct(count, differing),
            }
        return table

    def summary(self) -> dict[str, Any]:
        differing = self.n_pairs - int(self.categories[CATEGORY_INDEX["exact_normalized_match"]])
        caught = int(self.categories[CATEGORY_INDEX["exact_normalized_match"]])
        return {
            "n_true_pairs": self.n_pairs,
            "n_name_norm_identical": caught,
            "n_name_norm_differs": differing,
            "n_name_key_identical": self.n_exact_key_equal,
            "pct_name_norm_differs": _pct(differing, self.n_pairs),
            "pct_name_key_differs": _pct(self.n_pairs - self.n_exact_key_equal, self.n_pairs),
            "n_missing_name": self.n_missing_name,
            "n_country_mismatch": self.n_country_mismatch,
            "pct_country_mismatch": _pct(self.n_country_mismatch, self.n_pairs),
            "categories": self.category_table(),
            "unknown_reasons": {
                name: int(self.reasons[index]) for index, name in enumerate(UNKNOWN_REASONS) if index
            },
            "token_counts": {
                "source1": _histogram(self.token_hist_a),
                "target": _histogram(self.token_hist_b),
                "delta_target_minus_source1": {str(k): v for k, v in sorted(self.token_delta.items())},
            },
        }


def _pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def _histogram(counts: np.ndarray) -> dict[str, int]:
    out = {str(i): int(counts[i]) for i in range(len(counts) - 1) if counts[i]}
    if counts[-1]:
        out[f"{TOKEN_HISTOGRAM_MAX}+"] = int(counts[-1])
    return out


# ---------------------------------------------------------------------------
# Pair iteration
# ---------------------------------------------------------------------------
def _iter_pair_chunks(
    names_a: np.ndarray,
    names_b: np.ndarray,
    n_pairs: int,
    chunk_pairs: int,
) -> Iterator[tuple[np.ndarray, tuple[list[str], list[str]]]]:
    """Yield ``(indices, payload)`` for each chunk of pairs.

    A fresh ``arange`` per chunk rather than a materialized index array: the
    caller must not hold 7.6M int64 just to slice what it already has.
    """
    for start in range(0, n_pairs, chunk_pairs):
        block = np.arange(start, min(start + chunk_pairs, n_pairs), dtype=np.int64)
        yield block, (names_a[block].tolist(), names_b[block].tolist())


def _resolve_workers(requested: int, configured: int, n_chunks: int, log: logging.Logger) -> int:
    """Decide the worker count: CLI > config > auto, clamped to the chunk count."""
    if requested > 0:
        workers = requested
    elif configured > 0:
        workers = configured
    else:
        workers = min(mp.cpu_count(), 8)
    workers = max(1, min(workers, max(n_chunks, 1)))
    log.info("classification workers: %d (cpu_count=%d)", workers, mp.cpu_count())
    return workers


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _render_markdown(report: dict, examples: dict[str, list[dict]]) -> str:
    """Human-readable summary. Leads with the answer to the Phase 0.1 question."""
    totals = report["totals"]
    lines: list[str] = []
    add = lines.append

    add("# Phase 0.1 - Why true-match names differ")
    add("")
    add(f"Generated: {report['meta']['generated_at']}  ")
    add(f"Sources: {', '.join(report['meta']['sources'])}  ")
    add(f"True pairs analyzed: {fmt_int(totals['n_true_pairs'])}")
    add("")
    add("## The Phase 0.1 question")
    add("")
    add(
        f"**Exact-name blocking (keyed on `name_norm`) misses "
        f"{totals['pct_name_norm_differs']:.2f}% of true pairs "
        f"({fmt_int(totals['n_name_norm_differs'])} of {fmt_int(totals['n_true_pairs'])}).**"
    )
    add("")
    recovered = totals["n_name_key_identical"] - totals["n_name_norm_identical"]
    add(
        f"Keying the same index on `name_key` instead would miss "
        f"{totals['pct_name_key_differs']:.2f}% "
        f"({fmt_int(totals['n_true_pairs'] - totals['n_name_key_identical'])} pairs): "
        f"re-keying recovers {fmt_int(recovered)} "
        f"{'pair' if recovered == 1 else 'pairs'} for no new blocker."
    )
    add("")
    add("Dominant reasons among the pairs that differ:")
    add("")
    add("| reason | pairs | % of differing | % of all |")
    add("|---|---:|---:|---:|")
    for entry in report["dominant_reasons"]:
        add(
            f"| {entry['category']} | {fmt_int(entry['count'])} | "
            f"{entry['pct_of_differing']:.2f}% | {entry['pct_of_pairs']:.2f}% |"
        )
    add("")
    add("## Categories")
    add("")
    add("| category | combined | S2 | S3 | % of all | % of differing |")
    add("|---|---:|---:|---:|---:|---:|")
    combined = report["categories"]["combined"]
    source2 = report["categories"].get("source2", {})
    source3 = report["categories"].get("source3", {})
    for name in CATEGORIES:
        row = combined.get(name, {})
        add(
            f"| {name} | {fmt_int(row.get('count', 0))} | "
            f"{fmt_int(source2.get(name, {}).get('count', 0))} | "
            f"{fmt_int(source3.get(name, {}).get('count', 0))} | "
            f"{row.get('pct_of_pairs', 0.0):.2f}% | {row.get('pct_of_differing', 0.0):.2f}% |"
        )
    add("")
    add("## Per source")
    add("")
    add("| slice | true pairs | name differs | % differs | exact | separator | order | subset | typo | script | different | unknown |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in ("combined", "source2", "source3"):
        summary = report["slices"].get(label)
        if not summary:
            continue
        cats = summary["categories"]
        add(
            f"| {label} | {fmt_int(summary['n_true_pairs'])} | {fmt_int(summary['n_name_norm_differs'])} | "
            f"{summary['pct_name_norm_differs']:.2f}% | "
            + " | ".join(fmt_int(cats[name]["count"]) for name in CATEGORIES)
            + " |"
        )
    add("")
    add("## Unknown / other")
    add("")
    add("Pairs the heuristic refused to name a cause for:")
    add("")
    add("| why | combined | S2 | S3 |")
    add("|---|---:|---:|---:|")
    for reason, count in report["slices"]["combined"]["unknown_reasons"].items():
        if not count:
            continue
        add(
            f"| {reason} | {fmt_int(count)} | "
            f"{fmt_int(report['slices'].get('source2', {}).get('unknown_reasons', {}).get(reason, 0))} | "
            f"{fmt_int(report['slices'].get('source3', {}).get('unknown_reasons', {}).get(reason, 0))} |"
        )
    add("")
    add("## Name token counts")
    add("")
    add("| tokens | source1 names | target names |")
    add("|---|---:|---:|")
    histogram_a = report["slices"]["combined"]["token_counts"]["source1"]
    histogram_b = report["slices"]["combined"]["token_counts"]["target"]
    for key in sorted(set(histogram_a) | set(histogram_b), key=lambda k: (len(k), k)):
        add(f"| {key} | {fmt_int(histogram_a.get(key, 0))} | {fmt_int(histogram_b.get(key, 0))} |")
    add("")
    add("Token-count difference (target minus source1):")
    add("")
    add("| delta | pairs |")
    add("|---|---:|")
    for key, count in report["slices"]["combined"]["token_counts"]["delta_target_minus_source1"].items():
        add(f"| {key} | {fmt_int(count)} |")
    add("")
    country = report.get("country")
    if country and country.get("top"):
        add("## Country (source1 side)")
        add("")
        add("| country | pairs | differs | % differs | most common reason |")
        add("|---|---:|---:|---:|---|")
        for entry in country["top"]:
            add(
                f"| {entry['country']} | {fmt_int(entry['n_pairs'])} | {fmt_int(entry['n_differs'])} | "
                f"{entry['pct_differs']:.2f}% | {entry['top_category']} |"
            )
        add("")
    add("## Representative examples")
    add("")
    for name in CATEGORIES:
        items = examples.get(name, [])
        if not items:
            continue
        add(f"### {name}")
        add("")
        for item in items[:10]:
            add(f"- `{item['source1_name']}`  ->  `{item['target_name']}`  ({item['target_id']})")
        add("")
    add("## Method")
    add("")
    add("Categories are applied in a fixed precedence order, cheapest and most")
    add("structural first: exact -> separator-only -> token order -> token")
    add("subset/superset -> script difference -> edit distance. A pair is only")
    add("called a typo when it survived every structural check. A shared whole")
    add("word rules out `substantially_different` even at a large edit distance,")
    add("and `substantially_different` additionally requires almost no shared")
    add("character trigrams. Everything in between stays `unknown_other` with a")
    add("recorded reason rather than being guessed at.")
    add("")
    add(f"Thresholds: typo <= {TYPO_MAX_NORM_EDIT:.2f} normalized edit distance; "
        f"different >= {DIFFERENT_MIN_NORM_EDIT:.2f} with trigram Jaccard "
        f"<= {DIFFERENT_MAX_TRIGRAM_JACCARD:.2f}.")
    add("")
    add("`separator_only` is by definition `name_key` equality, so the name_key")
    add("miss rate above is derived from that category rather than recomputed.")
    add("")
    add("Country values are the normalized ones (`country_norm`), so they are")
    add("lower-cased and accent-folded.")
    add("")
    add("Only ground-truth pairs are examined - no cross-product is ever formed.")
    add("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0.1: classify why the names of true matches differ.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--output-dir", default=None, help="default: <work_dir>/analysis")
    parser.add_argument(
        "--sources",
        default="source2,source3",
        help="comma-separated target sources to analyze",
    )
    parser.add_argument("--workers", type=int, default=0, help="0 = auto (config, else min(cpu_count, 8))")
    parser.add_argument("--chunk-pairs", type=int, default=100_000, help="pairs per work chunk")
    parser.add_argument("--limit-pairs", type=int, default=None, help="analyze only the first N pairs (smoke tests)")
    parser.add_argument("--examples-per-category", type=int, default=25)
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

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    unknown_sources = [s for s in sources if s not in SOURCE_CODES]
    if unknown_sources:
        raise SystemExit(f"unknown source(s): {unknown_sources}; expected source2/source3")

    log.info("=" * 78)
    log.info("Phase 0.1: name differences among true matches")
    log.info(describe_environment(config))
    log.info("=" * 78)

    # -- 1. ground truth -> true pairs (no cross-product) -------------------
    ground_truth = load_ground_truth(config, log=log)
    log_memory(log, "after ground truth")

    lengths = ground_truth.lengths()
    owners = np.repeat(np.arange(ground_truth.n_entities, dtype=np.int64), lengths)
    target_codes = ground_truth.codes
    source_of_pair = (target_codes // 10**10).astype(np.int8)

    keep = np.isin(source_of_pair, [SOURCE_CODES[s] for s in sources])
    if args.limit_pairs is not None:
        keep_indices = np.flatnonzero(keep)[: args.limit_pairs]
    else:
        keep_indices = np.flatnonzero(keep)
    owners = owners[keep_indices]
    target_codes = target_codes[keep_indices]
    source_of_pair = source_of_pair[keep_indices]
    n_pairs = len(keep_indices)
    log.info("true pairs to analyze: %s (of %s total)", fmt_int(n_pairs), fmt_int(len(ground_truth.codes)))
    if n_pairs == 0:
        raise SystemExit("no ground-truth pairs matched the requested sources")

    # -- 2. resolve names for exactly the entities involved -----------------
    country_vocabulary: dict[str, int] = {}
    target_names_by_source = {}
    target_country_by_source = {}
    offset_of_source: dict[int, int] = {}
    total_targets = 0
    for source in sources:
        code = SOURCE_CODES[source]
        needed = np.unique(target_codes[source_of_pair == code])
        offset_of_source[code] = total_targets
        names, countries = _load_target_names(config, source, needed, country_vocabulary, log)
        target_names_by_source[code] = names
        target_country_by_source[code] = countries
        total_targets += len(needed)

    names_b = np.empty(n_pairs, dtype=object)
    country_b = np.zeros(n_pairs, dtype=np.int32)
    for code in offset_of_source:
        mask = source_of_pair == code
        needed = np.unique(target_codes[mask])
        slots = np.searchsorted(needed, target_codes[mask])
        names_b[mask] = target_names_by_source[code][slots]
        country_b[mask] = target_country_by_source[code][slots]

    s1_names, s1_countries = _load_s1_names(config, ground_truth, country_vocabulary, log)
    names_a = s1_names[owners]
    country_a = s1_countries[owners]
    del s1_names
    log_memory(log, "after name resolution")

    # -- 3. classify, in parallel over chunks -------------------------------
    country_names = {code: name for name, code in country_vocabulary.items()}
    slices = {"combined": _Counts("combined")}
    for source in sources:
        slices[source] = _Counts(source)
    source_pairs = {source: int((source_of_pair == SOURCE_CODES[source]).sum()) for source in sources}
    log.info("pairs per source: %s", {k: fmt_int(v) for k, v in source_pairs.items()})

    # Country aggregates are per source1 country: total pairs and differing pairs.
    country_totals = np.zeros(len(country_vocabulary) + 2, dtype=np.int64)
    country_differs = np.zeros(len(country_vocabulary) + 2, dtype=np.int64)
    country_category = np.zeros((len(country_vocabulary) + 2, len(CATEGORIES)), dtype=np.int64)

    rng = np.random.default_rng(config.get("project", {}).get("seed", 42))
    examples: dict[str, list[dict]] = {name: [] for name in CATEGORIES}
    examples_seen = {name: 0 for name in CATEGORIES}
    cap = max(0, args.examples_per_category)

    n_chunks = (n_pairs + args.chunk_pairs - 1) // args.chunk_pairs
    workers = _resolve_workers(args.workers, config.get("compute", {}).get("num_workers", 0), n_chunks, log)

    started = time.time()
    processed = 0

    def consume(block: np.ndarray, result: np.ndarray) -> None:
        """Fold one classified chunk into every aggregate."""
        nonlocal processed
        categories = result[:, 0]
        chunk_source = source_of_pair[block]

        # name_key equality is exactly what the classifier tests at its
        # "separator_only" step, so it is read off the resulting category rather
        # than recomputing millions of space-stripped string comparisons in the
        # parent process, where they would serialize behind the workers.
        key_equal = (categories == CATEGORY_INDEX["exact_normalized_match"]) | (
            categories == CATEGORY_INDEX["separator_only"]
        )

        block_country_a = country_a[block]
        block_country_b = country_b[block]
        slices["combined"].add_chunk(result, key_equal, block_country_a, block_country_b)
        for source in sources:
            mask = chunk_source == SOURCE_CODES[source]
            if mask.any():
                slices[source].add_chunk(
                    result[mask], key_equal[mask], block_country_a[mask], block_country_b[mask]
                )

        # Country tables, keyed by the source1 entity's country.
        np.add.at(country_totals, block_country_a, 1)
        differing = categories != CATEGORY_INDEX["exact_normalized_match"]
        if differing.any():
            np.add.at(country_differs, block_country_a[differing], 1)
            np.add.at(country_category, (block_country_a[differing], categories[differing]), 1)

        # Representative examples: a bounded reservoir per category. Illustrative
        # only - the counts in the report are exact, these are a sample.
        if cap:
            for index, name in enumerate(CATEGORIES):
                rows = np.flatnonzero(categories == index)
                if not rows.size:
                    continue
                examples_seen[name] += int(rows.size)
                take = min(cap, rows.size)
                chosen = rng.choice(rows, size=take, replace=False) if rows.size > take else rows
                for offset in chosen.tolist():
                    absolute = block[offset]
                    record = {
                        "source": "S2" if chunk_source[offset] == 2 else "S3",
                        "source1_id": str(ground_truth.entity_ids[owners[absolute]]),
                        "target_id": decode_entity_id(int(target_codes[absolute])),
                        "source1_name": names_a[absolute],
                        "target_name": names_b[absolute],
                        "source1_tokens": int(result[offset, 2]),
                        "target_tokens": int(result[offset, 3]),
                        "unknown_reason": UNKNOWN_REASONS[int(result[offset, 1])],
                        "country": country_names.get(int(block_country_a[offset]), ""),
                    }
                    bucket = examples[name]
                    if len(bucket) < cap:
                        bucket.append(record)
                    else:
                        j = int(rng.integers(0, examples_seen[name]))
                        if j < cap:
                            bucket[j] = record

        processed += len(block)
        if processed % (args.chunk_pairs * 10) < args.chunk_pairs:
            log.info("  classified %s / %s pairs", fmt_int(processed), fmt_int(n_pairs))

    chunks = _iter_pair_chunks(names_a, names_b, n_pairs, args.chunk_pairs)
    if workers == 1:
        log.info("running single-process (workers=1)")
        for block, payload in chunks:
            consume(block, _classify_chunk(payload))
    else:
        # ProcessPoolExecutor.map() submits EVERY chunk before the first one is
        # consumed, which would queue the entire pair set in memory - the exact
        # thing chunking exists to avoid. A bounded sliding window keeps at most
        # 4 x workers chunks in flight, and consuming strictly in submission
        # order keeps the example sampling reproducible run to run.
        window: deque = deque()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for block, payload in chunks:
                window.append((block, pool.submit(_classify_chunk, payload)))
                if len(window) >= workers * 4:
                    done_block, future = window.popleft()
                    consume(done_block, future.result())
            while window:
                done_block, future = window.popleft()
                consume(done_block, future.result())

    elapsed = time.time() - started
    log.info("classified %s pairs in %.1f min", fmt_int(processed), elapsed / 60.0)
    log_memory(log, "after classification")

    # -- 4. report ----------------------------------------------------------
    combined = slices["combined"]
    differing_total = combined.n_pairs - int(combined.categories[CATEGORY_INDEX["exact_normalized_match"]])
    dominant = sorted(
        (
            {
                "category": name,
                "count": int(combined.categories[index]),
                "pct_of_differing": _pct(int(combined.categories[index]), differing_total),
                "pct_of_pairs": _pct(int(combined.categories[index]), combined.n_pairs),
            }
            for index, name in enumerate(CATEGORIES)
            if index != CATEGORY_INDEX["exact_normalized_match"] and combined.categories[index]
        ),
        key=lambda entry: -entry["count"],
    )

    country_entries = []
    if country_vocabulary:
        for name, code in country_vocabulary.items():
            total = int(country_totals[code])
            if not total:
                continue
            counts = country_category[code]
            top_index = int(np.argmax(counts))
            country_entries.append(
                {
                    "country": name,
                    "n_pairs": total,
                    "n_differs": int(country_differs[code]),
                    "pct_differs": _pct(int(country_differs[code]), total),
                    "top_category": CATEGORIES[top_index] if counts[top_index] else "",
                }
            )
        country_entries.sort(key=lambda entry: -entry["n_pairs"])

    report = {
        "meta": {
            "phase": "0.1",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sources": sources,
            "category_order": CATEGORIES,
            "unknown_reasons": UNKNOWN_REASONS,
            "thresholds": {
                "typo_max_norm_edit": TYPO_MAX_NORM_EDIT,
                "different_min_norm_edit": DIFFERENT_MIN_NORM_EDIT,
                "different_max_trigram_jaccard": DIFFERENT_MAX_TRIGRAM_JACCARD,
            },
            "exact_blocker_key_field": config.get("blocking", {}).get("exact_name", {}).get("key", "name_norm"),
            "workers": workers,
            "chunk_pairs": args.chunk_pairs,
            "limit_pairs": args.limit_pairs,
            "elapsed_minutes": round(elapsed / 60.0, 3),
            "prepared_dir": str(config["resolved"]["prepared_dir"]),
        },
        "totals": combined.summary(),
        "slices": {label: counts.summary() for label, counts in slices.items()},
        "categories": {label: counts.category_table() for label, counts in slices.items()},
        "dominant_reasons": dominant,
        "country": {"top": country_entries[:40], "n_countries": len(country_vocabulary)},
        "examples": {name: items for name, items in examples.items() if items},
    }

    output_dir = Path(args.output_dir) if args.output_dir else Path(config["resolved"]["work_dir"]) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "name_difference_report.json"
    md_path = output_dir / "name_difference_report.md"
    tsv_path = output_dir / "name_difference_examples.tsv"

    write_json(json_path, report)
    md_path.write_text(_render_markdown(report, examples), encoding="utf-8")

    example_rows = []
    for name in CATEGORIES:
        for item in examples.get(name, []):
            example_rows.append(
                {
                    "category": name,
                    "source": item["source"],
                    "source1_entity_id": item["source1_id"],
                    "target_entity_id": item["target_id"],
                    "source1_name": item["source1_name"],
                    "target_name": item["target_name"],
                    "source1_tokens": item["source1_tokens"],
                    "target_tokens": item["target_tokens"],
                    "unknown_reason": item["unknown_reason"],
                    "country": item["country"],
                }
            )
    pd.DataFrame(example_rows).to_csv(tsv_path, sep="\t", index=False, encoding="utf-8")

    log.info("wrote %s", json_path)
    log.info("wrote %s", md_path)
    log.info("wrote %s", tsv_path)
    log.info(
        "RESULT: exact-name blocking (key=%s) misses %.2f%% of true pairs",
        report["meta"]["exact_blocker_key_field"],
        report["totals"]["pct_name_norm_differs"],
    )
    log_memory(log, "final")
    return 0


if __name__ == "__main__":
    # Windows consoles default to cp1252 and cannot encode Devanagari/Kannada,
    # which appear in the examples. Force UTF-8 so the script is runnable locally.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
