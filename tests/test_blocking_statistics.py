"""Fixture test for ``scripts/analyze_blocking_statistics.py`` (Phase 0.2-0.5).

The fixture is 14 S1 / 9 S2 / 6 S3 rows and 11 true pairs, hand-built so that
every phase has an example of every branch it can take and every headline number
can be recomputed by hand:

* signals - all four combinations of the four cheap signals occur: pairs reached
  by all four (identical name and address), by token+char only (a token subset
  with no address on the S1 side), by char only (a one-character typo in a
  single-token name), by address only (a Devanagari name whose latin
  transliteration shares no token and no 3-gram), and by nothing at all (the
  irreducible residue, twice);
* the char signal - ``blue sky exports`` vs ``bluesky exports`` is a pair whose
  ``name_key`` is identical, so the signal has to be computed on ``name_key``;
  ``acme industries private limited`` vs ``acme private limited`` lands on
  exactly the reference threshold (14/28 = 0.5), so the inclusive comparison is
  pinned down too;
* the token signal - a pair whose rarest shared token is not its most frequent
  one (``acme`` df 4 vs ``industries`` df 3 vs ``private``/``limited`` df 1), so
  a min/max confusion in the ``reduceat`` grouping cannot pass;
* the census - nine entities with a usable token, one whose rarest token is df 3,
  four structural zeros (two names with no token in the corpus at all, one whose
  token only exists in a longer form, one non-latin name), and a pair of
  exact-name duplicate keys (``acme industries`` twice in source2);
* zero-match S1 entities - one with an exact ``name_norm`` candidate, one whose
  candidate appears only under ``name_key`` (spacing differs), one with no
  candidate at all, and one with candidates in both target sources.

Nothing here touches the dataset: the fixture is written to a temp directory,
normalized by the real ``scripts/prepare_data.py``, and analysed by the real
analyzer, which also reads two real (tiny) ``ExactNameIndex`` instances that this
file builds and persists into the fixture's own index directory. Runs standalone
(``python tests/test_blocking_statistics.py``) and under pytest.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts import analyze_blocking_statistics as abs_  # noqa: E402
from scripts import prepare_data  # noqa: E402
from src.blocking import (  # noqa: E402
    BLOCKER_EXACT_NAME,
    ExactNameIndex,
    build_index,
    index_dir_for,
)
from src.data_loader import iter_prepared, load_config  # noqa: E402
from src.utils import read_json  # noqa: E402

HEADER = ["entity_id", "business_name", "business_address", "country"]

# ---------------------------------------------------------------------------
# The fixture. Keep in sync with the expectations below - every change here
# invalidates the hand-computed numbers.
# ---------------------------------------------------------------------------
S1 = [
    ("S1-1", "Sunrise Traders", "1 Main St, Austin, TX", "US"),
    ("S1-2", "Blue Sky Exports", "2 Dock Rd, Seattle, WA", "US"),
    ("S1-3", "Global Tech Solutions", "3 High St, London", "United Kingdom"),
    ("S1-4", "Acme Industries Private Limited", "", "India"),
    ("S1-5", "Meridian Logistics", "5 Freight Way, Denver, CO", "US"),
    ("S1-6", "Zenith Foods", "7 Rue de Rivoli, Paris", "France"),
    ("S1-7", "Delta Freight", "9 Depot Ave, Memphis, TN", "US"),
    ("S1-8", "Sunrise Traders", "1 Main St, Austin, TX", "US"),      # zero match
    ("S1-9", "Blue Sky Exports", "2 Dock Rd, Seattle, WA", "US"),    # zero match
    ("S1-10", "Unique Widgets", "", "US"),                          # zero match
    ("S1-11", "Acme Industries", "4 MG Road, Mumbai", "India"),     # zero match
    ("S1-12", "Aurora Wholesale", "20 Elm St, Dallas, TX", "US"),
    ("S1-13", "Freightways", "30 Canyon Rd, Tucson, AZ", "US"),
    ("S1-14", "राम मार्केटिंग", "5 Karol Bagh, New Delhi", "India"),
]
S2 = [
    ("S2-201", "Sunrise Traders", "1 Main St, Austin, TX", "US"),
    ("S2-202", "BlueSky Exports", "2 Dock Rd, Seattle, WA", "US"),
    ("S2-203", "Global Tech Solutions", "3 High Street, London", "United Kingdom"),
    ("S2-204", "Acme Industries", "4 MG Road, Mumbai", "India"),
    ("S2-205", "Acme Industries", "4 MG Road, Mumbai", "India"),
    ("S2-206", "Quasar Holdings", "99 Elsewhere, Lyon", "France"),
    ("S2-207", "Northern Lights Trading", "77 Harbour Road, Vancouver", "Canada"),
    ("S2-208", "Freightwayss", "31 Canyon Road, Phoenix, AZ", "US"),
    ("S2-209", "Cascade Textiles", "88 Harbour Way, Reno, NV", "US"),
]
S3 = [
    ("S3-301", "Acme Private Limited", "4 MG Road, Mumbai", "India"),
    ("S3-302", "Meridian Logistics", "5 Freight Way, Denver, CO", "US"),
    ("S3-303", "Zenith Foods", "1 Rue de Rivoli, Paris", "France"),
    ("S3-304", "Acme Industries", "4 MG Road, Mumbai", "India"),
    ("S3-305", "Delta Freight", "9 Depot Ave, Memphis, TN", "US"),
    ("S3-306", "Ram Marketing", "5 Karol Bagh, New Delhi", "India"),
]
GROUND_TRUTH = [
    ("S1-1", "S2-201"),          # P1  exact_normalized_match, all four signals
    ("S1-2", "S2-202"),          # P2  separator_only: name_key identical
    ("S1-3", "S2-203"),          # P3  exact_normalized_match, address Jaccard 0.6
    ("S1-4", "S3-301"),          # P4  token_subset_superset, no S1 address
    ("S1-5", "S3-302"),          # P5  exact_normalized_match
    ("S1-6", "S3-303"),          # P6  exact_normalized_match, Jaccard 2/3
    ("S1-7", "S3-305"),          # P7  exact_normalized_match
    ("S1-8", ""),
    ("S1-9", ""),
    ("S1-10", ""),
    ("S1-11", ""),
    ("S1-12", "S2-207"),         # P8  residue: nothing shared, on any signal
    ("S1-13", "S2-208,S2-209"),  # P9  char only; P10 residue
    ("S1-14", "S3-306"),         # P11 address only (different script, no token, no 3-gram)
]

# ---------------------------------------------------------------------------
# Hand-computed expectations
# ---------------------------------------------------------------------------
N_PAIRS = 11
# Entities owning at least one analysed pair: S1-1..S1-7, S1-12, S1-13, S1-14.
N_ENTITIES_WITH_PAIRS = 10

# signal -> (n_pairs reached, pct of pairs, n_entities reached)
EXPECTED_SIGNALS = {
    "exact_name": (5, 45.4545, 5),          # P1, P3, P5, P6, P7
    "rare_token_name": (7, 63.6364, 7),     # + P2, P4
    "char_3gram_name": (8, 72.7273, 8),     # + P9 (char 0.9)
    "address_jaccard": (7, 63.6364, 7),     # P1..P3, P5..P7 + P11 (identical address)
}
EXPECTED_UNION = (9, 81.8182, 9)
EXPECTED_RESIDUE = (2, 18.1818, 2)

# Reached-by-subset histogram. Only five cells are non-zero, and the shape is the
# point: the union is not the sum of the signals (they overlap heavily), and the
# two empty cells are the residue.
EXPECTED_COMBINATIONS = {
    "exact_name": 0,
    "rare_token_name": 0,
    "char_3gram_name": 1,                                  # P9
    "address_jaccard": 1,                                  # P11
    "exact_name+rare_token_name": 0,
    "exact_name+char_3gram_name": 0,
    "exact_name+address_jaccard": 0,
    "rare_token_name+char_3gram_name": 1,                  # P4: no S1 address
    "rare_token_name+address_jaccard": 0,
    "char_3gram_name+address_jaccard": 0,
    "exact_name+rare_token_name+char_3gram_name": 0,
    "exact_name+rare_token_name+address_jaccard": 0,
    "exact_name+char_3gram_name+address_jaccard": 0,
    "rare_token_name+char_3gram_name+address_jaccard": 1,  # P2: spacing only
    "exact_name+rare_token_name+char_3gram_name+address_jaccard": 5,   # P1,P3,P5,P6,P7
    "none": 2,                                             # P8, P10
}
# Entity outcomes: S1-1..S1-7 and S1-14 have every pair reached, S1-13 has one of
# two, S1-12 has none.
EXPECTED_ENTITY_OUTCOMES = {"all_pairs_reached": 8, "some_pairs_reached": 1, "no_pair_reached": 1}

EXPECTED_PER_SOURCE = {
    "source2": {  # P1, P2, P3, P8, P9, P10
        "n_pairs": 6,
        "signals": {"exact_name": 2, "rare_token_name": 3, "char_3gram_name": 4, "address_jaccard": 3},
        "union": 4,
        "residue": 2,
    },
    "source3": {  # P4, P5, P6, P7, P11
        "n_pairs": 5,
        "signals": {"exact_name": 3, "rare_token_name": 4, "char_3gram_name": 4, "address_jaccard": 4},
        "union": 5,
        "residue": 0,
    },
}

# The shipped address grid is calibrated for a real corpus, but this fixture has
# only eleven pairs so every row is hand-checkable. Jaccard values: P1 1.0,
# P2 1.0, P3 0.6, P5 1.0, P6 2/3, P7 1.0, P8 0.0, P9 0.25, P10 0.0, P11 1.0.
EXPECTED_ADDRESS_SENSITIVITY = {0.2: 8, 0.3: 7, 0.5: 7, 0.7: 5, 0.8: 5, 0.9: 5}

# Same for the character grid, and this one is NOT degenerate: P4's name_key
# Jaccard is exactly 0.5, so it drops out as soon as the threshold passes it.
# char_sim per pair: P1..P3, P5..P7 = 1.0, P4 = 0.5, P9 = 0.9, P8/P10/P11 = 0.
EXPECTED_CHAR_SENSITIVITY = {
    0.3: 8, 0.4: 8, 0.5: 8, 0.6: 7, 0.7: 7, 0.8: 7, 0.9: 7,
}

# Coverage sliced by Phase 0.1's category. The transliteration row is the
# decision-relevant one: an address-only pair is invisible to every name signal.
EXPECTED_BY_CATEGORY = {
    "exact_normalized_match": {
        "n_pairs": 5, "exact_name": 5, "rare_token_name": 5, "char_3gram_name": 5,
        "address_jaccard": 5, "union": 5, "residue": 0,
    },
    "separator_only": {
        "n_pairs": 1, "exact_name": 0, "rare_token_name": 1, "char_3gram_name": 1,
        "address_jaccard": 1, "union": 1, "residue": 0,
    },
    "token_subset_superset": {
        "n_pairs": 1, "exact_name": 0, "rare_token_name": 1, "char_3gram_name": 1,
        "address_jaccard": 0, "union": 1, "residue": 0,
    },
    "typo_small_edit": {
        "n_pairs": 1, "exact_name": 0, "rare_token_name": 0, "char_3gram_name": 1,
        "address_jaccard": 0, "union": 1, "residue": 0,
    },
    "transliteration_script": {
        "n_pairs": 1, "exact_name": 0, "rare_token_name": 0, "char_3gram_name": 0,
        "address_jaccard": 1, "union": 1, "residue": 0,
    },
    "substantially_different": {
        "n_pairs": 2, "exact_name": 0, "rare_token_name": 0, "char_3gram_name": 0,
        "address_jaccard": 0, "union": 0, "residue": 2,
    },
}

# Phase 0.3. Shared address tokens per pair: P1 5, P2 5, P3 3, P4 0, P5 5, P6 4,
# P7 5, P8 0, P9 2, P10 0, P11 5.
EXPECTED_ADDRESS_COUNTS = {
    "n_pairs": 11,
    "n_both_addresses_present": 10,     # P4 has an empty S1 address
    "n_either_address_missing": 1,
    "n_zero_shared_tokens": 3,          # P4 (missing), P8, P10
    "n_at_least_1_shared_token": 8,
    "n_at_least_2_shared_tokens": 8,
    "n_at_least_3_shared_tokens": 7,
    "n_jaccard_at_least_0_5": 7,        # P3 0.6 and P6 2/3 included
    "n_jaccard_at_least_0_8": 5,        # P3, P6 fall out
    "n_overlap_coefficient_at_least_0_8": 6,   # P6 = 4/5 exactly
    "n_overlap_coefficient_equal_1": 5,
}
EXPECTED_ADDRESS_PCT = {
    "zero_shared_tokens_of_comparable": 30.0,
    "jaccard_at_least_0_5_of_comparable": 70.0,
    "overlap_coefficient_at_least_0_8_of_comparable": 60.0,
}
EXPECTED_ADDRESS_SHARED_HISTOGRAM = {"0": 3, "2": 1, "3": 1, "4": 1, "5": 5}

# Phase 0.4. name_key of "acme"+"industries" spans source2 (2 rows) and source3
# (1 row) among the matched pairs, but the frequency table counts EVERY row of
# each analysed source, so df(acme) = 4 (S2-204, S2-205, S3-301, S3-304).
EXPECTED_TOKENS = {
    "source2": {"n_entities_in_scope": 9, "n_unique_tokens": 17, "n_token_occurrences": 19},
    "source3": {"n_entities_in_scope": 6, "n_unique_tokens": 12, "n_token_occurrences": 13},
    "all": {"n_entities_in_scope": 15, "n_unique_tokens": 27, "n_token_occurrences": 32},
}
EXPECTED_RARITY = {
    "n_all_true_pairs": 11,
    "n_name_norm_differs": 6,           # P2, P4, P8, P9, P10, P11
    "n_name_norm_identical": 5,
    "n_all_with_shared_token": 7,
    "n_usable_cap_1000": 7,
    "estimated_candidates_cap_1000": 7,
}

# The census. Estimates per S1 entity, in fixture order: one candidate for each of
# S1-1..S1-9 (rarest token df 1), three for S1-11 (rarest usable token is
# "industries", df 3), nothing for S1-10, S1-12, S1-13, S1-14.
EXPECTED_CENSUS_ESTIMATES = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0]
EXPECTED_STRUCTURAL_ZERO = {
    "empty_name_norm": 0,
    "no_token_in_target_corpus": 4,     # S1-10, S1-12, S1-13, S1-14
    "every_token_above_the_cap": 0,
}

EXPECTED_ZERO_MATCH = {
    "n_zero_match_entities": 4,         # S1-8, S1-9, S1-10, S1-11
    "pct_of_all_source1": 28.5714,      # 4 of 14
    "exact_name_norm_entities": 2,      # S1-8 (1 row), S1-11 (3 rows)
    "exact_name_norm_pct": 50.0,
    "name_key_only_entities": 1,        # S1-9: "blue sky exports" vs "bluesky exports"
    "name_key_only_pct": 25.0,
    "no_candidate_at_all_entities": 1,  # S1-10
    "no_candidate_at_all_pct": 25.0,
    "source2_entities": 2,
    "source3_entities": 1,
    "n_candidate_pairs": 4,
    "n_pairs_both_addresses_present": 4,
    "n_pairs_with_zero_address_overlap": 0,
    "max_candidates": 3,
    "n_entities_with_any_address_support": 2,
}


# ---------------------------------------------------------------------------
# fixture construction
# ---------------------------------------------------------------------------
def _write_tsv(path: Path, rows, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def _build_fixture(base: Path) -> Path:
    """Write the raw TSVs plus a config derived from the real one."""
    data = base / "data"
    work = base / "work"
    _write_tsv(data / "train_source1.tsv", S1, HEADER)
    _write_tsv(data / "train_source2.tsv", S2, HEADER)
    _write_tsv(data / "train_source3.tsv", S3, HEADER)
    _write_tsv(
        data / "train_ground_truth.tsv",
        GROUND_TRUTH,
        ["source1_entity_id", "matched_entity_ids"],
    )

    source = (REPO / "configs" / "config.yaml").read_text(encoding="utf-8")
    replacements = {
        '  data_root: "train"': f'  data_root: "{data.as_posix()}"',
        '  work_dir: "outputs"': f'  work_dir: "{work.as_posix()}"',
        '  prepared_dir: "outputs/prepared"': f'  prepared_dir: "{(work / "prepared").as_posix()}"',
        '  index_dir: "outputs/indexes"': f'  index_dir: "{(work / "indexes").as_posix()}"',
        '  log_dir: "logs"': f'  log_dir: "{(work / "logs").as_posix()}"',
        "  chunksize: 500000": "  chunksize: 3",  # force multi-chunk streaming
    }
    for old, new in replacements.items():
        assert old in source, f"config line not found: {old!r}"
        source = source.replace(old, new)
    # Nothing in the fixture config may point at a bare relative path.
    assert not re.search(r'^\s*(work_dir|prepared_dir|index_dir|log_dir): "(?![/A-Za-z]:)', source, re.M)

    config_path = base / "config_blocking_fixture.yaml"
    config_path.write_text(source, encoding="utf-8")
    return config_path


def _prepare(config_path: Path) -> None:
    """Run the real normalization stage over the fixture."""
    code = prepare_data.main(
        [
            "--config", str(config_path),
            "--splits", "train",
            "--sources", "source1,source2,source3",
            "--overwrite",
        ]
    )
    assert code == 0


def _build_exact_indexes(config_path: Path) -> None:
    """Persist the real exact-name index for each target source, via the real builder.

    Phase 0.4's census reads the SHIPPED index read-only, so the test has to give
    it one to read: without this the census degrades to its ``available: False``
    branch and the exact-name numbers would never be exercised.
    """
    config = load_config(str(config_path))
    for source in ("source2", "source3"):
        build_index(config, "train", source, BLOCKER_EXACT_NAME, overwrite=True)


_CACHE: dict[str, object] = {}


def _analyze(config_path: Path, output_dir: Path, workers: int, chunk_pairs: int) -> dict:
    """Run the analyzer; returns the parsed JSON report."""
    key = f"{config_path}:{workers}:{chunk_pairs}"
    if key in _CACHE:
        return _CACHE[key]  # type: ignore[return-value]
    code = abs_.main(
        [
            "--config", str(config_path),
            "--workers", str(workers),
            "--chunk-pairs", str(chunk_pairs),
            "--output-dir", str(output_dir),
            "--log-level", "WARNING",
        ]
    )
    assert code == 0
    report = read_json(output_dir / "blocking_statistics_report.json")
    _CACHE[key] = report
    return report


def _fixture() -> tuple[Path, Path]:
    """Prepare the fixture once per process; returns (config_path, work_dir).

    The prepared tables and the two exact-name indexes are built once and reused
    by every test, so a full run normalizes 29 rows and analyses them a handful of
    times - seconds, not minutes. The temp directory is removed at interpreter
    exit unless ``ER_TEST_KEEP_FIXTURE`` is set, which is how you inspect a
    failing run.
    """
    if "root" not in _CACHE:
        root = Path(tempfile.mkdtemp(prefix="er_blocking_"))
        config_path = _build_fixture(root)
        _prepare(config_path)
        _build_exact_indexes(config_path)
        _CACHE["root"] = root
        _CACHE["config_path"] = config_path
        print(f"[fixture] {root}")
        if not os.environ.get("ER_TEST_KEEP_FIXTURE"):
            atexit.register(shutil.rmtree, root, ignore_errors=True)
    return _CACHE["config_path"], _CACHE["root"]  # type: ignore[return-value]


def _report(workers: int = 1, chunk_pairs: int = 2) -> dict:
    config_path, root = _fixture()
    return _analyze(config_path, root / f"analysis_w{workers}_c{chunk_pairs}", workers, chunk_pairs)


def _run(config_path: Path, output_dir: Path, extra: list[str]) -> dict:
    """One extra analyzer run with a bespoke CLI, cached by its arguments."""
    key = f"{output_dir}:{' '.join(extra)}"
    if key not in _CACHE:
        code = abs_.main(
            [
                "--config", str(config_path),
                "--workers", "1",
                "--chunk-pairs", "2",
                "--output-dir", str(output_dir),
                "--log-level", "WARNING",
                *extra,
            ]
        )
        assert code == 0
        _CACHE[key] = read_json(output_dir / "blocking_statistics_report.json")
    return _CACHE[key]  # type: ignore[return-value]


# Fixture-scaled thresholds, so the sensitivity columns actually move. The shipped
# grids start at df 100 / Jaccard 0.3 - calibrated for millions of rows - and on a
# 15-target fixture every row of the token grid is the same number, which proves
# the column exists but not that the comparison is applied.
_PATCHED_CONSTANTS = {
    "SIGNAL_TOKEN_CAP_GRID": (0, 1),
    "SIGNAL_CHAR_JACCARD_GRID": (0.5, 0.95, 1.0),
    "SIGNAL_ADDRESS_JACCARD_GRID": (0.25, 0.3, 1.0),
    "REFERENCE_TOKEN_CAP": 0,
}


def _patched_report() -> dict:
    """One run with the module's thresholds replaced by fixture-scale values."""
    config_path, root = _fixture()
    key = "patched"
    if key not in _CACHE:
        saved = {name: getattr(abs_, name) for name in _PATCHED_CONSTANTS}
        try:
            for name, value in _PATCHED_CONSTANTS.items():
                setattr(abs_, name, value)
            _CACHE[key] = _run(config_path, root / "analysis_patched", [])
        finally:
            for name, value in saved.items():
                setattr(abs_, name, value)
    return _CACHE[key]  # type: ignore[return-value]


def _strip_volatile(report: dict) -> dict:
    """The report without anything that legitimately differs between runs.

    Anything machine- or schedule-dependent belongs in this set: the memory and
    hardware readings move between runs on one box, the timings move between
    workers, and the queue sizing follows the worker count. What must NOT be added
    here is any field that carries a result - that would hide a real regression.
    """
    cleaned = {key: value for key, value in report.items() if key != "meta"}
    meta = {
        key: value
        for key, value in report["meta"].items()
        if key
        not in {
            "generated_at",
            "elapsed_minutes",
            "workers",
            "chunk_pairs",
            "chunk_pairs_requested",
            "in_flight_window",
            "peak_rss",
            "resume_requested",
            "resumed_phases",
            "hardware",
            "phase_seconds",
        }
    }
    return {"meta": meta, **cleaned}


def _close(actual: float, expected: float, places: int = 4) -> bool:
    return abs(float(actual) - expected) < 10 ** (-places) + 1e-9


def _round6(value: float) -> float:
    return round(float(value), 6)


def _markdown() -> str:
    """The Markdown report of the default run."""
    _, root = _fixture()
    return (root / "analysis_w1_c2" / "blocking_statistics_report.md").read_text(encoding="utf-8")


def _all_source1_names(config: dict) -> list[str]:
    """The ``name_norm`` of every source1 entity, in file order."""
    names: list[str] = []
    for chunk in iter_prepared(config, "train", "source1"):
        names.extend(chunk["name_norm"].tolist())
    assert len(names) == len(S1)
    return names


def _zero_match_names(config: dict) -> list[str]:
    """The ``name_norm`` of every S1 entity whose ground truth is empty."""
    empty = {row[0] for row in GROUND_TRUTH if row[1] == ""}
    names: list[str] = []
    for chunk in iter_prepared(config, "train", "source1"):
        mask = chunk["entity_id"].isin(empty)
        if mask.any():
            names.extend(chunk.loc[mask, "name_norm"].tolist())
    assert len(names) == len(empty)
    return names


def _coverage() -> dict:
    return _report()["signal_coverage"]


# ---------------------------------------------------------------------------
# Phase 0.2: individual signals, union, entity weighting
# ---------------------------------------------------------------------------
def test_individual_signal_coverage() -> None:
    """Each of the four signals reaches the pairs the fixture says it should."""
    coverage = _coverage()
    assert coverage["n_pairs"] == N_PAIRS
    assert coverage["n_source1_entities_with_analysed_pairs"] == N_ENTITIES_WITH_PAIRS
    assert list(coverage["signals"]) == list(abs_.SIGNALS)
    for signal, (n_pairs, pct, n_entities) in EXPECTED_SIGNALS.items():
        entry = coverage["signals"][signal]
        assert entry["n_pairs"] == n_pairs, signal
        assert _close(entry["pct_of_pairs"], pct), signal
        assert entry["n_entities"] == n_entities, signal
        assert _close(entry["pct_of_entities_with_pairs"], round(100.0 * n_entities / N_ENTITIES_WITH_PAIRS, 4))
    # The reference point is a reading position shared by every table, and it is
    # reported so a reader knows which row of the sensitivity grid each headline
    # number came from.
    assert coverage["reference_point"] == {
        abs_.TOKEN_SIGNAL: abs_.REFERENCE_TOKEN_CAP,
        abs_.CHAR_SIGNAL: abs_.REFERENCE_CHAR_JACCARD,
        abs_.ADDRESS_SIGNAL: abs_.REFERENCE_ADDRESS_JACCARD,
    }
    assert set(coverage["signals"]) == set(coverage["signal_definitions"])
    assert "name_norm" in coverage["signal_definitions"][abs_.EXACT_SIGNAL]
    assert "name_key" in coverage["signal_definitions"][abs_.CHAR_SIGNAL]


def test_union_and_the_combination_histogram() -> None:
    """The joint table, and the residue as its 'none' cell."""
    coverage = _coverage()
    union = coverage["union"]
    n_pairs, pct, n_entities = EXPECTED_UNION
    assert union["n_pairs"] == n_pairs
    assert _close(union["pct_of_pairs"], pct)
    assert union["n_entities"] == n_entities

    observed = {row["signals"]: row["n_pairs"] for row in union["by_combination"]}
    assert observed == EXPECTED_COMBINATIONS
    # Every pair lands in exactly one cell, and the cells are a partition: a pair
    # cannot be reached by the union and be in the 'none' cell.
    assert sum(observed.values()) == N_PAIRS
    assert observed["none"] == coverage["residue"]["n_pairs"] == EXPECTED_RESIDUE[0]
    assert union["n_pairs"] == N_PAIRS - observed["none"]
    # The four-signal cell is why the union is smaller than the sum of the parts.
    assert observed["exact_name+rare_token_name+char_3gram_name+address_jaccard"] == 5
    assert sum(entry["n_pairs"] for entry in coverage["signals"].values()) == 27


def test_entity_weighted_outcomes() -> None:
    """The view macro F0.5 averages over: per entity, not per pair."""
    outcomes = _coverage()["union"]["entity_outcomes"]
    assert outcomes["n_entities"] == N_ENTITIES_WITH_PAIRS
    for label, expected in EXPECTED_ENTITY_OUTCOMES.items():
        assert outcomes[label]["n_entities"] == expected, label
    assert (
        sum(outcomes[label]["n_entities"] for label in EXPECTED_ENTITY_OUTCOMES)
        == N_ENTITIES_WITH_PAIRS
    )
    # S1-13 owns two pairs and only one of them is reached: the pair view alone
    # would never show that, and the entity view is why it is reported next to it.
    assert outcomes["some_pairs_reached"]["n_entities"] == 1
    assert _close(outcomes["all_pairs_reached"]["pct"], 80.0)
    assert _close(outcomes["no_pair_reached"]["pct"], 10.0)
    assert "entity" in outcomes["definition"]


def test_per_source_breakdown() -> None:
    """S2 and S3 are reported separately, and they partition the pair list."""
    per_source = _coverage()["per_source"]
    assert set(per_source) == {"source2", "source3"}
    for source, expected in EXPECTED_PER_SOURCE.items():
        entry = per_source[source]
        assert entry["n_pairs"] == expected["n_pairs"], source
        for signal, count in expected["signals"].items():
            assert entry["signals"][signal]["n_pairs"] == count, f"{source}.{signal}"
        assert entry["union"]["n_pairs"] == expected["union"], source
        assert entry["residue"]["n_pairs"] == expected["residue"], source
    assert (
        per_source["source2"]["n_pairs"] + per_source["source3"]["n_pairs"] == N_PAIRS
    )
    assert (
        per_source["source2"]["residue"]["n_pairs"]
        + per_source["source3"]["residue"]["n_pairs"]
        == EXPECTED_RESIDUE[0]
    )


def test_address_signal_needs_both_addresses() -> None:
    """A pair with an empty S1 address is out of reach for the address signal."""
    coverage = _coverage()
    # P4 has no S1 address; P11 shares an identical address but nothing else.
    assert coverage["signals"][abs_.ADDRESS_SIGNAL]["n_pairs"] == 7
    observed = {row["signals"]: row["n_pairs"] for row in coverage["union"]["by_combination"]}
    assert observed["address_jaccard"] == 1        # P11: address-only
    assert observed["rare_token_name+char_3gram_name"] == 1   # P4: no address to use


# ---------------------------------------------------------------------------
# Phase 0.2: thresholds, cost, residue, categories
# ---------------------------------------------------------------------------
def test_sensitivity_grids_are_reported_for_every_threshold() -> None:
    """Every threshold in every grid with its coverage and its candidate cost."""
    sensitivity = _coverage()["sensitivity"]
    assert set(sensitivity) == {abs_.TOKEN_SIGNAL, abs_.CHAR_SIGNAL, abs_.ADDRESS_SIGNAL}
    assert [row["threshold"] for row in sensitivity[abs_.TOKEN_SIGNAL]] == list(abs_.SIGNAL_TOKEN_CAP_GRID)
    assert [row["threshold"] for row in sensitivity[abs_.CHAR_SIGNAL]] == list(abs_.SIGNAL_CHAR_JACCARD_GRID)
    assert [row["threshold"] for row in sensitivity[abs_.ADDRESS_SIGNAL]] == list(abs_.SIGNAL_ADDRESS_JACCARD_GRID)

    # The fixture's largest df is 4, so the shipped caps (>= 100) never bite: the
    # same seven pairs are usable at every one of them. The cost column is a real
    # sum of posting-list sizes, not an estimate.
    tokens = sensitivity[abs_.TOKEN_SIGNAL]
    assert [row["n_pairs"] for row in tokens] == [7] * len(abs_.SIGNAL_TOKEN_CAP_GRID)
    for row in tokens:
        cost = row["candidate_cost"]
        assert cost["kind"] == "posting_list"
        assert cost["n_candidates"] == 7
        assert cost["mean_per_pair"] == 1.0
        assert cost["max_per_pair"] == 1
    # Coverage can only grow as the cap loosens.
    assert [row["n_pairs"] for row in tokens] == sorted(row["n_pairs"] for row in tokens)

    # A character-3-gram index does not exist yet, so its cost column must say so
    # rather than pretend to a number. Coverage is not flat here: P4 sits exactly
    # on 0.5 and is gone from 0.6 up, so the threshold column is doing work even
    # on a fixture this small.
    chars = {row["threshold"]: row for row in sensitivity[abs_.CHAR_SIGNAL]}
    for threshold, expected in EXPECTED_CHAR_SENSITIVITY.items():
        assert chars[threshold]["n_pairs"] == expected, threshold
    assert all(row["candidate_cost"]["kind"] == "not_estimable_in_phase_0" for row in chars.values())
    assert all(row["candidate_cost"]["n_candidates"] is None for row in chars.values())

    # Address Jaccard is a filter, not a generator: it can only remove candidates.
    address = {row["threshold"]: row for row in sensitivity[abs_.ADDRESS_SIGNAL]}
    for threshold, expected in EXPECTED_ADDRESS_SENSITIVITY.items():
        assert address[threshold]["n_pairs"] == expected, threshold
        assert address[threshold]["candidate_cost"] == {
            "kind": "filter", "n_candidates": 0, "mean_per_pair": 0.0, "max_per_pair": 0,
        }
    counts = [address[threshold]["n_pairs"] for threshold in abs_.SIGNAL_ADDRESS_JACCARD_GRID]
    assert counts == sorted(counts, reverse=True)


def test_thresholds_actually_bite_when_scaled_to_the_fixture() -> None:
    """With fixture-scaled grids the threshold columns move, not just exist."""
    sensitivity = _patched_report()["signal_coverage"]["sensitivity"]
    # (0, 1): nothing is usable at cap 0, all seven shared-token pairs at cap 1.
    assert [row["n_pairs"] for row in sensitivity[abs_.TOKEN_SIGNAL]] == [0, 7]
    assert sensitivity[abs_.TOKEN_SIGNAL][0]["candidate_cost"]["n_candidates"] == 0
    # 0.95 and 1.0 keep only the six pairs whose keys are byte-identical: P4 is
    # 0.5 and P9 is 0.9, so both are gone at 0.95.
    assert [row["n_pairs"] for row in sensitivity[abs_.CHAR_SIGNAL]] == [8, 6, 6]
    # 0.25 is exactly P9's Jaccard, so the inclusive comparison shows up here.
    assert [row["n_pairs"] for row in sensitivity[abs_.ADDRESS_SIGNAL]] == [8, 7, 5]


def test_residue_is_what_no_cheap_signal_reaches() -> None:
    """Two pairs, both substantially different, both in source2, same script."""
    residue = _coverage()["residue"]
    assert residue["n_pairs"] == EXPECTED_RESIDUE[0]
    assert _close(residue["pct_of_pairs"], EXPECTED_RESIDUE[1])
    assert residue["n_entities"] == EXPECTED_RESIDUE[2]
    assert residue["by_name_difference_category"] == {
        "substantially_different": {"n_pairs": 2, "pct_of_residue": 100.0}
    }
    # substantially_different is a confident verdict, not an unknown_other outcome,
    # so there is no reason to report - the heuristic is not confused, the pairs
    # are simply unrelated by every cheap signal.
    assert residue["by_name_difference_reason"] == {}
    # Cross-tab with the script relation: both residue pairs are latin-latin, so
    # the residue is not a transliteration problem in this fixture - that problem
    # shows up in the category table as an address-only category instead.
    assert residue["by_script_relation"] == {"different_script": 0, "same_script": 2}
    assert residue["per_source"]["source2"]["n_pairs"] == 2
    assert residue["per_source"]["source3"]["n_pairs"] == 0

    examples = residue["examples"]
    assert [(row["source1_entity_id"], row["target_entity_id"]) for row in examples] == [
        ("S1-12", "S2-207"),
        ("S1-13", "S2-209"),
    ]
    assert examples[0]["name_norm_source1"] == "aurora wholesale"
    assert examples[0]["name_norm_target"] == "northern lights trading"
    assert examples[1]["name_norm_source1"] == "freightways"
    assert examples[1]["name_norm_target"] == "cascade textiles"
    assert all(row["name_difference_category"] == "substantially_different" for row in examples)


def test_coverage_by_name_difference_category() -> None:
    """The same coverage sliced by why the two names differ."""
    table = _coverage()["coverage_by_name_difference_category"]
    assert set(table) == set(EXPECTED_BY_CATEGORY)
    # unknown_other is absent because no pair in this fixture lands in it.
    assert "unknown_other" not in table
    for category, expected in EXPECTED_BY_CATEGORY.items():
        entry = table[category]
        assert entry["n_pairs"] == expected["n_pairs"], category
        for signal in abs_.SIGNALS:
            assert entry["signals"][signal]["n_pairs"] == expected[signal], f"{category}.{signal}"
        assert entry["union"]["n_pairs"] == expected["union"], category
        assert entry["residue"]["n_pairs"] == expected["residue"], category

    # The decision-relevant row: a transliterated name is invisible to every name
    # signal and reachable only through the address.
    transliteration = table["transliteration_script"]
    assert transliteration["signals"][abs_.ADDRESS_SIGNAL]["n_pairs"] == 1
    for signal in (abs_.EXACT_SIGNAL, abs_.TOKEN_SIGNAL, abs_.CHAR_SIGNAL):
        assert transliteration["signals"][signal]["n_pairs"] == 0, signal
    assert _close(transliteration["union"]["pct_of_category"], 100.0)

    # A one-character typo in a single-token name is a char-signal problem only.
    typo = table["typo_small_edit"]
    assert typo["signals"][abs_.CHAR_SIGNAL]["n_pairs"] == 1
    assert typo["signals"][abs_.EXACT_SIGNAL]["n_pairs"] == 0
    assert typo["signals"][abs_.TOKEN_SIGNAL]["n_pairs"] == 0

    # The categorised pairs partition the pair list.
    assert sum(entry["n_pairs"] for entry in table.values()) == N_PAIRS


def test_character_signal_is_computed_on_the_separator_free_key() -> None:
    """P2 is reachable by the char signal because name_key ignores the space."""
    stats = abs_._pair_statistics(
        (
            ["blue sky exports"],
            ["bluesky exports"],
            ["blueskyexports"],
            ["blueskyexports"],
            [""],
            [""],
        ),
        classify_names=False,
    )
    assert float(stats["char_sim"][0]) == 1.0
    # ...and the same two names on name_norm would not be, which is the point of
    # using the key: separators must not look like name differences.
    assert abs_._trigram_jaccard("blue sky exports", "bluesky exports") < 1.0


def test_reference_threshold_is_inclusive() -> None:
    """P4 sits exactly on 0.5 and must be counted as reached."""
    assert abs_._trigram_jaccard("acmeindustriesprivatelimited", "acmeprivatelimited") == 0.5
    coverage = _coverage()
    assert coverage["reference_point"][abs_.CHAR_SIGNAL] == 0.5
    observed = {row["signals"]: row["n_pairs"] for row in coverage["union"]["by_combination"]}
    # P4 is the only pair reached by token+char and nothing else, which can only
    # be true if the 0.5 boundary counts as a hit.
    assert observed["rare_token_name+char_3gram_name"] == 1


# ---------------------------------------------------------------------------
# Phase 0.3
# ---------------------------------------------------------------------------
def test_address_overlap_counts() -> None:
    """Shared-token thresholds, Jaccard counts and the missing-address pair."""
    entry = _report()["address_overlap"]["slices"]["all"]
    for key, expected in EXPECTED_ADDRESS_COUNTS.items():
        assert entry["counts"][key] == expected, f"{key}: {entry['counts'][key]} != {expected}"
    for key, expected in EXPECTED_ADDRESS_PCT.items():
        assert _close(entry["pct"][key], expected), key
    assert _report()["address_overlap"]["shared_token_histogram"] == EXPECTED_ADDRESS_SHARED_HISTOGRAM


def test_address_jaccard_distribution() -> None:
    """Jaccard is computed over pairs with two addresses, and equals 0.6 and 2/3."""
    entry = _report()["address_overlap"]["slices"]["all"]
    jaccard = entry["jaccard"]
    assert jaccard["n"] == 10
    assert jaccard["min"] == 0.0
    assert jaccard["max"] == 1.0
    # (1 + 1 + 0.6 + 1 + 0.6667 + 1 + 0 + 0.25 + 0 + 1) / 10
    assert _close(jaccard["mean"], 0.651667)
    assert entry["source1_address_tokens"]["mean"] is not None
    # P9 (0.25) and P3 (0.6) are the two partial overlaps; the rest are exact or
    # disjoint, which is why the counts above separate 0.5 from 0.8.
    assert _close(entry["overlap_coefficient"]["max"], 1.0)


def test_address_overlap_slices_by_source_and_name_equality() -> None:
    """The split that survives is name equality, not country agreement."""
    overlap = _report()["address_overlap"]
    assert set(overlap["slices"]) == {"all", "source2", "source3"}
    groups = overlap["slices"]["all"]["by_group"]
    assert set(groups) == {"both_addresses_present", "name_norm_identical", "name_norm_differs"}
    identical = groups["name_norm_identical"]
    assert identical["counts"]["n_pairs"] == 5
    assert identical["counts"]["n_zero_shared_tokens"] == 0
    assert identical["counts"]["n_jaccard_at_least_0_5"] == 5
    differing = groups["name_norm_differs"]
    assert differing["counts"]["n_pairs"] == 6
    # P4 (missing address), P8 and P10 share no address token.
    assert differing["counts"]["n_zero_shared_tokens"] == 3
    assert differing["counts"]["n_jaccard_at_least_0_5"] == 2   # P2, P11
    assert overlap["slices"]["source2"]["counts"]["n_pairs"] == 6
    assert overlap["slices"]["source3"]["counts"]["n_pairs"] == 5
    assert "country" not in (overlap["slices"]["all"].get("by_group") or {})


def test_no_country_phase_remains() -> None:
    """Phase 0.2 is now signal coverage: the country phase is gone, not hidden."""
    report = _report()
    assert "country_agreement" not in report
    assert "country_vocabulary_size" not in report["meta"]
    assert "signal_coverage" in report
    markdown = _markdown()
    assert "Phase 0.2 - signal coverage and complementarity" in markdown
    for stale in ("country_agreement", "country_vocabulary", "Country agreement"):
        assert stale not in markdown, stale
    _, root = _fixture()
    coverage_csv = pd.read_csv(root / "analysis_w1_c2" / "signal_coverage.csv")
    assert not any("country" in str(value) for value in coverage_csv["section"].unique())


# ---------------------------------------------------------------------------
# Phase 0.4: token frequency, pair rarity, census
# ---------------------------------------------------------------------------
def test_token_frequency_per_scope_and_combined() -> None:
    """Unique tokens and total postings, per source and summed."""
    tokens = _report()["token_frequency"]
    assert set(tokens) == {"source2", "source3", "all"}
    for scope, expected in EXPECTED_TOKENS.items():
        for key, value in expected.items():
            assert tokens[scope][key] == value, f"{scope}.{key}: {tokens[scope][key]} != {value}"
    # The combined scope is the union of the two sources, not a separate count.
    assert (
        tokens["source2"]["n_token_occurrences"] + tokens["source3"]["n_token_occurrences"]
        == tokens["all"]["n_token_occurrences"]
    )
    # df is per ENTITY, not per occurrence: "acme" appears in 4 target records and
    # in no record twice, so the two numbers agree here and would not if a token
    # were repeated inside one name.
    top = tokens["all"]["top_tokens"]
    assert [(row["token"], row["document_frequency"]) for row in top[:2]] == [
        ("acme", 4),
        ("industries", 3),
    ]
    assert tokens["source2"]["top_tokens"][0]["document_frequency"] == 2
    assert tokens["source3"]["top_tokens"][0]["document_frequency"] == 2
    for scope, entry in tokens.items():
        assert entry["posting_size"]["max"] <= entry["n_entities_in_scope"]


def test_posting_cap_table_is_reported_even_when_nothing_exceeds_it() -> None:
    """The cap table covers every requested cap; the fixture's max df is 4."""
    entry = _report()["token_frequency"]["all"]
    assert [row["cap"] for row in entry["posting_caps"]] == list(abs_.POSTING_CAP_THRESHOLDS)
    assert all(row["n_tokens_over_cap"] == 0 for row in entry["posting_caps"])
    assert entry["posting_size"]["max"] == 4


def test_pair_rarity_separates_reachable_from_unreachable_pairs() -> None:
    """6 differing pairs: only the ones sharing a token can ever be reached."""
    rarity = _report()["pair_rarity"]
    assert rarity["subset_sizes"]["all_true_pairs"] == EXPECTED_RARITY["n_all_true_pairs"]
    assert rarity["subset_sizes"]["name_norm_differs"] == EXPECTED_RARITY["n_name_norm_differs"]
    assert rarity["subset_sizes"]["name_norm_identical"] == EXPECTED_RARITY["n_name_norm_identical"]
    assert (
        rarity["subset_sizes"]["name_norm_differs"]
        + rarity["subset_sizes"]["name_norm_identical"]
        == rarity["subset_sizes"]["all_true_pairs"]
    )
    all_shared = rarity["has_shared_token_at_all"]["all_true_pairs"]
    assert all_shared["n_pairs"] == EXPECTED_RARITY["n_all_with_shared_token"]
    differing = rarity["has_shared_token_at_all"]["name_norm_differs"]
    # P2 and P4 share a token; P8, P10, P11 (and the S1-13 pair with no shared
    # token) do not - six differing pairs, only two of them reachable.
    assert differing["n_pairs"] == 2
    assert _close(differing["pct_of_subset"], 33.3333)
    assert rarity["shared_name_token_histogram"]["all_true_pairs"] == {
        "0": 4, "1": 1, "2": 4, "3": 2,
    }


def test_pair_rarity_usability_and_candidate_volume() -> None:
    """No cap in the shipped grid can change the answer; the volume is the sum."""
    rarity = _report()["pair_rarity"]
    rows = {row["cap"]: row for row in rarity["usability_by_cap"]["all_true_pairs"]}
    for cap in abs_.RARITY_CAP_THRESHOLDS:
        assert rows[cap]["n_pairs_with_usable_token"] == EXPECTED_RARITY["n_usable_cap_1000"]
        assert rows[cap]["n_pairs_missed"] == 4
    volume = rarity["candidate_volume_at_cap"]["all_true_pairs"]
    assert volume["n_pairs"] == 7
    assert volume["estimated_candidates"] == EXPECTED_RARITY["estimated_candidates_cap_1000"]
    assert volume["mean_postings_per_pair"] == 1.0
    assert volume["max_postings_per_pair"] == 1
    # Every usable shared token has df 1, so the rarest-shared distribution is a
    # spike - which is exactly why the census is needed to say anything about
    # volume at scale.
    assert rarity["rarest_shared_token_df"]["all_true_pairs"]["max"] == 1.0


def test_token_hashes_resolve_to_document_frequency() -> None:
    """min_df is the RAREST shared token: for P4 it is 1, not 4 ('acme')."""
    report = _report()
    volume = report["pair_rarity"]["rarest_shared_token_df"]["all_true_pairs"]
    # Shared tokens of P4: acme (df 4), private (1), limited (1). If the reduceat
    # grouping were wrong this would come out as 4 (or 0), not 1.
    assert volume["min"] == 1.0
    assert volume["max"] == 1.0
    # No shared token was missing from the frequency table: every shared token of
    # a true pair is in a matched target record by construction.
    assert report["meta"]["n_unknown_df"] == 0


# ---------------------------------------------------------------------------
# Phase 0.4 census (R2): every source1 entity, matched or not
# ---------------------------------------------------------------------------
def test_census_token_volume() -> None:
    """One estimated posting list per entity, over all 14 of them."""
    census = _report()["candidate_census"]
    assert census["n_source1_entities"] == len(S1)
    token = census["token_signal"]
    assert token["cap"] == abs_.REFERENCE_TOKEN_CAP
    assert token["entities_with_a_candidate"] == {"n_entities": 10, "pct": 71.4286}
    assert token["total_estimated_candidate_rows"] == sum(EXPECTED_CENSUS_ESTIMATES) == 12
    assert _close(token["mean_per_entity"], 0.857143)
    # All entities, zeros included: the distribution macro F0.5 actually feels.
    assert token["estimated_candidates_all_entities"]["n"] == len(S1)
    assert token["estimated_candidates_all_entities"]["max"] == 3.0
    # Reachable entities only: how large a probe is when it happens.
    assert token["estimated_candidates_reachable_entities"]["n"] == 10
    assert token["reachable_entities_p50_p90_p99_max"] == {
        "p50": 1.0, "p90": 1.2, "p99": 2.82, "max": 3,
    }
    assert token["histogram_reachable_entities"] == {
        "1": 9, "2-5": 1, "6-20": 0, "21-100": 0, "101-1000": 0, "1000+": 0,
    }


def test_census_structural_zeros_partition() -> None:
    """No usable token for four entities, for two different reasons."""
    token = _report()["candidate_census"]["token_signal"]
    zero = token["structural_zero_entities"]
    assert zero["n_entities"] == 4
    assert _close(zero["pct"], 28.5714)
    assert zero["by_reason"] == EXPECTED_STRUCTURAL_ZERO
    # The reasons partition the structural zeros exactly - and the analyzer asserts
    # the same identity internally, so this test only has to cover the fixture.
    assert sum(zero["by_reason"].values()) == zero["n_entities"]
    # S1-13 ("freightways", whose only token exists in the corpus as
    # "freightwayss") is a structural zero, and it is NOT a zero-match entity: the
    # census is about reachability, not about the ground truth.
    assert token["by_cap"][0]["n_structural_zero_entities"] == 4
    for row in token["by_cap"]:
        assert row["n_entities_with_a_candidate"] == 10
        assert row["total_estimated_candidate_rows"] == 12
        assert row["mean_postings_per_reached_entity"] == 1.2
        assert row["max_postings_per_reached_entity"] == 3


def test_census_cap_bites_when_set_below_the_fixture_dfs() -> None:
    """At cap 0 every entity becomes a structural zero, via the third reason."""
    census = _patched_report()["candidate_census"]
    token = census["token_signal"]
    assert token["cap"] == 0
    assert token["estimated_candidates_all_entities"]["n"] == len(S1)
    assert token["reachable_entities_p50_p90_p99_max"] == {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0}
    zero = token["structural_zero_entities"]
    assert zero["n_entities"] == len(S1)
    assert zero["by_reason"] == {
        "empty_name_norm": 0,
        "no_token_in_target_corpus": 4,
        "every_token_above_the_cap": 10,
    }
    assert sum(zero["by_reason"].values()) == len(S1)
    # The exact-name signal is unaffected: the two signals fail for different
    # reasons, so one cap cannot zero out reachability.
    assert census["reachability"]["exact_name_only"] == 7
    assert census["reachability"]["neither_signal_reaches"]["n_entities"] == 7


def test_census_candidate_to_truth_ratio() -> None:
    """Candidates per entity against that entity's own true pairs."""
    ratio = _report()["candidate_census"]["token_signal"]["candidate_to_truth_ratio"]
    assert ratio["n_entities_with_analysed_pairs"] == N_ENTITIES_WITH_PAIRS
    assert ratio["n_source1_entities_without_analysed_pairs"] == 4
    assert ratio["n_analysed_true_pairs"] == N_PAIRS
    # 7 candidates over the entities with pairs, 11 true pairs.
    assert _close(ratio["aggregate"], 0.636364)
    # Below one: S1-12 and S1-14 have one pair each and no candidate, S1-13 has two
    # pairs and none. Those entities cannot score, however good the scorer is.
    assert ratio["n_entities_below_one"] == 3
    assert _close(ratio["pct_entities_below_one"], 30.0)
    assert ratio["n_entities_with_zero_candidates"] == 3
    assert ratio["per_entity"]["n"] == N_ENTITIES_WITH_PAIRS
    assert _close(ratio["per_entity"]["mean"], 0.7)
    assert ratio["per_entity"]["max"] == 1.0


def test_census_top_entities_by_estimated_volume() -> None:
    """The largest volumes, named, with the token responsible."""
    top = _report()["candidate_census"]["token_signal"]["top_entities_by_estimated_candidates"]
    assert [row["source1_entity_id"] for row in top] == [
        "S1-11", "S1-1", "S1-2", "S1-3", "S1-4", "S1-5", "S1-6", "S1-7", "S1-8", "S1-9",
    ]
    assert [row["estimated_candidates"] for row in top] == [3] + [1] * 9
    # S1-11 "acme industries": "acme" has df 4 and "industries" df 3, so the
    # rarest usable token is the second one - the estimate follows the rarest, not
    # the first or the most frequent.
    assert top[0]["rarest_usable_token"] == "industries"
    assert top[0]["rarest_usable_token_df"] == 3
    assert top[0]["analysed_true_pairs"] == 0
    assert top[1]["rarest_usable_token"] == "sunrise"
    # Entities with no candidate at all never appear: the table is about volume.
    assert all(row["estimated_candidates"] > 0 for row in top)


def test_exact_name_census_reads_the_real_index() -> None:
    """The census probes the persisted index read-only, and counts what it holds."""
    exact_index = _report()["exact_name_census"]
    assert exact_index["available"] is True
    assert exact_index["n_sources_with_index"] == 2
    assert exact_index["n_source1_entities"] == len(S1)
    assert exact_index["n_entities_with_any_exact_candidate"] == 7
    assert _close(exact_index["pct_entities_with_any_exact_candidate"], 50.0)
    assert exact_index["total_candidate_rows"] == 9

    source2 = exact_index["per_source"]["source2"]
    assert source2["available"] is True
    assert source2["key_field"] == "name_norm"
    assert source2["describe"]["n_entities_indexed"] == 9
    assert source2["describe"]["n_unique_keys"] == 8
    # "acme industries" is in two source2 records: the exact duplicate case.
    assert source2["duplicate_keys"]["n_keys_shared_by_two_or_more_targets"] == 1
    assert source2["duplicate_keys"]["n_postings_in_duplicate_keys"] == 2
    assert _close(source2["duplicate_keys"]["pct_keys_with_duplicates"], 12.5)
    assert source2["posting_size"]["max"] == 2.0
    assert [(row["key"], row["postings"]) for row in source2["largest_posting_lists"]][:2] == [
        ("acme industries", 2),
        ("bluesky exports", 1),
    ]
    # The same lookup the blocker would run, over every S1 entity: 4 of them hit
    # source2 (S1-1, S1-3, S1-8, S1-11) for 5 rows.
    assert source2["source1_query"]["n_entities_with_a_candidate"] == 4
    assert source2["source1_query"]["total_candidate_rows"] == 5

    source3 = exact_index["per_source"]["source3"]
    assert source3["describe"]["n_entities_indexed"] == 6
    assert source3["describe"]["n_unique_keys"] == 6
    assert source3["duplicate_keys"]["n_keys_shared_by_two_or_more_targets"] == 0
    assert source3["source1_query"]["n_entities_with_a_candidate"] == 4
    assert source3["source1_query"]["total_candidate_rows"] == 4


def test_census_exact_signal_per_entity_and_reachability() -> None:
    """The token and exact signals fail for different reasons; the union matters."""
    census = _report()["candidate_census"]
    exact = census["exact_name_signal"]
    assert exact["available"] is True
    assert exact["n_entities_with_a_candidate"] == 7
    assert exact["total_candidate_rows"] == 9
    assert exact["mean_per_entity"] == _round6(9 / len(S1))
    assert exact["histogram_reachable_entities"] == {
        "1": 6, "2-5": 1, "6-20": 0, "21-100": 0, "101-1000": 0, "1000+": 0,
    }
    assert exact["top_entities_by_exact_candidates"][0]["source1_entity_id"] == "S1-11"
    assert exact["top_entities_by_exact_candidates"][0]["exact_candidates"] == 3

    reach = census["reachability"]
    assert reach["both_signals_reach"] == 7
    assert reach["token_only"] == 3       # S1-2, S1-4, S1-9
    assert reach["exact_name_only"] == 0
    neither = reach["neither_signal_reaches"]
    assert neither["n_entities"] == 4     # S1-10, S1-12, S1-13, S1-14
    assert _close(neither["pct"], 28.5714)
    assert neither["n_entities_with_analysed_pairs"] == 3
    assert _close(neither["pct_of_entities_with_analysed_pairs"], 30.0)


# ---------------------------------------------------------------------------
# Phase 0.5
# ---------------------------------------------------------------------------
def test_zero_match_population() -> None:
    """4 of 14 S1 entities have no true match."""
    zero = _report()["zero_match"]
    assert zero["n_zero_match_entities"] == EXPECTED_ZERO_MATCH["n_zero_match_entities"]
    assert _close(zero["pct_of_all_source1"], EXPECTED_ZERO_MATCH["pct_of_all_source1"])
    assert zero["keys_available"] is True


def test_zero_match_candidate_kinds() -> None:
    """exact name_norm / name_key-only / no candidate - the false-positive split."""
    kinds = _report()["zero_match"]["by_candidate_kind"]
    assert kinds["exact_name_norm"]["n_entities"] == EXPECTED_ZERO_MATCH["exact_name_norm_entities"]
    assert _close(kinds["exact_name_norm"]["pct_of_zero_match"], EXPECTED_ZERO_MATCH["exact_name_norm_pct"])
    assert kinds["name_key_only"]["n_entities"] == EXPECTED_ZERO_MATCH["name_key_only_entities"]
    assert _close(kinds["name_key_only"]["pct_of_zero_match"], EXPECTED_ZERO_MATCH["name_key_only_pct"])
    assert kinds["no_candidate_at_all"]["n_entities"] == EXPECTED_ZERO_MATCH["no_candidate_at_all_entities"]
    assert _close(
        kinds["no_candidate_at_all"]["pct_of_zero_match"],
        EXPECTED_ZERO_MATCH["no_candidate_at_all_pct"],
    )
    # The three kinds partition the zero-match population.
    assert (
        kinds["exact_name_norm"]["n_entities"]
        + kinds["name_key_only"]["n_entities"]
        + kinds["no_candidate_at_all"]["n_entities"]
        == EXPECTED_ZERO_MATCH["n_zero_match_entities"]
    )


def test_zero_match_candidate_counts() -> None:
    """Candidate counts per scope, including the entity with candidates in both."""
    zero = _report()["zero_match"]
    assert zero["exact_name_norm"]["n_entities"] == EXPECTED_ZERO_MATCH["exact_name_norm_entities"]
    assert zero["exact_name_norm"]["candidate_count"]["max"] == EXPECTED_ZERO_MATCH["max_candidates"]
    assert zero["name_key_total"]["n_entities"] == 3
    per_source = zero["per_source"]
    assert per_source["source2"]["n_entities"] == EXPECTED_ZERO_MATCH["source2_entities"]
    assert per_source["source3"]["n_entities"] == EXPECTED_ZERO_MATCH["source3_entities"]
    assert per_source["source2"]["candidate_count"]["max"] == 2
    assert per_source["source3"]["candidate_count"]["max"] == 1
    # Buckets cover exactly the entities that have at least one candidate.
    buckets = zero["exact_name_norm"]["buckets"]
    assert sum(buckets.values()) == EXPECTED_ZERO_MATCH["exact_name_norm_entities"]
    assert buckets["1"] == 1 and buckets["2-5"] == 1


def test_zero_match_false_positive_risk() -> None:
    """Every candidate pair here has a matching address, so none is rejected outright."""
    risk = _report()["zero_match"]["false_positive_risk"]
    assert risk["n_candidate_pairs"] == EXPECTED_ZERO_MATCH["n_candidate_pairs"]
    assert risk["n_pairs_both_addresses_present"] == EXPECTED_ZERO_MATCH["n_pairs_both_addresses_present"]
    assert risk["n_pairs_with_zero_address_overlap"] == EXPECTED_ZERO_MATCH["n_pairs_with_zero_address_overlap"]
    assert _close(risk["pct_pairs_with_zero_address_overlap"], 0.0)
    assert risk["n_pairs_with_an_empty_address"] == 0
    # All four candidate pairs sit in the top Jaccard bucket.
    assert risk["address_jaccard_histogram"] == {"(0.9,1]": 4}
    assert risk["n_entities_with_any_address_support"] == EXPECTED_ZERO_MATCH["n_entities_with_any_address_support"]


def test_zero_match_probe_agrees_with_the_real_exact_index() -> None:
    """The probe's counts must equal what ``ExactNameIndex`` proposes.

    Phase 0.5 replicates the exact blocker's semantics with a dictionary lookup
    instead of building the index, and Phase 0.4's census probes the index itself.
    This is the check that the two agree: compare the number of proposed pairs and
    the number of S1 entities receiving candidates, over the SAME population the
    census queries (all S1 entities).
    """
    config_path, _ = _fixture()
    config = load_config(str(config_path))
    total_pairs = 0
    per_entity = None
    all_names = _all_source1_names(config)
    for source in ("source2", "source3"):
        index = ExactNameIndex.load(index_dir_for(config, "train", source, BLOCKER_EXACT_NAME))
        positions, counts = index.lookup_many(all_names)
        assert np.array_equal(positions >= 0, counts > 0)
        total_pairs += int(counts.sum())
        per_entity = counts if per_entity is None else per_entity + counts

    report = _report()
    census = report["exact_name_census"]
    assert total_pairs == census["total_candidate_rows"] == 9
    assert int((per_entity > 0).sum()) == census["n_entities_with_any_exact_candidate"] == 7

    # ...and the zero-match probe sees the same rows among its own population.
    zero = report["zero_match"]
    zero_names = _zero_match_names(config)
    per_entity_zero = np.array([])
    zero_pairs = 0
    for source in ("source2", "source3"):
        index = ExactNameIndex.load(index_dir_for(config, "train", source, BLOCKER_EXACT_NAME))
        _, counts = index.lookup_many(zero_names)
        zero_pairs += int(counts.sum())
        per_entity_zero = counts if not per_entity_zero.size else per_entity_zero + counts
    assert zero_pairs == zero["false_positive_risk"]["n_candidate_pairs"]
    assert int((per_entity_zero > 0).sum()) == zero["exact_name_norm"]["n_entities"]


# ---------------------------------------------------------------------------
# determinism / single source / CLI
# ---------------------------------------------------------------------------
def test_serial_and_parallel_agree() -> None:
    """--workers 1 and --workers 4 (with a tiny chunk) give identical results."""
    serial = _report(workers=1, chunk_pairs=2)
    parallel = _report(workers=4, chunk_pairs=2)
    assert _strip_volatile(serial) == _strip_volatile(parallel)
    assert serial["meta"]["n_true_pairs"] == N_PAIRS


def test_chunk_size_does_not_change_results() -> None:
    """One chunk or many: chunking is an execution detail, not a semantic one."""
    small = _strip_volatile(_report(workers=1, chunk_pairs=1))
    large = _strip_volatile(_report(workers=1, chunk_pairs=1000))
    assert small == large


def test_single_source_run() -> None:
    """--sources source2 analyses, scans and censuses source2 only."""
    config_path, root = _fixture()
    report = _run(config_path, root / "analysis_source2", ["--sources", "source2"])
    assert report["meta"]["sources"] == ["source2"]
    assert report["meta"]["target_rows_per_source"] == {"source2": 9}
    assert report["signal_coverage"]["n_pairs"] == EXPECTED_PER_SOURCE["source2"]["n_pairs"]
    assert set(report["signal_coverage"]["per_source"]) == {"source2"}
    assert set(report["address_overlap"]["slices"]) == {"all", "source2"}
    assert set(report["token_frequency"]) == {"source2", "all"}
    # Only the analysed source's index is read, and the zero-match probe only
    # counts candidates in it.
    assert set(report["exact_name_census"]["per_source"]) == {"source2"}
    assert report["exact_name_census"]["n_sources_with_index"] == 1
    assert report["exact_name_census"]["total_candidate_rows"] == 5
    assert set(report["zero_match"]["per_source"]) == {"source2"}


def test_report_files_and_columns() -> None:
    """All seven outputs are written, and the Markdown mentions all four phases."""
    _, root = _fixture()
    output_dir = root / "analysis_w1_c2"
    for name in (
        "blocking_statistics_report.json",
        "blocking_statistics_report.md",
        "signal_coverage.csv",
        "candidate_census.csv",
        "address_overlap.csv",
        "token_frequency.csv",
        "zero_match_statistics.csv",
    ):
        assert (output_dir / name).is_file(), name
    markdown = (output_dir / "blocking_statistics_report.md").read_text(encoding="utf-8")
    for phase in ("Phase 0.2", "Phase 0.3", "Phase 0.4", "Phase 0.5"):
        assert phase in markdown
    # Measurement only: the report must say so, and must not have chosen a cap.
    assert "Measurement only" in markdown
    assert "irreducible residue" in markdown
    assert "Candidate-volume census over all source1 entities" in markdown
    assert "Exact-name index census (read-only)" in markdown

    coverage_csv = pd.read_csv(output_dir / "signal_coverage.csv")
    assert list(coverage_csv.columns) == ["section", "scope", "key", "metric", "value"]
    sections = set(coverage_csv["section"])
    assert {"signal", "combination", "entity_outcome", "per_source", "sensitivity",
            "residue", "residue_category", "residue_script", "residue_source",
            "coverage_by_category"} <= sections
    assert "country" not in " ".join(sections)
    # The four signals and the residue row are all in the CSV, not only the JSON.
    assert {"exact_name", "rare_token_name", "char_3gram_name", "address_jaccard"} <= set(
        coverage_csv["key"]
    )
    assert "none" in set(coverage_csv["key"])

    census_csv = pd.read_csv(output_dir / "candidate_census.csv")
    assert list(census_csv.columns) == ["section", "scope", "key", "metric", "value"]
    assert {"token_volume", "token_cap", "structural_zero", "ratio", "top_entity",
            "reachability", "exact_volume", "posting_list"} <= set(census_csv["section"])

    token_csv = pd.read_csv(output_dir / "token_frequency.csv")
    assert set(token_csv["section"]) == {"top_token", "posting_cap"}
    assert {"source2", "source3", "all"} <= set(token_csv["scope"])
    zero_csv = pd.read_csv(output_dir / "zero_match_statistics.csv")
    assert set(zero_csv["section"]) >= {"summary", "candidate_kind", "candidate_count"}
    address_csv = pd.read_csv(output_dir / "address_overlap.csv")
    assert set(address_csv["unit"]) <= {"tokens", "ratio", "pairs", "percent"}


def test_limit_pairs_and_no_name_categories() -> None:
    """--limit-pairs narrows the pair phases; the census still covers every entity."""
    config_path, root = _fixture()
    report = _run(
        config_path,
        root / "analysis_limited",
        ["--limit-pairs", "3", "--no-name-categories"],
    )
    assert report["meta"]["n_true_pairs"] == 3
    assert report["meta"]["name_categories"] is False
    assert report["signal_coverage"]["n_pairs"] == 3
    # P1, P2, P3 are all reached by every signal, and none of them is residue.
    assert report["signal_coverage"]["union"]["n_pairs"] == 3
    assert report["signal_coverage"]["residue"]["n_pairs"] == 0
    # The category cross-tabs are reported as unavailable rather than guessed at.
    assert report["signal_coverage"]["residue"]["by_name_difference_category"] == {}
    assert report["signal_coverage"]["residue"]["by_script_relation"] == {}
    assert report["signal_coverage"]["coverage_by_name_difference_category"] == {}
    assert "--no-name-categories" in report["signal_coverage"]["residue"]["note"]
    # Phase 0.5 and the census do not depend on the pair list, so they are intact.
    assert report["zero_match"]["n_zero_match_entities"] == EXPECTED_ZERO_MATCH["n_zero_match_entities"]
    assert (
        report["zero_match"]["false_positive_risk"]["n_candidate_pairs"]
        == EXPECTED_ZERO_MATCH["n_candidate_pairs"]
    )
    assert report["candidate_census"]["n_source1_entities"] == len(S1)
    assert report["candidate_census"]["token_signal"]["total_estimated_candidate_rows"] == 12
    # The ratio's denominator follows the analysed pairs, which --limit-pairs cut.
    ratio = report["candidate_census"]["token_signal"]["candidate_to_truth_ratio"]
    assert ratio["n_analysed_true_pairs"] == 3
    assert ratio["n_entities_with_analysed_pairs"] == 3


def test_rejects_unknown_source_and_test_split() -> None:
    """Bad CLI input fails loudly instead of producing a misleading report."""
    config_path, _ = _fixture()
    for argv in (
        ["--config", str(config_path), "--sources", "source9"],
        ["--config", str(config_path), "--split", "test"],
    ):
        try:
            abs_.main(argv)
        except SystemExit as exc:
            assert "source9" in str(exc) or "test" in str(exc)
        else:  # pragma: no cover - a passing main() here is the failure
            raise AssertionError(f"{argv} should have raised SystemExit")


def test_imports_and_help() -> None:
    """The module imports cleanly and --help documents the required flags."""
    import subprocess

    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "analyze_blocking_statistics.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    for flag in (
        "--workers", "--chunk-pairs", "--split", "--sources", "--config", "--data-root",
        "--limit-pairs", "--no-name-categories",
    ):
        assert flag in result.stdout, flag


# ---------------------------------------------------------------------------
# standalone runner (no pytest required)
# ---------------------------------------------------------------------------
def _main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except AssertionError as exc:
            failures += 1
            print(f"[FAIL] {test.__name__}: {exc}")
        else:
            print(f"[PASS] {test.__name__}")
    print()
    if failures:
        print(f"{failures} of {len(tests)} tests failed")
        return 1
    print(f"all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
