"""Fixture test for ``scripts/analyze_blocking_statistics.py`` (Phase 0.2-0.5).

The fixture is 12 S1 / 7 S2 / 5 S3 rows and 8 true pairs, hand-built so that
every branch of every phase has at least one example and so that every headline
number can be computed by hand:

* country: 6 agreeing pairs, 1 disagreeing pair (US vs Canada), 1 pair with a
  missing country on the target side;
* address: identical addresses, a partial overlap (``3 high st london`` vs
  ``3 high street london`` = Jaccard 0.6), one pair with a missing address, and
  non-overlapping addresses that still share the common tokens ``st``/``tx``;
* names: pairs whose ``name_norm`` is identical (the exact blocker sees them) and
  pairs whose ``name_norm`` differs - including one differing pair with NO shared
  name token, which no token blocker can ever reach;
* tokens: a token with a document frequency above 1 (``acme``, ``industries``) so
  the posting-size distribution is not degenerate;
* zero-match S1 entities: one with an exact ``name_norm`` candidate, one whose
  candidate appears only under ``name_key`` (spacing differs), one with no
  candidate at all, and one with candidates in both target sources.

Nothing here touches the dataset: the fixture is written to a temp directory,
normalized by the real ``scripts/prepare_data.py``, and analysed by the real
analyzer. Runs standalone (``python tests/test_blocking_statistics.py``) and
under pytest.
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
from src.blocking import ExactNameIndex  # noqa: E402
from src.data_loader import iter_prepared, load_config  # noqa: E402
from src.utils import read_json  # noqa: E402

HEADER = ["entity_id", "business_name", "business_address", "country"]

# ---------------------------------------------------------------------------
# The fixture. Keep in sync with the expectations block below - every change
# here invalidates the hand-computed numbers.
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
]
S2 = [
    ("S2-201", "Sunrise Traders", "1 Main St, Austin, TX", "US"),
    ("S2-202", "BlueSky Exports", "2 Dock Rd, Seattle, WA", "Canada"),
    ("S2-203", "Global Tech Solutions", "3 High Street, London", "United Kingdom"),
    ("S2-204", "Acme Industries", "4 MG Road, Mumbai", "India"),
    ("S2-205", "Acme Industries", "4 MG Road, Mumbai", "India"),
    ("S2-206", "Quasar Holdings", "99 Elsewhere, Lyon", "France"),
    ("S2-207", "Northern Lights Trading", "20 Elm St, Dallas, TX", "US"),
]
S3 = [
    ("S3-301", "Acme Private Limited", "4 MG Road, Mumbai", "India"),
    ("S3-302", "Meridian Logistics", "5 Freight Way, Denver, CO", "US"),
    ("S3-303", "Zenith Foods", "1 Rue de Rivoli, Paris", ""),
    ("S3-304", "Acme Industries", "4 MG Road, Mumbai", "India"),
    ("S3-305", "Delta Freight", "9 Depot Ave, Memphis, TN", "US"),
]
GROUND_TRUTH = [
    ("S1-1", "S2-201"),   # country agree, identical name_norm, identical address
    ("S1-2", "S2-202"),   # country DISAGREE (US vs Canada)
    ("S1-3", "S2-203"),   # country agree, address Jaccard 0.6
    ("S1-4", "S3-301"),   # missing address on the S1 side
    ("S1-5", "S3-302"),   # identical everything
    ("S1-6", "S3-303"),   # country missing on the target side
    ("S1-7", "S3-305"),   # identical everything
    ("S1-8", ""),
    ("S1-9", ""),
    ("S1-10", ""),
    ("S1-11", ""),
    ("S1-12", "S2-207"),  # name_norm differs AND shares no name token at all
]

# ---------------------------------------------------------------------------
# Hand-computed expectations. See the module docstring for how to re-derive them.
# ---------------------------------------------------------------------------
EXPECTED_COUNTRY_ALL = {
    "n_true_pairs": 8,
    "n_agree": 6,
    "n_disagree": 1,
    "n_missing_on_either_side": 1,
    "n_missing_target_only": 1,
    "n_missing_source1_only": 0,
    "n_missing_both_sides": 0,
    "pct_agree_of_comparable": 85.7143,
    "pct_disagree_of_comparable": 14.2857,
}
EXPECTED_COUNTRY_BY_SOURCE = {
    "source2": {"n_true_pairs": 4, "n_agree": 3, "n_disagree": 1},
    "source3": {"n_true_pairs": 4, "n_agree": 3, "n_disagree": 0, "n_missing_on_either_side": 1},
}
EXPECTED_ENTITY_LEVEL = {
    "n_source1_entities_with_analysed_pairs": 8,
    "all_pairs_agree": 6,
    "any_pair_disagrees": 1,
    "no_usable_country_information": 1,
}
# country_norm is lowercased by the normalizer, so the report is keyed on the
# normalized values - which is exactly what a blocker would compare.
EXPECTED_COUNTRY_PAIRS = {("us", "us"): 4, ("us", "canada"): 1, ("india", "india"): 1,
                          ("united kingdom", "united kingdom"): 1, ("france", ""): 1}

EXPECTED_ADDRESS = {
    "n_pairs": 8,
    "n_both_addresses_present": 7,
    "n_either_address_missing": 1,
    "n_zero_shared_tokens": 1,
    "n_at_least_1_shared_token": 7,
    "n_at_least_3_shared_tokens": 7,
    "n_jaccard_at_least_0_5": 7,
    "n_jaccard_at_least_0_8": 5,
    "n_overlap_coefficient_at_least_0_8": 6,
    "n_overlap_coefficient_equal_1": 5,
}
# (1,1,0.6,1,0.666667,1,1) - the pair with the missing address is excluded.
EXPECTED_JACCARD_BOTH_PRESENT = {"n": 7, "mean": 0.895238, "p50": 1.0, "max": 1.0}

EXPECTED_TOKENS = {
    "source2": {"n_entities_in_scope": 7, "n_unique_tokens": 14, "n_token_occurrences": 16},
    "source3": {"n_entities_in_scope": 5, "n_unique_tokens": 10, "n_token_occurrences": 11},
    "all": {"n_entities_in_scope": 12, "n_unique_tokens": 22, "n_token_occurrences": 27},
}
EXPECTED_TOP_TOKEN = {"token": "acme", "document_frequency": 4}
EXPECTED_SECOND_TOKEN = {"token": "industries", "document_frequency": 3}

EXPECTED_RARITY = {
    "n_all_true_pairs": 8,
    "n_name_norm_differs": 3,
    "n_name_norm_identical": 5,
    "n_differs_with_shared_token": 2,
    "n_differs_with_usable_token_cap_1000": 2,
    "n_identical_with_usable_token_cap_1000": 5,
    "estimated_candidates_cap_1000": 7,
}

EXPECTED_ZERO_MATCH = {
    "n_zero_match_entities": 4,
    "pct_of_all_source1": 33.3333,
    "exact_name_norm_entities": 2,
    "exact_name_norm_pct": 50.0,
    "name_key_only_entities": 1,
    "name_key_only_pct": 25.0,
    "no_candidate_at_all_entities": 1,
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

    The prepared tables are built once and reused by every test, so a full run
    normalizes 24 rows and analyses them a handful of times - seconds, not
    minutes. The temp directory is removed at interpreter exit unless
    ``ER_TEST_KEEP_FIXTURE`` is set, which is how you inspect a failing run.
    """
    if "root" not in _CACHE:
        root = Path(tempfile.mkdtemp(prefix="er_blocking_"))
        config_path = _build_fixture(root)
        _prepare(config_path)
        _CACHE["root"] = root
        _CACHE["config_path"] = config_path
        print(f"[fixture] {root}")
        if not os.environ.get("ER_TEST_KEEP_FIXTURE"):
            atexit.register(shutil.rmtree, root, ignore_errors=True)
    return _CACHE["config_path"], _CACHE["root"]  # type: ignore[return-value]


def _report(workers: int = 1, chunk_pairs: int = 2) -> dict:
    config_path, root = _fixture()
    return _analyze(config_path, root / f"analysis_w{workers}_c{chunk_pairs}", workers, chunk_pairs)


def _strip_volatile(report: dict) -> dict:
    """The report without anything that legitimately differs between runs."""
    cleaned = {key: value for key, value in report.items() if key != "meta"}
    meta = {
        key: value
        for key, value in report["meta"].items()
        if key not in {"generated_at", "elapsed_minutes", "workers", "chunk_pairs"}
    }
    return {"meta": meta, **cleaned}


def _close(actual: float, expected: float, places: int = 4) -> bool:
    return abs(float(actual) - expected) < 10 ** (-places) + 1e-9


# ---------------------------------------------------------------------------
# Phase 0.2
# ---------------------------------------------------------------------------
def test_country_agreement_all_pairs() -> None:
    """Total / agree / disagree / missing counts, and the comparable-only rates."""
    entry = _report()["country_agreement"]["all"]
    for key, expected in EXPECTED_COUNTRY_ALL.items():
        assert _close(entry[key], expected), f"{key}: {entry[key]} != {expected}"


def test_country_agreement_by_source() -> None:
    """S2 and S3 are reported separately, not merged into one number."""
    per_source = _report()["country_agreement"]["per_source"]
    assert set(per_source) == {"source2", "source3"}
    for source, expected in EXPECTED_COUNTRY_BY_SOURCE.items():
        for key, value in expected.items():
            assert per_source[source][key] == value, f"{source}.{key}: {per_source[source][key]} != {value}"


def test_country_agreement_entity_level() -> None:
    """Per-entity outcomes: all agree, any disagree, no usable country info."""
    entry = _report()["country_agreement"]["entity_level"]
    for key, expected in EXPECTED_ENTITY_LEVEL.items():
        actual = entry[key] if isinstance(entry[key], int) else entry[key]["n_entities"]
        assert actual == expected, f"{key}: {actual} != {expected}"
    # The three outcomes partition the entities that have any analysed pair.
    total = (
        entry["all_pairs_agree"]["n_entities"]
        + entry["any_pair_disagrees"]["n_entities"]
        + entry["no_usable_country_information"]["n_entities"]
        + entry["partially_comparable_never_disagreeing"]["n_entities"]
    )
    assert total == EXPECTED_ENTITY_LEVEL["n_source1_entities_with_analysed_pairs"]


def test_country_pair_frequency_table() -> None:
    """The requested S1-country | target-country | pairs table, with a missing row."""
    table = _report()["country_agreement"]["pair_table"]
    all_rows = [row for row in table if row["target_source"] == "all"]
    assert len(all_rows) == len(EXPECTED_COUNTRY_PAIRS)
    observed = {(row["source1_country"], row["target_country"]): row["n_pairs"] for row in all_rows}
    assert observed == EXPECTED_COUNTRY_PAIRS, observed
    # Percentages are over the 8 analysed pairs.
    us = next(row for row in all_rows if (row["source1_country"], row["target_country"]) == ("us", "us"))
    assert _close(us["pct_of_pairs"], 50.0)
    # Both per-source scopes are present too.
    assert {row["target_source"] for row in table} == {"all", "source2", "source3"}


def test_no_country_is_generic() -> None:
    """The vocabulary is discovered from the data - no country is special-cased."""
    report = _report()
    assert report["meta"]["country_vocabulary_size"] == 5  # US, United Kingdom, India, Canada, France
    countries = {row["source1_country"] for row in report["country_agreement"]["pair_table"]} | {
        row["target_country"] for row in report["country_agreement"]["pair_table"]
    }
    assert "canada" in countries  # a country that appears on one side only


# ---------------------------------------------------------------------------
# Phase 0.3
# ---------------------------------------------------------------------------
def test_address_overlap_counts() -> None:
    """Shared-token thresholds, Jaccard counts and the missing-address pair."""
    entry = _report()["address_overlap"]["slices"]["all"]
    for key, expected in EXPECTED_ADDRESS.items():
        assert entry["counts"][key] == expected, f"{key}: {entry['counts'][key]} != {expected}"


def test_address_jaccard_distribution() -> None:
    """Jaccard is computed over pairs with two addresses, and equals 0.6 and 2/3."""
    entry = _report()["address_overlap"]["slices"]["all"]["jaccard"]
    for key, expected in EXPECTED_JACCARD_BOTH_PRESENT.items():
        assert _close(entry[key], expected), f"{key}: {entry[key]} != {expected}"
    shared = _report()["address_overlap"]["slices"]["all"]["shared_address_tokens"]
    assert shared["min"] == 0.0 and shared["max"] == 5.0


def test_address_overlap_slices_by_source_and_country() -> None:
    """S2/S3 and country-agree/disagree slices exist and are consistent."""
    overlap = _report()["address_overlap"]
    assert set(overlap["slices"]) == {"all", "source2", "source3"}
    groups = overlap["slices"]["all"]["by_group"]
    assert set(groups) == {"both_addresses_present", "country_agree", "country_disagree"}
    # 6 comparable country pairs agree, 1 disagrees, 1 has a missing country.
    assert groups["country_agree"]["counts"]["n_pairs"] == 6
    assert groups["country_disagree"]["counts"]["n_pairs"] == 1
    assert (
        groups["country_agree"]["counts"]["n_pairs"]
        + groups["country_disagree"]["counts"]["n_pairs"]
        == 7
    )
    assert overlap["slices"]["source2"]["counts"]["n_pairs"] == 4
    assert overlap["slices"]["source3"]["counts"]["n_pairs"] == 4


# ---------------------------------------------------------------------------
# Phase 0.4
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


def test_token_document_frequency_is_per_entity_not_per_occurrence() -> None:
    """A token repeated inside one name counts once, and 'acme' spans sources."""
    tokens = _report()["token_frequency"]
    top = tokens["all"]["top_tokens"]
    assert top[0]["token"] == EXPECTED_TOP_TOKEN["token"]
    assert top[0]["document_frequency"] == EXPECTED_TOP_TOKEN["document_frequency"]
    assert top[1]["token"] == EXPECTED_SECOND_TOKEN["token"]
    assert top[1]["document_frequency"] == EXPECTED_SECOND_TOKEN["document_frequency"]
    # "acme" is in 2 S2 records and 2 S3 records: the combined df must be 4, which
    # only holds if each source contributes distinct-entity counts.
    assert tokens["source2"]["top_tokens"][0]["document_frequency"] == 2
    assert tokens["source3"]["top_tokens"][0]["document_frequency"] == 2
    # Document frequency can never exceed the number of entities in scope.
    for scope, entry in tokens.items():
        assert entry["posting_size"]["max"] <= entry["n_entities_in_scope"]


def test_posting_cap_table() -> None:
    """The cap table is reported for every requested cap, even when all are empty."""
    entry = _report()["token_frequency"]["all"]
    caps = [row["cap"] for row in entry["posting_caps"]]
    assert caps == list(abs_.POSTING_CAP_THRESHOLDS)
    # 22 tokens, the largest posting list is 4 - nothing exceeds any cap.
    assert all(row["n_tokens_over_cap"] == 0 for row in entry["posting_caps"])


def test_pair_rarity_separates_reachable_from_unreachable_pairs() -> None:
    """3 differing pairs: 2 share a rare token, 1 shares no token at all."""
    rarity = _report()["pair_rarity"]
    assert rarity["subset_sizes"]["all_true_pairs"] == EXPECTED_RARITY["n_all_true_pairs"]
    assert rarity["subset_sizes"]["name_norm_differs"] == EXPECTED_RARITY["n_name_norm_differs"]
    assert rarity["subset_sizes"]["name_norm_identical"] == EXPECTED_RARITY["n_name_norm_identical"]
    assert (
        rarity["subset_sizes"]["name_norm_differs"]
        + rarity["subset_sizes"]["name_norm_identical"]
        == rarity["subset_sizes"]["all_true_pairs"]
    )
    shared = rarity["has_shared_token_at_all"]["name_norm_differs"]
    assert shared["n_pairs"] == EXPECTED_RARITY["n_differs_with_shared_token"]
    assert _close(shared["pct_of_subset"], 66.6667)


def test_pair_rarity_usability_by_cap() -> None:
    """A cap cannot change which pairs have a shared token: 1 of 3 stays unreachable."""
    rarity = _report()["pair_rarity"]
    rows = {row["cap"]: row for row in rarity["usability_by_cap"]["name_norm_differs"]}
    for cap in abs_.RARITY_CAP_THRESHOLDS:
        assert rows[cap]["n_pairs_with_usable_token"] == EXPECTED_RARITY["n_differs_with_usable_token_cap_1000"]
        assert rows[cap]["n_pairs_missed"] == 1
        assert _close(rows[cap]["pct_of_subset"], 66.6667)
    identical = {row["cap"]: row for row in rarity["usability_by_cap"]["name_norm_identical"]}
    assert identical[1_000]["n_pairs_with_usable_token"] == EXPECTED_RARITY["n_identical_with_usable_token_cap_1000"]
    volume = rarity["candidate_volume_at_cap"]["all_true_pairs"]
    assert volume["estimated_candidates"] == EXPECTED_RARITY["estimated_candidates_cap_1000"]
    # The rarest shared token of the fixtures has df 1, so every pair that shares
    # a token contributes exactly one candidate posting list of size 1.
    assert rarity["rarest_shared_token_df"]["all_true_pairs"]["max"] == 1.0


def test_token_hashes_resolve_to_document_frequency() -> None:
    """min_df is the RAREST shared token: for S1-4 it is 'private'/'limited', not 'acme'."""
    report = _report()
    # S1-4 (acme industries private limited) vs S3-301 (acme private limited):
    # shared = {acme(4), private(1), limited(1)} -> the minimum is 1, not 4.
    # If the reduceat grouping were wrong this would come out as 4 (or 0).
    volume = report["pair_rarity"]["rarest_shared_token_df"]["all_true_pairs"]
    assert volume["max"] == 1.0
    assert report["meta"]["n_unknown_df"] == 0


# ---------------------------------------------------------------------------
# Phase 0.5
# ---------------------------------------------------------------------------
def test_zero_match_population() -> None:
    """4 of 12 S1 entities have no true match."""
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
    instead of building the index. This is the check that "replicates" is true:
    build the real index from the same prepared tables and compare the number of
    proposed pairs and the number of S1 entities receiving candidates.
    """
    config_path, _ = _fixture()
    config = load_config(str(config_path))
    entity_column = config.get("columns", {}).get("entity_id", "entity_id")

    total_pairs = 0
    per_entity = None
    for source, prefix in (("source2", "S2"), ("source3", "S3")):
        index = ExactNameIndex.build(
            iter_prepared(config, "train", source),
            source=source,
            prefix=prefix,
            key_field="name_norm",
            entity_column=entity_column,
        )
        # Look up every zero-match S1 name_norm, exactly as the blocker would.
        zero_names = _zero_match_names(config)
        positions, counts = index.lookup_many(zero_names)
        assert np.array_equal(positions >= 0, counts > 0)
        total_pairs += int(counts.sum())
        per_entity = counts if per_entity is None else per_entity + counts

    zero = _report()["zero_match"]
    assert total_pairs == zero["false_positive_risk"]["n_candidate_pairs"]
    assert int((per_entity > 0).sum()) == zero["exact_name_norm"]["n_entities"]


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


# ---------------------------------------------------------------------------
# determinism / parallelism / CLI
# ---------------------------------------------------------------------------
def test_serial_and_parallel_agree() -> None:
    """--workers 1 and --workers 4 (with a tiny chunk) give byte-identical results."""
    serial = _report(workers=1, chunk_pairs=2)
    parallel = _report(workers=4, chunk_pairs=2)
    assert _strip_volatile(serial) == _strip_volatile(parallel)
    # The fixture is 8 pairs with chunk_pairs=2, so 4 chunks really do run.
    assert serial["meta"]["n_true_pairs"] == 8


def test_chunk_size_does_not_change_results() -> None:
    """One chunk or many: chunking is an execution detail, not a semantic one."""
    small = _strip_volatile(_report(workers=1, chunk_pairs=1))
    large = _strip_volatile(_report(workers=1, chunk_pairs=1000))
    assert small == large


def test_report_is_json_serialisable_and_complete() -> None:
    """All six outputs are written, and the Markdown mentions all four phases."""
    config_path, root = _fixture()
    output_dir = root / "analysis_w1_c2"
    for name in (
        "blocking_statistics_report.json",
        "blocking_statistics_report.md",
        "country_agreement.csv",
        "address_overlap.csv",
        "token_frequency.csv",
        "zero_match_statistics.csv",
    ):
        assert (output_dir / name).is_file(), name
    markdown = (output_dir / "blocking_statistics_report.md").read_text(encoding="utf-8")
    for phase in ("Phase 0.2", "Phase 0.3", "Phase 0.4", "Phase 0.5"):
        assert phase in markdown
    # Measurement only: the report must say so, and must not have picked a cap.
    assert "Measurement only" in markdown

    country_csv = pd.read_csv(output_dir / "country_agreement.csv")
    assert list(country_csv.columns) == [
        "target_source", "source1_country", "target_country", "n_pairs", "pct_of_pairs",
    ]
    token_csv = pd.read_csv(output_dir / "token_frequency.csv")
    assert set(token_csv["section"]) == {"top_token", "posting_cap"}
    assert {"source2", "source3", "all"} <= set(token_csv["scope"])
    zero_csv = pd.read_csv(output_dir / "zero_match_statistics.csv")
    assert set(zero_csv["section"]) >= {"summary", "candidate_kind", "candidate_count"}
    address_csv = pd.read_csv(output_dir / "address_overlap.csv")
    assert set(address_csv["unit"]) <= {"tokens", "ratio", "pairs", "percent"}


def test_limit_pairs_restricts_the_pair_phases() -> None:
    """--limit-pairs narrows the pair statistics but not the corpus scan."""
    config_path, root = _fixture()
    code = abs_.main(
        [
            "--config", str(config_path),
            "--workers", "1",
            "--chunk-pairs", "2",
            "--limit-pairs", "3",
            "--output-dir", str(root / "analysis_limited"),
            "--log-level", "WARNING",
        ]
    )
    assert code == 0
    report = read_json(root / "analysis_limited" / "blocking_statistics_report.json")
    assert report["meta"]["n_true_pairs"] == 3
    assert report["country_agreement"]["all"]["n_true_pairs"] == 3
    # Phase 0.5 does not depend on the pair list, so it must be unchanged.
    assert report["zero_match"]["n_zero_match_entities"] == EXPECTED_ZERO_MATCH["n_zero_match_entities"]
    assert (
        report["zero_match"]["false_positive_risk"]["n_candidate_pairs"]
        == EXPECTED_ZERO_MATCH["n_candidate_pairs"]
    )


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
    for flag in ("--workers", "--chunk-pairs", "--split", "--sources", "--config", "--data-root"):
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
