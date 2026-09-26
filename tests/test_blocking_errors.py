#!/usr/bin/env python
"""Guards for ``scripts/analyze_blocking_errors.py``.

The script decides which blocker to change next, so the guards are about *meaning*,
not plumbing:

* the per-blocker table reports **incremental** contribution over the union, not each
  blocker's standalone count - a blocker that finds 5 pairs the other two also find
  contributes nothing, and a table that shows "5" would mis-rank it;
* the candidate-growth estimate is the exact ``sum_k n_s1(k) * n_target(k)`` identity,
  checked here against hand-computed values on a fixture small enough to count by eye;
* the recall chain keeps A (candidate), B (matcher) and C (macro F0.5) as three
  numbers and never equates them, which is the specific arithmetic error the analysis
  exists to avoid;
* provenance parsing is anchored on the source prefix, so a label that merely contains
  ``token`` cannot be counted as the token blocker;
* ``_example_rows`` maps a ground-truth *row position* through ``s1_codes`` before
  looking a name up - positions and entity codes are different numbers and conflating
  them silently returns the wrong name.

The fixture is chosen so the expectations are countable: 6 S1 entities, 7 true pairs,
5 retrievable, with exactly two unreachable ones - a Devanagari/Latin pair (no
transliteration exists in this repository, so no lexical blocker can reach it) and a
pair with no shared name text but an identical address, which is the interesting case
because the signal is present and simply unused.

Run: ``python tests/test_blocking_errors.py``
"""

from __future__ import annotations

import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import analyze_blocking_errors as abe  # noqa: E402
from scripts import build_indexes, generate_candidates, prepare_data  # noqa: E402
from src.data_loader import candidates_path, load_config  # noqa: E402
from src.utils import read_json  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("test_blocking_errors")

_TEMP_DIRS: list[Path] = []
_CACHE: dict[str, object] = {}

# ---------------------------------------------------------------------------
# The fixture. Every expectation below is derived from these rows by hand.
# ---------------------------------------------------------------------------
HEADER = ["entity_id", "business_name", "business_address", "country"]

S1 = [
    ("S1-1", "Sunrise Traders", "1 Main St, Austin, TX", "US"),
    ("S1-2", "Blue Sky Exports", "2 Dock Rd, Seattle", "US"),
    ("S1-3", "Acme Industries Private Limited", "4 MG Road, Mumbai", "India"),
    ("S1-4", "राम मार्केटिंग", "5 Karol Bagh, Delhi", "India"),
    ("S1-5", "Zenith Foods", "", "France"),
    ("S1-6", "Quasar Holdings", "4 MG Road, Mumbai", "India"),
]
S2 = [
    ("S2-1", "Sunrise Traders", "1 Main St, Austin, TX", "US"),
    ("S2-2", "BlueSky Exports", "2 Dock Rd, Seattle", "US"),
    ("S2-3", "Zenith Foods", "", "France"),
    ("S2-4", "Cascade Textiles", "88 Harbour Way, Reno", "US"),
]
S3 = [
    ("S3-1", "Acme Industries", "4 MG Road, Mumbai", "India"),
    ("S3-2", "Acme Private Limited", "4 MG Road, Mumbai", "India"),
    ("S3-3", "Ram Marketing", "5 Karol Bagh, Delhi", "India"),
    ("S3-4", "Meridian Logistics", "4 MG Road, Mumbai", "India"),
]
GROUND_TRUTH = [
    ("S1-1", "S2-1"),
    ("S1-2", "S2-2"),
    ("S1-3", "S3-1,S3-2"),
    ("S1-4", "S3-3"),
    ("S1-5", "S2-3"),
    ("S1-6", "S3-4"),
]

N_TRUE_PAIRS = 7
N_ENTITIES = 6
RETRIEVED_PAIRS = 5
MISSED_PAIRS = {("S1-4", "S3-3"), ("S1-6", "S3-4")}


def _write_tsv(path: Path, rows, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def _fixture() -> tuple[Path, Path]:
    """Build the fixture once per process; returns ``(config_path, candidates_path)``.

    Every path the config names explicitly is redirected into the temp root. This is
    load-bearing: ``config.yaml`` sets ``prepared_dir``/``index_dir``/``candidates_dir``
    to explicit repo-relative paths, and an explicit path resolves against the
    repository root rather than against ``work_dir`` - so redirecting only ``work_dir``
    writes candidate files into the real ``outputs/`` tree.
    """
    if "config_path" in _CACHE:
        return _CACHE["config_path"], _CACHE["candidates_path"]  # type: ignore[return-value]

    root = Path(tempfile.mkdtemp(prefix="er_blocking_errors_"))
    _TEMP_DIRS.append(root)
    data = root / "data"
    _write_tsv(data / "train_source1.tsv", S1, HEADER)
    _write_tsv(data / "train_source2.tsv", S2, HEADER)
    _write_tsv(data / "train_source3.tsv", S3, HEADER)
    _write_tsv(data / "train_ground_truth.tsv", GROUND_TRUTH, ["source1_entity_id", "matched_entity_ids"])

    work = root / "work"
    replacements = {
        '  data_root: "train"': f'  data_root: "{data.as_posix()}"',
        '  work_dir: "outputs"': f'  work_dir: "{work.as_posix()}"',
        '  prepared_dir: "outputs/prepared"': f'  prepared_dir: "{(work / "prepared").as_posix()}"',
        '  index_dir: "outputs/indexes"': f'  index_dir: "{(work / "indexes").as_posix()}"',
        '  candidates_dir: "outputs/candidates"': f'  candidates_dir: "{(work / "candidates").as_posix()}"',
        '  log_dir: "logs"': f'  log_dir: "{(root / "logs").as_posix()}"',
        "  chunksize: 500000": "  chunksize: 3",  # force the streaming path over chunks
    }
    source = (REPO / "configs" / "config.yaml").read_text(encoding="utf-8")
    for old, new in replacements.items():
        assert old in source, f"config line not found: {old!r}"
        source = source.replace(old, new)
    config_path = root / "config.yaml"
    config_path.write_text(source, encoding="utf-8")

    assert prepare_data.main(
        ["--config", str(config_path), "--splits", "train", "--sources", "source1,source2,source3",
         "--overwrite", "--log-level", "CRITICAL"]
    ) == 0
    assert build_indexes.main(
        ["--config", str(config_path), "--split", "train", "--sources", "source2,source3",
         "--blockers", "exact_name,token,char_ngram", "--overwrite", "--log-level", "CRITICAL"]
    ) == 0
    # --workers 1 keeps Jaccard verification in-process: a process pool under a test
    # runner re-imports __main__ and dies, and a single worker is deterministic anyway.
    assert generate_candidates.main(
        ["--config", str(config_path), "--split", "train", "--name", "fixture",
         "--workers", "1", "--log-level", "CRITICAL"]
    ) == 0

    config = load_config(str(config_path))
    candidates = candidates_path(config, "fixture")
    assert candidates.is_file(), candidates
    _CACHE["config_path"] = config_path
    _CACHE["candidates_path"] = candidates
    _CACHE["config"] = config
    return config_path, candidates


def _analyze(extra: list[str] | None = None, tag: str = "default") -> dict:
    """Run the analysis on the fixture; returns the parsed JSON report."""
    if tag in _CACHE:
        return _CACHE[tag]  # type: ignore[return-value]
    config_path, candidates = _fixture()
    output_dir = Path(_TEMP_DIRS[0]) / f"analysis_{tag}"
    argv = [
        "--config", str(config_path),
        "--candidates", str(candidates),
        "--output-dir", str(output_dir),
        "--split", "all",
        "--split-candidates", "train",
        "--log-level", "CRITICAL",
        *(extra or []),
    ]
    code = abe.main(argv)
    assert code == 0, code
    report = read_json(output_dir / "blocking_recall_report.json")
    report["_output_dir"] = str(output_dir)
    report["_candidates"] = str(candidates)
    _CACHE[tag] = report
    return report


def _proposal(report: dict, label: str) -> dict:
    for row in report["proposals"]:
        if row["proposal"] == label:
            return row
    raise AssertionError(f"proposal {label!r} missing from the report")


# ---------------------------------------------------------------------------
# provenance parsing
# ---------------------------------------------------------------------------
def test_provenance_flags_are_anchored_on_the_source_prefix():
    """A label that merely contains ``token`` must not count as the token blocker.

    The union writes ``source2:token``; matching bare ``token`` would also fire on a
    future label like ``source2:token_signature`` and quietly inflate that blocker.
    """
    series = pd.Series(["source2:exact_name,source2:token", "source3:char_ngram", "source2:xtoken"])
    flags = abe.provenance_flags(series)
    assert flags["exact_name"].tolist() == [True, False, False]
    assert flags["token"].tolist() == [True, False, False]
    assert flags["char_ngram"].tolist() == [False, True, False]


def test_combination_labels_cover_the_seven_subsets():
    assert abe.combination_label(1) == "exact_name"
    assert abe.combination_label(2) == "token"
    assert abe.combination_label(4) == "char_ngram"
    assert abe.combination_label(3) == "exact_name+token"
    assert abe.combination_label(5) == "exact_name+char_ngram"
    assert abe.combination_label(7) == "all_three"
    assert abe.combination_label(0) == "none"
    assert len(abe.COMBINATION_ORDER) == 7


# ---------------------------------------------------------------------------
# pair diagnostics
# ---------------------------------------------------------------------------
def _diagnose(name_a: str, name_b: str, key_a: str = "", key_b: str = "", **rest):
    return abe.diagnose_pair(
        name_a, name_b,
        key_a or name_a.replace(" ", ""), key_b or name_b.replace(" ", ""),
        rest.get("address_a", ""), rest.get("address_b", ""),
        rest.get("country_a", ""), rest.get("country_b", ""),
    )


def test_diagnose_pair_names_the_intended_failure_modes():
    assert _diagnose("", "acme").get("primary_cause") == "empty_normalized_name"
    # identical names that blocking still missed is a defect, not a benign bucket
    assert _diagnose("acme ltd", "acme ltd")["primary_cause"] == "DEFECT_identical_name_not_retrieved"
    # Devanagari vs Latin: no transliteration exists, so no lexical blocker can reach it
    assert _diagnose("राम मार्केटिंग", "ram marketing")["primary_cause"] == "cross_script_no_transliteration"
    assert _diagnose("sunrise traders", "traders sunrise")["primary_cause"] == "token_reorder_only"
    assert _diagnose("acme traders", "acme traders pvt")["primary_cause"] == "token_insertion_or_deletion"
    assert _diagnose("acme pvt ltd", "acme limited")["primary_cause"] == "legal_suffix_variation"
    assert _diagnose("international business machines", "ibm")["primary_cause"] == "abbreviation_initials"
    # "traders" is a legal-form token, so a single-token short name would be a *subset*
    # and report token insertion; the prefix case needs the token split to differ too
    assert _diagnose("acme traders", "acmetraderspvtltd")["primary_cause"] == "prefix_truncation"
    assert _diagnose("acme traders", "acme logistics")["primary_cause"] == "shares_token_but_not_retrieved"
    # 2 of 4 trigrams shared -> 0.5, above the 0.3 cut-off, and no shared token
    above = _diagnose("alpha", "alphb")
    assert above["primary_cause"] == "char_overlap_above_threshold_but_not_retrieved"
    assert abs(above["trigram_jaccard"] - 0.5) < 1e-9
    assert _diagnose("abcdef", "abcxyz")["primary_cause"] == "char_overlap_below_threshold"
    assert _diagnose("abcdef", "uvwxyz")["primary_cause"] == "no_lexical_overlap"


def test_diagnose_pair_reports_address_and_country_independently():
    """Address and country are the signal blocking does not use; they must be visible."""
    row = _diagnose(
        "quasar holdings", "meridian logistics",
        address_a="4 mg road mumbai", address_b="4 mg road mumbai",
        country_a="india", country_b="india",
    )
    assert row["primary_cause"] == "no_lexical_overlap"
    assert row["token_jaccard"] == 0.0
    assert row["trigram_jaccard"] == 0.0
    assert row["address_jaccard"] == 1.0
    assert row["country_equal"] is True
    assert row["address_both_blank"] is False


def test_diagnose_pair_flags_blank_addresses_and_countries():
    row = _diagnose("zenith foods", "zenith foods", address_a="", address_b="", country_a="", country_b="")
    assert row["address_both_blank"] is True
    assert row["address_a_blank"] is True
    assert row["address_jaccard"] == 0.0
    assert row["country_both_blank"] is True
    assert row["country_equal"] is False
    assert abs(row["token_jaccard"] - 1.0) < 1e-9


def test_jaccard_and_suffix_helpers_are_safe_on_empty_input():
    assert abe._jaccard(frozenset(), frozenset({"a"})) == 0.0
    assert abe._jaccard(frozenset({"a"}), frozenset({"a", "b"})) == 0.5
    # stripping must never reduce a name to nothing, or every legal-form-only name
    # would collide with every other legal-form-only name
    assert abe.strip_legal_suffixes(("pvt", "ltd")) == ("pvt", "ltd")
    assert abe.strip_legal_suffixes(("acme", "pvt", "ltd")) == ("acme",)
    assert abe.strip_legal_suffixes(()) == ()


def test_summarise_diagnostics_shares_sum_to_one():
    rows = [
        _diagnose("acme pvt ltd", "acme limited"),
        _diagnose("acme traders", "traders acme"),
        _diagnose("sunrise traders", "sunrise traders"),
    ]
    summary = abe.summarise_diagnostics(rows)
    assert summary["n_pairs"] == 3
    assert abs(sum(summary["primary_cause_share"].values()) - 1.0) < 1e-12
    assert sum(summary["primary_cause"].values()) == 3
    assert 0.0 <= summary["token_jaccard"]["mean"] <= 1.0
    assert 0.0 <= summary["name_equal"] <= 1.0


def test_summarise_diagnostics_handles_no_rows():
    summary = abe.summarise_diagnostics([])
    assert summary == {"n_pairs": 0}


# ---------------------------------------------------------------------------
# the key-intersection machinery
# ---------------------------------------------------------------------------
def _keys_for(names: dict[int, str], splitter=str.split):
    """Build a per-entity key function over a ``{code: name}`` mapping."""
    table = {code: (name, name.replace(" ", ""), "", "") for code, name in names.items()}

    def key_fn(values):
        return tuple(values[0].split())

    return table, key_fn


def test_collect_entity_keys_offsets_and_hashes_agree():
    table, key_fn = _keys_for({3: "alpha beta", 1: "gamma", 2: ""})
    codes, offsets, hashes = abe.collect_entity_keys(table, key_fn)
    assert codes.tolist() == [1, 2, 3]
    assert offsets.tolist() == [0, 1, 1, 3]
    assert len(hashes) == offsets[-1] == 3
    # the offsets must address the right keys, not merely the right count
    assert hashes[0] == hash("gamma")
    assert set(hashes[1:].tolist()) == {hash("alpha"), hash("beta")}


def test_collect_entity_keys_on_an_empty_mapping():
    codes, offsets, hashes = abe.collect_entity_keys({}, str.split)
    assert len(codes) == 0
    assert offsets.tolist() == [0]
    assert len(hashes) == 0


def test_build_s1_membership_excludes_entities_absent_from_ground_truth():
    """A key shared with an entity that has no ground-truth row must not count."""
    # entity 0 carries "shared"; entity 1 (absent from the ground truth, position -1)
    # carries "orphan"
    offsets = np.array([0, 1, 2], dtype=np.int64)
    hashes = np.array([hash("shared"), hash("orphan")], dtype=np.int64)
    positions = np.array([0, -1], dtype=np.int64)
    unique, combined = abe.build_s1_membership(offsets, hashes, positions, n_s1=2)
    assert len(unique) == 2  # both key *values* survive; only the membership is filtered
    assert len(combined) == 1
    # the surviving membership must address position 0, and position 1 must be absent
    assert set((combined % 2).tolist()) == {0}
    shared_id = int(np.searchsorted(unique, hash("shared")))
    assert combined.tolist() == [shared_id * 2 + 0]


def test_pairs_with_shared_key_recovers_exactly_the_pairs_that_share_a_key():
    # S1 side: entity position 0 has {alpha, beta}, position 1 has {gamma}
    s1_offsets = np.array([0, 2, 3], dtype=np.int64)
    s1_hashes = np.array([hash("alpha"), hash("beta"), hash("gamma")], dtype=np.int64)
    s1_positions = np.array([0, 1], dtype=np.int64)
    unique, combined = abe.build_s1_membership(s1_offsets, s1_hashes, s1_positions, n_s1=2)

    # targets: code 20 has {alpha}, 21 has {delta}, 22 has {}
    target_codes = np.array([20, 21, 22], dtype=np.int64)
    target_offsets = np.array([0, 1, 2, 2], dtype=np.int64)
    target_hashes = np.array([hash("alpha"), hash("delta")], dtype=np.int64)

    # pairs: (pos 0 -> 20) shares alpha; (pos 0 -> 21) shares nothing;
    # (pos 1 -> 20) shares nothing; (pos 0 -> 22) target has no keys;
    # (pos 1 -> 99) target does not exist at all
    pair_positions = np.array([0, 0, 1, 0, 1], dtype=np.int64)
    pair_targets = np.array([20, 21, 20, 22, 99], dtype=np.int64)
    recovered = abe.pairs_with_shared_key(
        pair_positions, pair_targets, target_codes, target_offsets, target_hashes, unique, combined, n_s1=2
    )
    assert recovered.tolist() == [True, False, False, False, False]


def test_pairs_with_shared_key_is_chunk_size_invariant():
    """Chunking is a memory decision and must never change an answer."""
    rng = np.random.default_rng(7)
    n_s1, n_target = 40, 25
    s1_keys = [f"k{index % 7}" for index in range(n_s1)]
    s1_hashes = np.array([hash(key) for key in s1_keys], dtype=np.int64)
    s1_offsets = np.arange(n_s1 + 1, dtype=np.int64)
    s1_positions = np.arange(n_s1, dtype=np.int64)
    unique, combined = abe.build_s1_membership(s1_offsets, s1_hashes, s1_positions, n_s1=n_s1)

    target_keys = [f"k{index % 5}" for index in range(n_target)]
    target_codes = np.arange(100, 100 + n_target, dtype=np.int64)
    target_offsets = np.arange(n_target + 1, dtype=np.int64)
    target_hashes = np.array([hash(key) for key in target_keys], dtype=np.int64)

    count = 500
    pair_positions = rng.integers(0, n_s1, count).astype(np.int64)
    pair_targets = rng.integers(100, 100 + n_target, count).astype(np.int64)

    reference = abe.pairs_with_shared_key(
        pair_positions, pair_targets, target_codes, target_offsets, target_hashes, unique, combined, n_s1=n_s1, chunk=count
    )
    for chunk in (1, 3, 7, 64):
        again = abe.pairs_with_shared_key(
            pair_positions, pair_targets, target_codes, target_offsets, target_hashes, unique, combined,
            n_s1=n_s1, chunk=chunk,
        )
        assert again.tolist() == reference.tolist(), chunk
    assert reference.any(), "the fixture must actually match something"


def test_pairs_with_shared_key_returns_nothing_without_s1_keys():
    recovered = abe.pairs_with_shared_key(
        np.array([0], dtype=np.int64), np.array([20], dtype=np.int64),
        np.array([20], dtype=np.int64), np.array([0, 1], dtype=np.int64),
        np.array([hash("alpha")], dtype=np.int64),
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), n_s1=1,
    )
    assert recovered.tolist() == [False]


# ---------------------------------------------------------------------------
# the metric chain
# ---------------------------------------------------------------------------
def test_metric_chain_never_equates_candidate_recall_with_f05():
    """The user's explicit constraint: A, B and C are three different numbers."""
    chain = abe.metric_chain(
        candidate_pair_recall=0.57,
        candidate_macro_recall=0.52,
        proposal_macro_recall=0.70,
        matcher_retention=0.90,
        v1_pair_recall=0.570374,
        v1_macro_f05=0.67111,
    )
    assert chain["A_candidate_pair_recall"] == 0.57
    assert chain["B_proposed_macro_entity_recall"] == 0.70
    assert abs(chain["B_matcher_macro_recall_at_fixed_retention"] - 0.63) < 1e-12
    # C is a ceiling on macro F0.5, and the *uplift* is far below the naive reading
    # "recall rose 0.52 -> 0.70, so F0.5 rose 0.18": 0.895 - 0.815 = 0.080
    assert abs(chain["C_f05_ceiling_at_fixed_retention"] - (1.25 * 0.63 / 0.88)) < 1e-12
    assert chain["C_f05_ceiling_uplift"] > 0.0
    assert chain["C_f05_ceiling_uplift"] < 0.18 / 2
    assert chain["A_to_C_is_not_an_equality"].startswith("candidate recall is an upper bound")
    assert "not" in chain["A_to_C_is_not_an_equality"].lower()
    assert "regenerating features" in chain["caveat"]


def test_metric_chain_omits_proposal_stage_when_absent():
    chain = abe.metric_chain(0.57, 0.52, None, 0.9, None, None)
    assert "B_proposed_macro_entity_recall" not in chain
    assert "C_f05_ceiling_at_fixed_retention" not in chain


def test_f05_ceiling_is_monotonic_and_capped():
    assert abe._f05_ceiling(0.0) == 0.0
    assert abe._f05_ceiling(1.0) == 1.0
    assert abe._f05_ceiling(0.5) < abe._f05_ceiling(0.8)
    # at beta=0.5 a recall of 0.5 with perfect precision can only reach 0.833
    assert abs(abe._f05_ceiling(0.5) - (1.25 * 0.5 / 0.75)) < 1e-12


# ---------------------------------------------------------------------------
# synthetic tables
# ---------------------------------------------------------------------------
def _synthetic_scan() -> dict:
    """A hand-built scan: 3 entities, 4 true pairs, 3 recovered.

    entity 0: 2 true pairs (both recovered) and 3 candidate rows
    entity 1: 1 true pair (missed) and no candidate rows
    entity 2: 1 true pair (recovered) and 1 candidate row

    The four rows are the three recovered true pairs plus one false positive, so every
    row-level count here is derivable from the pair-level ones - the same invariant the
    real scan maintains:
        row A -> pair 0, entity 0, provenance all three   (bitmask 7)
        row B -> pair 1, entity 0, provenance token+char  (bitmask 6)
        row C -> pair 3, entity 2, provenance char only   (bitmask 4)
        row D -> not a true pair, entity 0, exact only    (bitmask 1)
    """
    lengths = np.array([2, 1, 1], dtype=np.int64)
    recovered = np.array([True, True, False, True])
    recovered_by_blocker = {
        "exact_name": np.array([True, False, False, False]),
        "token": np.array([True, True, False, False]),
        "char_ngram": np.array([True, True, False, True]),
    }
    pair_combination = np.array([7, 6, 0, 4], dtype=np.uint8)
    combination_counts = np.bincount(pair_combination, minlength=8)
    row_combination_by_entity = np.zeros((8, 3), dtype=np.int64)
    row_combination_by_entity[1, 0] = 1
    row_combination_by_entity[4, 2] = 1
    row_combination_by_entity[6, 0] = 1
    row_combination_by_entity[7, 0] = 1
    return {
        "rows_seen": 4,
        "unknown_s1_rows": 0,
        "unlabelled_rows": 0,
        "evidence_rows": 0,
        "flags_total": {"exact_name": 2, "token": 2, "char_ngram": 3},
        "flags_by_source": {label: {2: 0, 3: 0} for label in recovered_by_blocker},
        "rows_by_entity": np.array([3, 0, 1], dtype=np.int64),
        "rows_by_entity_blocker": {
            "exact_name": np.array([2, 0, 0], dtype=np.int64),
            "token": np.array([2, 0, 0], dtype=np.int64),
            "char_ngram": np.array([2, 0, 1], dtype=np.int64),
        },
        "recovered": recovered,
        "recovered_by_blocker": recovered_by_blocker,
        "combination_counts": combination_counts,
        "combination_counts_by_source": {2: combination_counts.copy(), 3: np.zeros(8, dtype=np.int64)},
        "row_combination_counts": row_combination_by_entity.sum(axis=1),
        "pair_combination": pair_combination,
        "row_combination_by_entity": row_combination_by_entity,
        "lengths": lengths,
    }


def test_blocker_table_reports_incremental_contribution_not_standalone_counts():
    scan = _synthetic_scan()
    rows = {row["blocker"]: row for row in abe.blocker_table(scan, np.ones(3, dtype=bool), 4)}
    # char_ngram finds all 3 recovered pairs; token finds 2; exact finds 1
    assert rows["char_ngram"]["true_pairs_recovered"] == 3
    assert rows["token"]["true_pairs_recovered"] == 2
    assert rows["exact_name"]["true_pairs_recovered"] == 1
    # but removing exact_name or token loses nothing: char_ngram covers every pair
    assert rows["exact_name"]["pairs_union_would_lose_without_it"] == 0
    assert rows["token"]["pairs_union_would_lose_without_it"] == 0
    # char_ngram is the only blocker holding up pair 3, so it is the only one whose
    # removal costs recall - one pair, not the three it retrieves on its own
    assert rows["char_ngram"]["pairs_union_would_lose_without_it"] == 1
    assert rows["char_ngram"]["pairs_only_this_blocker_finds"] == 1
    # rows where only this blocker fires, from the row-level combination counts
    assert rows["exact_name"]["rows_where_only_this_blocker_fires"] == 1
    assert rows["token"]["rows_where_only_this_blocker_fires"] == 0
    assert rows["char_ngram"]["rows_where_only_this_blocker_fires"] == 1
    # three of the four candidate rows carry char_ngram
    assert rows["char_ngram"]["candidate_rows"] == 3
    assert abs(rows["char_ngram"]["share_of_candidate_rows"] - 0.75) < 1e-12


def test_blocker_table_recall_is_relative_to_true_pairs_in_scope():
    scan = _synthetic_scan()
    rows = {row["blocker"]: row for row in abe.blocker_table(scan, np.ones(3, dtype=bool), 4)}
    assert abs(rows["char_ngram"]["true_pair_recall"] - 3 / 4) < 1e-12
    # exact_name proposes 2 rows and recovers 1 pair, so 0.5 - precision is pairs per row
    assert abs(rows["exact_name"]["candidate_precision"] - 0.5) < 1e-12
    assert abs(rows["char_ngram"]["candidate_precision"] - 1.0) < 1e-12
    # dropping entity 2 shrinks the denominator AND the numerator, together
    masked = {row["blocker"]: row for row in abe.blocker_table(scan, np.array([True, True, False]), 4)}
    assert abs(masked["char_ngram"]["true_pair_recall"] - 2 / 3) < 1e-12
    assert masked["char_ngram"]["true_pairs_recovered"] == 2
    assert masked["char_ngram"]["candidate_rows"] == 2


def test_tables_never_exceed_recall_one_under_any_entity_mask():
    """Guards the mixed-scope bug: a denominator that ignores the mask inflates recall.

    The scan's recovery arrays are per pair while the mask is per entity. Slicing one
    with the other either raises (2.2M entities against 7.6M pairs) or, if the counts
    happen to agree, divides a masked numerator by an unmasked denominator. Both are
    caught here, on every mask shape, for both tables.
    """
    scan = _synthetic_scan()
    probes = [
        np.ones(3, dtype=bool),
        np.array([True, True, False]),
        np.array([True, False, True]),
        np.array([False, True, True]),
        np.array([True, False, False]),
    ]
    for probe in probes:
        for row in abe.blocker_table(scan, probe, 4):
            assert 0.0 <= row["true_pair_recall"] <= 1.0, (probe, row)
            assert row["true_pairs_recovered"] <= 4, (probe, row)
        for row in abe.combination_table(scan, probe):
            assert 0.0 <= row["true_pair_recall"] <= 1.0, (probe, row)
        macro = abe.entity_accounting(scan, probe)["candidate_macro_entity_recall"]
        assert 0.0 <= macro <= 1.0, (probe, macro)


def test_entity_accounting_buckets_are_exhaustive_and_disjoint():
    scan = _synthetic_scan()
    entity = abe.entity_accounting(scan, np.ones(3, dtype=bool))
    assert entity["entities_scored"] == 3
    assert entity["entities_with_true_matches"] == 3
    assert entity["entities_with_no_true_match"] == 0
    # entity 0 is 2/2 -> full; entity 1 is 0/1 -> zero; entity 2 is 1/1 -> full
    assert entity["entities_fully_covered"] == 2
    assert entity["entities_with_zero_recovered"] == 1
    assert entity["entities_partially_covered"] == 0
    assert entity["entities_with_zero_candidates"] == 1
    assert entity["entities_with_zero_candidates_and_true_matches"] == 1
    # macro recall is the per-entity mean: (1 + 0 + 1) / 3
    assert abs(entity["candidate_macro_entity_recall"] - 2 / 3) < 1e-12
    assert abs(entity["candidate_macro_f05_ceiling"] - abe._f05_ceiling(2 / 3)) < 1e-12
    assert entity["missed_pairs_per_entity_max"] == 1


def test_entity_accounting_counts_rows_spent_on_entities_with_no_true_match():
    scan = _synthetic_scan()
    # entity 2 has no true match at all, yet attracts 7 candidate rows - pure precision
    # cost. Lengths must stay consistent with the pair arrays: 3 pairs over entities 0-1.
    scan["lengths"] = np.array([2, 1, 0], dtype=np.int64)
    scan["recovered"] = np.array([True, True, False])
    scan["rows_by_entity"] = np.array([3, 0, 7], dtype=np.int64)
    entity = abe.entity_accounting(scan, np.ones(3, dtype=bool))
    assert entity["entities_with_no_true_match"] == 1
    assert entity["entities_with_true_matches"] == 2
    assert entity["candidate_rows_on_no_match_entities"] == 7
    assert entity["missed_pairs_per_entity_max"] == 1


def test_combination_table_splits_recall_by_provenance_mix():
    scan = _synthetic_scan()
    rows = {row["combination"]: row for row in abe.combination_table(scan, np.ones(3, dtype=bool))}
    assert rows["all_three"]["true_pairs_recovered"] == 1
    assert rows["token+char_ngram"]["true_pairs_recovered"] == 1
    assert rows["char_ngram"]["true_pairs_recovered"] == 1
    # no pair was recovered by exact_name alone, even though a row fired exact_name alone
    assert rows["exact_name"]["true_pairs_recovered"] == 0
    assert sum(row["true_pairs_recovered"] for row in rows.values()) == 3
    # the exact-only row is real volume, so it must still appear in the row column
    assert rows["exact_name"]["candidate_rows"] == 1


# ---------------------------------------------------------------------------
# end to end on the real pipeline
# ---------------------------------------------------------------------------
def test_fixture_is_a_prefix_of_truth_as_designed():
    """Guard the fixture itself: if this drifts, every number below is meaningless."""
    _, candidates = _fixture()
    frame = pd.read_csv(candidates, sep="\t", dtype=str, keep_default_na=False)
    found = {(row.source1_entity_id, row.matched_entity_id) for row in frame.itertuples(index=False)}
    expected = {(s1, t) for s1, matches in GROUND_TRUTH for t in matches.split(",")}
    assert len(frame) == len(found), "the union emitted a duplicate pair"
    assert found <= expected, f"a candidate is not a true pair: {found - expected}"
    assert expected - found == MISSED_PAIRS, expected - found


def test_end_to_end_baseline_matches_the_hand_count():
    report = _analyze()
    official = report["official_metrics"]
    assert official["n_true_pairs"] == N_TRUE_PAIRS
    assert official["true_pairs_retrieved"] == RETRIEVED_PAIRS
    assert abs(official["blocking_recall_pair"] - RETRIEVED_PAIRS / N_TRUE_PAIRS) < 1e-12
    assert official["n_s1_entities"] == N_ENTITIES
    # no false positives on this fixture, unlike the 1.3% precision of the real corpus
    assert official["candidate_precision"] == 1.0
    entity = report["entity"]
    assert entity["entities_scored"] == N_ENTITIES
    assert entity["entities_fully_covered"] == 4  # S1-1, S1-2, S1-3, S1-5
    assert entity["entities_with_zero_recovered"] == 2  # S1-4 cross-script, S1-6 no overlap
    assert abs(entity["candidate_macro_entity_recall"] - 4 / 6) < 1e-12


def test_end_to_end_blocker_table_identifies_char_as_the_only_irreplaceable_blocker():
    """Hand-verified: every retrieved pair carries char_ngram provenance.

    exact_name and token each retrieve fewer pairs, and every pair they retrieve is
    also retrieved by char_ngram - so the union loses nothing when either is removed.
    This is exactly the distinction the table exists to make visible.
    """
    report = _analyze()
    rows = {row["blocker"]: row for row in report["blockers"]}
    assert rows["exact_name"]["candidate_rows"] == 2
    assert rows["token"]["candidate_rows"] == 3
    assert rows["char_ngram"]["candidate_rows"] == 5
    assert rows["exact_name"]["true_pairs_recovered"] == 2
    assert rows["token"]["true_pairs_recovered"] == 3
    assert rows["char_ngram"]["true_pairs_recovered"] == 5
    assert rows["exact_name"]["pairs_union_would_lose_without_it"] == 0
    assert rows["token"]["pairs_union_would_lose_without_it"] == 0
    # only the two char-only pairs hang on char_ngram, and they are the only rows whose
    # provenance is char_ngram alone: one blocker, two pairs, two rows
    assert rows["char_ngram"]["pairs_union_would_lose_without_it"] == 2
    assert rows["char_ngram"]["rows_where_only_this_blocker_fires"] == 2
    combinations = {row["combination"]: row for row in report["combinations"]}
    assert combinations["all_three"]["true_pairs_recovered"] == 2
    assert combinations["char_ngram"]["true_pairs_recovered"] == 2
    assert combinations["token+char_ngram"]["true_pairs_recovered"] == 1


def test_end_to_end_missed_pairs_are_the_two_unreachable_ones():
    report = _analyze()
    summary = read_json(Path(report["_output_dir"]) / "missed_pairs_summary.json")
    assert summary["n_missed_in_scope"] == 2
    causes = summary["missed"]["primary_cause_share"]
    assert causes["cross_script_no_transliteration"] == 0.5
    assert causes["no_lexical_overlap"] == 0.5
    # the recovered contrast arm is what makes these shares interpretable
    assert summary["recovered_contrast"]["n_pairs"] == RETRIEVED_PAIRS
    assert "cross_script_no_transliteration" not in summary["recovered_contrast"]["primary_cause_share"]


def test_end_to_end_examples_are_the_missed_pairs_and_carry_both_names():
    report = _analyze()
    examples = pd.read_csv(Path(report["_output_dir"]) / "missed_pairs_examples.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert len(examples) == 2
    pairs = set(zip(examples["s1_id"], examples["target_id"]))
    assert pairs == MISSED_PAIRS
    names = set(examples["s1_name"]) | set(examples["target_name"])
    assert "quasar holdings" in names
    assert "meridian logistics" in names
    # the cross-script row must carry the Devanagari name, not a mis-mapped neighbour
    devanagari = examples[examples["s1_id"] == "S1-4"]
    assert len(devanagari) == 1
    assert devanagari.iloc[0]["s1_name"] == "राम मार्केटिंग"
    assert devanagari.iloc[0]["target_name"] == "ram marketing"


def test_end_to_end_proposal_recall_is_hand_verifiable():
    """token_any is df-free, so its recall is countable: every true pair sharing a token.

    S1-2/S2-2 share "exports"; S1-3 shares "acme" with both S3 rows; S1-4 (Devanagari)
    and S1-6 (no shared token) share nothing. That is 5 of 7.
    """
    report = _analyze()
    token_any = _proposal(report, "token_any")
    assert token_any["true_pairs_recovered"] == 5
    assert abs(token_any["true_pair_recall"] - 5 / N_TRUE_PAIRS) < 1e-12
    # the union already recovers those same 5, so fixing token blocking adds nothing here
    assert token_any["incremental_true_pairs"] == 0
    assert token_any["upper_bound"] is True
    assert token_any["growth_factor"] == token_any["candidate_rows"] / report["scan"]["rows_seen"]


def test_end_to_end_proposal_growth_matches_the_hand_computed_identity():
    """``sum_k n_s1(k) * n_target(k)`` computed by eye from the fixture rows.

    token_any: sunrise 1x1, traders 1x1, exports 1x1, zenith 1x1, foods 1x1,
    acme 1x2 (two S3 rows), industries 1x1, private 1x1, limited 1x1 = 10.
    prefix4: sunr 1x1, blue 1x1, acme 1x2, zeni 1x1 = 5.
    address_tokens: the four "4 mg road mumbai" tokens are 2x3 each (S1-3 and S1-6
    against S3-1, S3-2, S3-4) = 24, plus 5+4+4 for the three unique addresses = 37.
    """
    report = _analyze()
    assert _proposal(report, "token_any")["candidate_rows"] == 10
    assert _proposal(report, "prefix4")["candidate_rows"] == 5
    assert _proposal(report, "address_tokens")["candidate_rows"] == 37


def test_end_to_end_only_the_address_proposal_recovers_a_missed_pair():
    """The payoff of the whole exercise, in miniature.

    address_tokens recovers both missed pairs and nothing else does. S1-6/S3-4 shares no
    name text with its match but sits at the same address - a signal blocking ignores.
    S1-4/S3-3 is the sharper case: its names are in different scripts, so *no* lexical
    blocker can ever reach it by name, yet the address is byte-identical. char_2gram_any
    reaches S1-6/S3-4 too, by tolerating the shared "di" bigram. Most proposals add
    nothing, and the report must say so rather than implying that any proposal helps.
    """
    report = _analyze()
    assert _proposal(report, "address_tokens")["true_pairs_recovered"] == 6
    assert _proposal(report, "address_tokens")["incremental_true_pairs"] == 2
    assert _proposal(report, "char_2gram_any")["incremental_true_pairs"] == 1
    for label in ("token_sorted", "initials", "suffix_stripped", "digits_stripped", "country_prefix6"):
        assert _proposal(report, label)["incremental_true_pairs"] == 0, label
    # a proposal can only ever add the pairs the union missed
    for row in report["proposals"]:
        assert 0 <= row["incremental_true_pairs"] <= 2, row
        assert row["incremental_true_pairs"] <= row["true_pairs_recovered"], row
        assert 0.0 <= row["true_pair_recall"] <= 1.0, row
        assert row["candidate_rows"] >= 0


def test_end_to_end_proposals_are_ranked_by_incremental_recall():
    report = _analyze()
    increments = [row["incremental_true_pairs"] for row in report["proposals"]]
    assert increments == sorted(increments, reverse=True), increments


def test_end_to_end_metric_chain_applies_retention_to_entity_recall():
    report = _analyze()
    chain = report["metric_chain"]
    assert abs(chain["A_candidate_macro_entity_recall"] - 4 / 6) < 1e-12
    # A is pair recall, not macro recall - two different aggregates, both reported
    assert abs(chain["A_candidate_pair_recall"] - 5 / 7) < 1e-12
    assert chain["matcher_retention_applied"] == abe.V1_MATCHER_RETENTION
    assert chain["A_candidate_f05_ceiling"] > chain["A_candidate_macro_entity_recall"]
    assert "not" in chain["A_to_C_is_not_an_equality"].lower()


def test_end_to_end_writes_every_required_artifact():
    report = _analyze()
    output_dir = Path(report["_output_dir"])
    for name in (
        "blocking_recall_report.json",
        "blocking_recall_by_blocker.tsv",
        "missed_pairs_analysis.tsv",
        "missed_pairs_summary.json",
        "candidate_growth_estimates.tsv",
        "missed_pairs_examples.tsv",
        "blocking_error_report.md",
    ):
        assert (output_dir / name).is_file(), name

    by_blocker = pd.read_csv(output_dir / "blocking_recall_by_blocker.tsv", sep="\t")
    assert set(by_blocker["kind"]) == {"blocker", "combination"}
    growth = pd.read_csv(output_dir / "candidate_growth_estimates.tsv", sep="\t")
    assert len(growth) == len(abe.PROPOSALS)
    assert {"proposal", "true_pair_recall", "candidate_rows", "growth_factor"} <= set(growth.columns)
    analysis = pd.read_csv(output_dir / "missed_pairs_analysis.tsv", sep="\t")
    assert set(analysis["kind"]) == {"numeric", "boolean", "cause"}
    markdown = (output_dir / "blocking_error_report.md").read_text(encoding="utf-8")
    for heading in ("## 1. Current baseline", "## 5. Proposed blockers", "## 7. Recommendation"):
        assert heading in markdown, heading
    assert "upper bound" in markdown or "exact" in markdown


def test_run_respects_the_proposal_subset_flag():
    report = _analyze(["--proposals", "token_any,prefix4"], tag="subset")
    labels = {row["proposal"] for row in report["proposals"]}
    assert labels == {"token_any", "prefix4"}


def test_run_supports_the_val_mask_path():
    """``--split val`` must exercise the mask without inventing entities."""
    report = _analyze(["--split", "val"], tag="val")
    assert 0 <= report["entity"]["entities_scored"] <= N_ENTITIES
    assert report["context"]["split"] == "val"


def test_skip_flags_produce_a_report_without_optional_sections():
    report = _analyze(["--skip-proposals", "--skip-diagnostics"], tag="minimal")
    assert report["proposals"] == []
    output_dir = Path(report["_output_dir"])
    summary = read_json(output_dir / "missed_pairs_summary.json")
    assert summary["n_missed_in_scope"] == 2
    assert (output_dir / "blocking_recall_by_blocker.tsv").is_file()


def test_limit_rows_restricts_the_prefix_and_is_flagged():
    full = _analyze()
    limited = _analyze(["--limit-rows", "2"], tag="limited")
    assert limited["scan"]["rows_seen"] == 2
    assert limited["scan"]["rows_seen"] < full["scan"]["rows_seen"]
    # a prefix cannot retrieve more than the whole file
    assert limited["official_metrics"]["true_pairs_retrieved"] <= full["official_metrics"]["true_pairs_retrieved"]


def test_scan_counts_unknown_and_unlabelled_rows_separately():
    """Provenance drift must be visible, not silently folded into a blocker's count."""
    series = pd.Series(["source2:exact_name", "source2:mystery", ""])
    flags = abe.provenance_flags(series)
    any_flag = flags["exact_name"] | flags["token"] | flags["char_ngram"]
    assert any_flag.tolist() == [True, False, False]


def _main() -> int:
    logging.basicConfig(level=logging.CRITICAL)
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    try:
        for name, test in tests:
            try:
                test()
            except AssertionError as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
            else:
                print(f"[PASS] {name}")
    finally:
        for directory in _TEMP_DIRS:
            shutil.rmtree(directory, ignore_errors=True)
    print()
    if failures:
        print(f"{failures} of {len(tests)} tests failed")
        return 1
    print(f"all {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
