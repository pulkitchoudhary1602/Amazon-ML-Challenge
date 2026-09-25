"""Tests for ``scripts/extract_pair_features.py`` (the Step 3 de-risk experiment).

What this file is guarding
--------------------------
Every feature the first matcher will see is computed here, so the two mistakes
that would silently poison the model are the ones under test:

1. **A blank evidence cell read as a zero.** The union writes ``token_df`` blank
   for every pair the token blocker did not propose. That blank means "not
   measured", not "scored 0", and it must become NaN.
2. **A failed text join read as a measurement.** If an id is missing from the
   prepared file, the pair still gets a row (never dropped - one row per
   candidate pair is the contract), but every text-derived feature must be
   blanked, while provenance, evidence and the S1 candidate count - which come
   from the candidate file, not the join - must survive.

It also guards the two invariants the sampling design rests on: whole S1 entities
are kept or dropped together (even when a small chunk size splits an entity across
chunks), and the per-S1 row count in the sample agrees with the count phase 1
recorded. The agreement check is the whole-entity invariant, and it is exercised
directly by corrupting the sample file.

There is no ground-truth file anywhere in this fixture, deliberately: the split
comes from ``assign_splits``, the same pure function ``src/evaluation.py`` uses,
so nothing here can leak a label into a feature.

The fixture is synthetic and self-contained. Nothing reads the dataset, and every
file written goes into a temp directory - ``outputs/`` is never touched.

Runs standalone (``python tests/test_pair_features.py``) and under pytest.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import assign_splits  # noqa: E402
from src.utils import stable_hash64  # noqa: E402

import scripts.extract_pair_features as epf  # noqa: E402

RAPIDFUZZ_INSTALLED = importlib.util.find_spec("rapidfuzz") is not None
RATIO_FEATURES = ("name_token_set_ratio", "name_token_sort_ratio", "name_partial_ratio")

CANDIDATE_COLUMNS = [
    epf.CANDIDATE_S1_COLUMN,
    epf.CANDIDATE_TARGET_COLUMN,
    epf.CANDIDATE_SOURCE_COLUMN,
    epf.CANDIDATE_BLOCKERS_COLUMN,
    "token_df",
    "char_jaccard",
]

# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------
N_S1 = 60
S2_NAMES = {"S2-1": "acme alpha limited", "S2-2": "acme beta limited",
            "S2-3": "acme gamma limited"}
S3_NAMES = {"S3-1": "acme delta limited"}


def _text_row(entity_id: str, name: str, address: str, country: str) -> dict:
    return {
        "entity_id": entity_id,
        epf.PREPARED_NAME_NORM: name,
        epf.PREPARED_NAME_KEY: name.replace(" ", ""),
        epf.PREPARED_ADDRESS_NORM: address,
        epf.PREPARED_COUNTRY_NORM: country,
    }


def _s1_row(index: int) -> dict:
    """An S1 record with every missing/differing case the fixture needs represented.

    Address: missing every fifth entity. Country: missing every seventh, and a
    different value every eleventh, so ``country_equal`` and ``country_missing``
    both see a 0 and a 1.
    """
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    address = "" if index % 5 == 0 else "1 main st bengaluru"
    if index % 7 == 0:
        country = ""
    elif index % 11 == 0:
        country = "us"
    else:
        country = "in"
    return _text_row(f"S1-{index}", f"acme {words[index % len(words)]} limited",
                     address, country)


def _prepared_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": [row["entity_id"] for row in rows],
            "business_name": [row[epf.PREPARED_NAME_NORM] for row in rows],
            "business_address": [row[epf.PREPARED_ADDRESS_NORM] for row in rows],
            "country": [row[epf.PREPARED_COUNTRY_NORM] for row in rows],
            epf.PREPARED_NAME_NORM: [row[epf.PREPARED_NAME_NORM] for row in rows],
            epf.PREPARED_NAME_KEY: [row[epf.PREPARED_NAME_KEY] for row in rows],
            epf.PREPARED_ADDRESS_NORM: [row[epf.PREPARED_ADDRESS_NORM] for row in rows],
            epf.PREPARED_COUNTRY_NORM: [row[epf.PREPARED_COUNTRY_NORM] for row in rows],
        }
    )


def _fixture_candidate_rows() -> list[tuple]:
    """Three pairs per S1 - both evidence columns get measured and blank rows.

    The blank evidence is the point: ``token_df`` is blank on every pair the token
    blocker did not propose and ``char_jaccard`` on every pair the char blocker did
    not propose, so a fixture where only one of them is ever blank could not catch
    a blank being read as a zero.
    """
    rows = []
    for index in range(N_S1):
        entity_id = f"S1-{index}"
        rows.append((entity_id, "S2-1", "S2", "source2:exact_name", "3", ""))
        rows.append((entity_id, "S2-2", "S2", "source2:token", "5", ""))
        rows.append((entity_id, "S3-1", "S3", "source3:char_ngram", "", "0.4444"))
        if index == 0:
            # A duplicated pair: both copies must survive and be counted.
            rows.append((entity_id, "S2-1", "S2", "source2:exact_name", "3", ""))
            # A target id that is absent from the prepared files.
            rows.append((entity_id, "S9-7", "S2", "source2:token", "5", ""))
            # Two blockers on one pair, comma-joined as union_blockers writes them.
            rows.append((entity_id, "S2-3", "S2", "source2:exact_name,source2:token", "7", "0.6"))
    return rows


def _write_fixture(root: Path) -> Path:
    prepared = root / "prepared"
    candidates = root / "candidates"
    for path in (prepared, candidates):
        path.mkdir(parents=True, exist_ok=True)

    s1_rows = [_s1_row(index) for index in range(N_S1)]
    _prepared_frame(s1_rows).to_csv(prepared / "train_source1_norm.tsv", sep="\t", index=False)
    _prepared_frame([_text_row("S2-1", S2_NAMES["S2-1"], "1 main st bengaluru", "in"),
                     _text_row("S2-2", S2_NAMES["S2-2"], "", ""),
                     _text_row("S2-3", S2_NAMES["S2-3"], "1 main st bengaluru", "in")]).to_csv(
        prepared / "train_source2_norm.tsv", sep="\t", index=False
    )
    _prepared_frame([_text_row("S3-1", S3_NAMES["S3-1"], "9 other rd pune", "in")]).to_csv(
        prepared / "train_source3_norm.tsv", sep="\t", index=False
    )

    pd.DataFrame(_fixture_candidate_rows(), columns=CANDIDATE_COLUMNS).to_csv(
        candidates / "candidate_pairs.tsv", sep="\t", index=False
    )

    config_path = root / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "project: {name: er, seed: 42}",
                f"paths: {{data_root: '{(root / 'raw').as_posix()}', "
                f"prepared_dir: '{prepared.as_posix()}', "
                f"index_dir: '{(root / 'indexes').as_posix()}', "
                f"candidates_dir: '{candidates.as_posix()}', "
                f"log_dir: '{(root / 'logs').as_posix()}'}}",
                "io: {chunksize: 50, prepared_format: tsv, candidates_format: tsv}",
                "columns: {entity_id: entity_id, name: business_name, "
                "address: business_address, country: country}",
                "evaluation: {split: {enabled: true, val_fraction: 0.2, mode: hash}, "
                "zero_match_policy: exclude}",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _candidate_frame(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "candidates" / "candidate_pairs.tsv", sep="\t", dtype=str)


def _val_ids() -> set[str]:
    ids = pd.Series([f"S1-{index}" for index in range(N_S1)], dtype=object)
    labels = assign_splits(ids, val_fraction=0.2, mode="hash", seed=42)
    return {entity_id for entity_id, label in zip(ids, labels) if label == "val"}


class _Fixture:
    """Temp tree plus a logger; ``close`` detaches the handlers before deleting."""

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="step3_features_test_"))
        self.config_path = _write_fixture(self.root)
        self.out = self.root / "out"
        self.out.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(epf.LOG_NAME)
        self.log.setLevel(logging.CRITICAL)

    @property
    def config(self) -> dict:
        return epf.load_config(str(self.config_path))

    def argv(self, **overrides) -> list[str]:
        argv = ["--config", str(self.config_path), "--split", "train",
                "--output-dir", str(self.out)]
        for key, value in overrides.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        return argv

    def args(self, **overrides):
        return epf.parse_args(self.argv(**overrides))

    def run(self, **overrides):
        code = epf.main(self.argv(**overrides))
        report = json.loads((self.out / "step3_features_report.json").read_text())
        return code, report

    def features(self) -> pd.DataFrame:
        return pd.read_csv(self.out / "features.tsv", sep="\t", dtype=str)

    def sample(self) -> pd.DataFrame:
        return pd.read_csv(self.out / "sample_candidates.tsv", sep="\t", dtype=str)

    def close(self) -> None:
        # The handler holds the log file open, and on Windows an open file cannot
        # be deleted - detach it so each fixture gets a clean logger and its own
        # directory really does go away.
        for handler in list(self.log.handlers):
            self.log.removeHandler(handler)
            handler.close()
        shutil.rmtree(self.root, ignore_errors=True)


def _fixture() -> _Fixture:
    return _Fixture()


# ---------------------------------------------------------------------------
# pure helpers: the semantics of every similarity live here
# ---------------------------------------------------------------------------
def test_token_set_splits_and_dedupes():
    assert epf.token_set("acme alpha limited") == frozenset({"acme", "alpha", "limited"})
    assert epf.token_set("alpha alpha alpha") == frozenset({"alpha"})
    assert epf.token_set("") == frozenset()


def test_set_jaccard_hand_computed():
    left, right = frozenset({"a", "b", "c"}), frozenset({"b", "c", "d"})
    assert epf.set_jaccard(left, right) == 0.5
    assert epf.set_jaccard(left, left) == 1.0
    assert epf.set_jaccard(left, frozenset({"z"})) == 0.0
    # An empty side means "nothing to compare", never a perfect match.
    assert epf.set_jaccard(left, frozenset()) == 0.0
    assert epf.set_jaccard(frozenset(), frozenset()) == 0.0


def test_length_ratio_hand_computed():
    assert epf.length_ratio("abcd", "abcd") == 1.0
    assert epf.length_ratio("abcd", "ab") == 0.5
    assert epf.length_ratio("", "abcd") == 0.0


def test_first_token_hand_computed():
    assert epf.first_token("acme alpha limited") == "acme"
    assert epf.first_token("   spaced   out  ") == "spaced"
    assert epf.first_token("") == ""


def test_parse_provenance_hand_computed():
    assert epf.parse_provenance("source2:exact_name") == (1, 0, 0, 1, 0)
    assert epf.parse_provenance("source2:token") == (0, 1, 0, 1, 0)
    assert epf.parse_provenance("source3:char_ngram") == (0, 0, 1, 1, 0)
    assert epf.parse_provenance("source2:exact_name,source2:token") == (1, 1, 0, 2, 0)
    assert epf.parse_provenance("source2:exact_name,source2:token,source3:char_ngram") == (1, 1, 1, 3, 0)
    assert epf.parse_provenance("") == (0, 0, 0, 0, 0)
    # An unrecognised label is counted, never silently dropped.
    assert epf.parse_provenance("source2:levenshtein") == (0, 0, 0, 0, 1)
    # The label after the LAST ":" identifies the blocker, whatever the source.
    assert epf.parse_provenance("source3:token") == (0, 1, 0, 1, 0)


def test_evidence_columns_follow_the_union():
    assert list(epf.EVIDENCE_FLOAT_COLUMNS) == list(epf.evidence_columns_for(epf.UNION_BLOCKERS))
    assert set(epf.EVIDENCE_FLOAT_COLUMNS) == {"token_df", "char_jaccard"}


def test_unit_interval_guard_excludes_the_document_frequency():
    # token_df is a document frequency (values in the thousands); putting it in the
    # [0, 1] guard would make every run fail its own sanity check.
    assert "token_df" not in epf.UNIT_INTERVAL_FEATURES
    assert "char_jaccard" in epf.UNIT_INTERVAL_FEATURES
    assert set(RATIO_FEATURES) <= set(epf.UNIT_INTERVAL_FEATURES)


# ---------------------------------------------------------------------------
# build_features: exact per-column expectations
# ---------------------------------------------------------------------------
def _lookup(source: str, rows: list[dict]) -> epf.PreparedLookup:
    ids = np.array([row["entity_id"] for row in rows], dtype=object)
    columns = {
        column: np.array([row[column] for row in rows], dtype=object)
        for column in (epf.PREPARED_NAME_NORM, epf.PREPARED_NAME_KEY,
                       epf.PREPARED_ADDRESS_NORM, epf.PREPARED_COUNTRY_NORM)
    }
    return epf.PreparedLookup(source, ids, columns)


def _mini_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Identical name and address; token_df measured, char_jaccard not.
            ("S1-1", "S2-1", "S2", "source2:token", "3", ""),
            # One name token differs; the char blocker proposed it, the token one did not.
            ("S1-1", "S2-2", "S2", "source2:char_ngram", "", "0.6"),
            # Target id absent from the prepared source -> join failure.
            ("S1-2", "S2-9", "S2", "source2:token", "12", ""),
            # Source label that was never loaded -> join failure, unknown source.
            ("S1-2", "S9-1", "S9", "source9:token", "12", ""),
        ],
        columns=CANDIDATE_COLUMNS,
    )


def _mini_built() -> tuple[pd.DataFrame, dict]:
    s1 = _lookup("source1", [
        _text_row("S1-1", "acme alpha limited", "1 main st bengaluru", "in"),
        _text_row("S1-2", "gamma delta limited", "", ""),
    ])
    targets = {
        "S2": _lookup("source2", [
            _text_row("S2-1", "acme alpha limited", "1 main st bengaluru", "in"),
            _text_row("S2-2", "acme beta limited", "1 main st bengaluru", "in"),
        ]),
        "S3": _lookup("source3", [_text_row("S3-1", "acme gamma limited", "1 main st bengaluru", "in")]),
    }
    integrity: dict = {}
    features = epf.build_features(
        _mini_frame(), targets, s1, {"S1-1": 2, "S1-2": 2}, integrity
    )
    return features, integrity


def test_build_features_keeps_one_row_per_candidate_pair():
    features, _ = _mini_built()
    assert len(features) == len(_mini_frame())
    # Same order and same ids: nothing dropped, reordered or deduplicated.
    assert list(features[epf.CANDIDATE_TARGET_COLUMN]) == ["S2-1", "S2-2", "S2-9", "S9-1"]
    assert list(features[epf.CANDIDATE_S1_COLUMN]) == ["S1-1", "S1-1", "S1-2", "S1-2"]


def _approx(value: float, expected: float, tolerance: float = 1e-5) -> bool:
    return abs(float(value) - expected) <= tolerance


def test_build_features_name_columns_hand_computed():
    features, _ = _mini_built()
    row = features.iloc[0]
    assert row["name_norm_equal"] == 1
    assert row["name_key_equal"] == 1
    assert row["name_token_jaccard"] == 1.0
    assert row["name_length_ratio"] == 1.0
    assert row["name_token_count_diff"] == 0
    assert row["name_first_token_equal"] == 1
    assert row["name_char3_jaccard"] == 1.0

    # "acme alpha limited" vs "acme beta limited": one token differs.
    # 17 characters vs 18, 2 of 4 distinct tokens shared.
    row = features.iloc[1]
    assert row["name_norm_equal"] == 0
    assert row["name_token_jaccard"] == 0.5
    assert row["name_token_count_diff"] == 0
    assert row["name_first_token_equal"] == 1
    assert _approx(row["name_length_ratio"], 17 / 18)


def test_build_features_rapidfuzz_ratios():
    features, integrity = _mini_built()
    assert integrity["rapidfuzz_available"] == (1 if RAPIDFUZZ_INSTALLED else 0)
    if not RAPIDFUZZ_INSTALLED:
        # The fallback must be NaN, never a fabricated 0.
        for column in RATIO_FEATURES:
            assert features[column].isna().all(), column
        return

    for column in RATIO_FEATURES:
        assert features.iloc[0][column] == 1.0, column
    # Measured values for "acme alpha limited" vs "acme beta limited" - asserted
    # exactly so a rapidfuzz upgrade that changes a score shows up as a failure
    # rather than as silently different training data.
    assert _approx(features.iloc[1]["name_token_set_ratio"], 0.82758623)
    assert _approx(features.iloc[1]["name_token_sort_ratio"], 0.8)
    assert _approx(features.iloc[1]["name_partial_ratio"], 0.7647059)
    assert features.iloc[1]["name_token_set_ratio"] >= features.iloc[1]["name_token_sort_ratio"]


def test_token_set_ratio_scores_containment_as_a_perfect_match():
    """Why the ratio features cannot be used alone.

    Measured, not assumed: ``token_set_ratio("acme alpha limited", "acme alpha")``
    is 1.0, because one token set contains the other. A short trading name that is
    a strict prefix of a long one therefore looks like a perfect name match to that
    feature. The accompanying jaccard and length ratio are what keep containment
    from reading as a match, so a test that forgot them would not notice if the
    ratio feature were quietly doing all the work.
    """
    if not RAPIDFUZZ_INSTALLED:  # pragma: no cover - depends on the environment
        return
    lookups = {"S2": _lookup("source2", [_text_row("S2-1", "acme alpha", "", "")])}
    s1 = _lookup("source1", [_text_row("S1-1", "acme alpha limited", "", "")])
    frame = pd.DataFrame(
        [("S1-1", "S2-1", "S2", "source2:token", "1", "0.0")],
        columns=CANDIDATE_COLUMNS,
    )
    row = epf.build_features(frame, lookups, s1, {"S1-1": 1}, {}).iloc[0]
    assert row["name_token_set_ratio"] == 1.0
    assert _approx(row["name_token_jaccard"], 2 / 3)
    assert _approx(row["name_length_ratio"], 10 / 18)
    assert row["name_norm_equal"] == 0


def test_build_features_address_columns_hand_computed():
    features, _ = _mini_built()
    row = features.iloc[0]
    assert row["address_norm_equal"] == 1
    assert row["address_token_jaccard"] == 1.0
    assert row["address_shared_token_count"] == 4     # 1 / main / st / bengaluru
    assert row["address_length_ratio"] == 1.0
    assert row["s1_address_missing"] == 0
    assert row["target_address_missing"] == 0
    assert row["both_address_missing"] == 0


def test_build_features_blank_evidence_becomes_nan_not_zero():
    features, integrity = _mini_built()
    # Row 0: token blocker measured token_df; char blocker did not propose the pair.
    assert features.iloc[0]["token_df"] == 3.0
    assert np.isnan(features.iloc[0]["char_jaccard"])
    # Row 1: the reverse.
    assert np.isnan(features.iloc[1]["token_df"])
    assert features.iloc[1]["char_jaccard"] == 0.6
    # One blank token_df (row 1) and three blank char_jaccard (rows 0, 2, 3).
    assert integrity["token_df_blank"] == 1
    assert integrity["char_jaccard_blank"] == 3


def test_build_features_join_failure_blanks_text_but_keeps_file_evidence():
    features, integrity = _mini_built()
    row = features.iloc[2]              # S2-9 is not in the prepared source
    assert row["text_join_ok"] == 0
    # Text-derived features are blanked...
    assert row["name_norm_equal"] == 0
    assert row["name_token_count_diff"] == 0
    assert np.isnan(row["name_token_jaccard"])
    assert np.isnan(row["name_char3_jaccard"])
    assert np.isnan(row["name_length_ratio"])
    assert np.isnan(row["address_token_jaccard"])
    # ...but provenance, evidence and the S1 candidate count come from the
    # candidate file, so they stay valid and must survive.
    assert row["blocker_token"] == 1
    assert row["n_blockers"] == 1
    assert row["token_df"] == 12.0
    assert row["s1_candidate_count"] == 2

    assert integrity["s1_join_failures"] == 0
    assert integrity["target_join_failures"] == 1     # row 2 only
    assert integrity["unknown_source_labels"] == 1    # row 3
    assert features.iloc[3]["text_join_ok"] == 0
    assert features.iloc[3]["token_df"] == 12.0


def test_build_features_blank_text_is_not_evidence_of_a_match():
    lookups = {"S2": _lookup("source2", [_text_row("S2-1", "acme alpha limited", "", "in")])}
    s1 = _lookup("source1", [_text_row("S1-1", "", "", "")])
    frame = pd.DataFrame(
        [("S1-1", "S2-1", "S2", "source2:exact_name", "1", "0.0")],
        columns=CANDIDATE_COLUMNS,
    )
    features = epf.build_features(frame, lookups, s1, {"S1-1": 1}, {})
    row = features.iloc[0]
    # The join SUCCEEDED, so this is not the blanked-on-failure path: a genuinely
    # empty field must still not read as "the two records agree".
    assert row["text_join_ok"] == 1
    assert row["name_norm_equal"] == 0
    assert row["name_key_equal"] == 0
    assert row["address_norm_equal"] == 0
    assert row["country_equal"] == 0
    assert row["country_missing"] == 1
    assert row["name_length_ratio"] == 0.0
    assert row["address_token_jaccard"] == 0.0


def test_build_features_dtypes_match_the_declaration():
    features, _ = _mini_built()
    for column, dtype in epf.FEATURE_DTYPES.items():
        assert str(features[column].dtype) == dtype, (column, features[column].dtype, dtype)


def test_build_features_unit_interval_features_stay_in_range():
    features, _ = _mini_built()
    for column in epf.UNIT_INTERVAL_FEATURES:
        values = features[column].to_numpy(dtype=np.float64)
        finite = values[~np.isnan(values)]
        assert finite.size, column
        assert finite.min() >= 0.0 and finite.max() <= 1.0, (column, finite.min(), finite.max())


def test_build_features_has_no_ground_truth_column():
    features, _ = _mini_built()
    forbidden = {"label", "is_match", "is_true", "target", "y", "ground_truth",
                 "true_match", "match", "score"}
    assert not (set(features.columns) & forbidden)
    # The frame is exactly the declared features plus the three id columns.
    extra = set(features.columns) - set(epf.FEATURE_DTYPES)
    assert extra == {epf.CANDIDATE_S1_COLUMN, epf.CANDIDATE_TARGET_COLUMN,
                     epf.CANDIDATE_SOURCE_COLUMN}


# ---------------------------------------------------------------------------
# PreparedLookup
# ---------------------------------------------------------------------------
def test_prepared_lookup_take_and_values():
    lookup = _lookup("source2", [
        _text_row("S2-1", "acme alpha limited", "1 main st", "in"),
        _text_row("S2-2", "beta limited", "", ""),
    ])
    positions, found = lookup.take(np.array(["S2-2", "S2-9", "S2-1"], dtype=object))
    assert list(positions) == [1, -1, 0]
    assert list(found) == [True, False, True]
    assert list(lookup.values(epf.PREPARED_NAME_NORM, positions, found)) == [
        "beta limited", "", "acme alpha limited"
    ]
    assert lookup.values(epf.PREPARED_NAME_NORM, np.array([-1]), np.array([False]))[0] == ""
    assert lookup.n_entities == 2


def test_prepared_lookup_memory_counts_the_string_payload():
    lookup = _lookup("source2", [
        _text_row(f"S2-{index}", "acme alpha limited", "1 main st bengaluru", "in")
        for index in range(500)
    ])
    # A naive nbytes-only estimate would be a few kB; the strings are the point.
    assert lookup.memory_bytes() > 500 * 40


# ---------------------------------------------------------------------------
# sampling: whole entities, deterministic, and a different hash slice than the split
# ---------------------------------------------------------------------------
def test_sample_mask_keeps_whole_entities_and_agrees_with_assign_splits():
    fixture = _fixture()
    try:
        config = fixture.config
        ids = np.array([f"S1-{index}" for index in range(N_S1)], dtype=object)

        labels = assign_splits(pd.Series(ids, dtype=object), val_fraction=0.2, mode="hash", seed=42)
        # Sample everything: the kept set is then exactly the validation split, which
        # is the split the evaluator uses - so the sample cannot straddle it.
        kept = epf._sample_mask_for_ids(ids.copy(), {}, config, 1_000_000, fixture.log)
        assert list(kept) == list(labels == "val")
        assert 0 < kept.sum() < len(ids)

        partial = epf._sample_mask_for_ids(ids.copy(), {}, config, 300_000, fixture.log)
        assert (partial & ~kept).sum() == 0, "the subsample must be a subset of the val set"
        assert 0 < partial.sum() < kept.sum()

        # Deterministic from a cold cache, and a warm cache never changes an answer.
        assert list(epf._sample_mask_for_ids(ids.copy(), {}, config, 300_000, fixture.log)) == list(partial)
        mixed: dict[str, int] = {}
        assert list(epf._sample_mask_for_ids(ids[:5].copy(), mixed, config, 300_000, fixture.log)) == list(partial[:5])
        assert list(epf._sample_mask_for_ids(ids.copy(), mixed, config, 300_000, fixture.log)) == list(partial)
    finally:
        fixture.close()


def test_subsample_bucket_is_a_different_hash_slice_than_the_split():
    """The two decisions must read different bits of the same hash.

    If they read the same bits, "val" would already imply "kept" and the
    subsample could not select a fraction of the val entities at all.
    """
    hashed = stable_hash64(pd.Series([f"S1-{index}" for index in range(400)], dtype=object))
    high = (hashed // np.uint64(1_000_000)) % np.uint64(1_000_000)
    low = hashed % np.uint64(1_000_000)
    assert list(high) != list(low)
    assert high.min() < 300_000 < high.max()


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
def test_main_writes_every_output_and_exits_zero():
    fixture = _fixture()
    try:
        code, report = fixture.run(sample_fraction=1.0)
        assert code == 0, report["integrity"]
        for name in ("sample_candidates.tsv", "features.tsv",
                     "feature_missingness.csv", "step3_features_report.json",
                     "extract_pair_features.log"):
            assert (fixture.out / name).is_file(), name
        assert report["integrity"]["count_mismatches"] == 0
        assert report["integrity"]["rapidfuzz_available"] == (1 if RAPIDFUZZ_INSTALLED else 0)
    finally:
        fixture.close()


def test_sample_is_exactly_the_val_entities():
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0)
        candidates = _candidate_frame(fixture.root)
        val = _val_ids()
        sample = fixture.sample()

        # Every sampled entity is in val, and every val entity is sampled.
        assert set(sample[epf.CANDIDATE_S1_COLUMN]) == val

        # Whole entities: each kept entity contributes exactly its file rows, which
        # is also the proof that no partial entity slipped through.
        expected = candidates[candidates[epf.CANDIDATE_S1_COLUMN].isin(val)]
        assert len(sample) == len(expected)
        assert sample[epf.CANDIDATE_S1_COLUMN].value_counts().to_dict() == (
            expected[epf.CANDIDATE_S1_COLUMN].value_counts().to_dict()
        )
    finally:
        fixture.close()


def test_whole_entities_survive_a_chunk_boundary():
    """A 7-row chunk size splits entities across chunks; the sample must not care."""
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0, chunksize=7)
        sample = fixture.sample()
        expected = _candidate_frame(fixture.root)
        val = _val_ids()
        assert set(sample[epf.CANDIDATE_S1_COLUMN]) == val
        assert len(sample) == len(expected[expected[epf.CANDIDATE_S1_COLUMN].isin(val)])
        # The per-entity count recorded for the feature came from the whole file,
        # not from the chunk it happened to arrive in, so it must match the file's
        # own count for that entity.
        file_counts = expected[epf.CANDIDATE_S1_COLUMN].value_counts().to_dict()
        seen = (fixture.features()[[epf.CANDIDATE_S1_COLUMN, "s1_candidate_count"]]
                .drop_duplicates().set_index(epf.CANDIDATE_S1_COLUMN)["s1_candidate_count"])
        assert set(seen.index) == val
        for entity_id, value in seen.items():
            assert int(value) == file_counts[entity_id], (entity_id, value, file_counts[entity_id])
    finally:
        fixture.close()


def test_one_row_per_candidate_pair_is_preserved_end_to_end():
    fixture = _fixture()
    try:
        code, report = fixture.run(sample_fraction=1.0)
        assert code == 0
        sample = fixture.sample()
        features = fixture.features()
        assert len(features) == len(sample)
        assert list(features[epf.CANDIDATE_TARGET_COLUMN]) == list(sample[epf.CANDIDATE_TARGET_COLUMN])
        assert list(features[epf.CANDIDATE_S1_COLUMN]) == list(sample[epf.CANDIDATE_S1_COLUMN])
        assert report["sample"]["n_candidate_pairs_sampled"] == len(features)
        assert report["sample"]["n_s1_entities_sampled"] == len(_val_ids())
    finally:
        fixture.close()


def test_duplicate_pairs_are_counted_not_collapsed():
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0)
        candidates = _candidate_frame(fixture.root)
        sample = fixture.sample()
        keys = list(zip(sample[epf.CANDIDATE_S1_COLUMN], sample[epf.CANDIDATE_TARGET_COLUMN]))
        expected_duplicates = len(keys) - len(set(keys))
        report = json.loads((fixture.out / "step3_features_report.json").read_text())
        assert report["integrity"]["duplicate_sampled_pairs"] == expected_duplicates
        # S1-0 is the entity carrying the duplicated pair, and both copies survive.
        if "S1-0" in _val_ids():
            assert expected_duplicates == 1
            assert keys.count(("S1-0", "S2-1")) == 2
        else:
            assert expected_duplicates == 0
    finally:
        fixture.close()


def test_join_failures_are_kept_and_counted():
    fixture = _fixture()
    try:
        code, report = fixture.run(sample_fraction=1.0)
        assert code == 0
        features = fixture.features()
        if "S1-0" in _val_ids():
            broken = features[features[epf.CANDIDATE_TARGET_COLUMN] == "S9-7"]
            assert len(broken) == 1, "a pair whose target is missing must still get a row"
            assert broken.iloc[0]["text_join_ok"] == "0"
            assert report["integrity"]["target_join_failures"] == 1
        else:
            assert report["integrity"]["target_join_failures"] == 0
        assert report["integrity"]["unknown_source_labels"] == 0
    finally:
        fixture.close()


def test_missingness_records_blank_evidence_not_zero():
    fixture = _fixture()
    try:
        _, report = fixture.run(sample_fraction=1.0)
        missingness = report["features"]["missingness"]
        # Every declared feature is reported.
        assert set(missingness) == set(epf.FEATURE_DTYPES)
        for column, entry in missingness.items():
            assert 0.0 <= entry["rate"] <= 1.0, column
            assert entry["dtype"] == epf.FEATURE_DTYPES[column], column
            assert entry["out_of_unit_range"] == 0, column

        # The report's missing counts must agree with the written matrix, so a
        # missingness figure can never describe a file other than the one on disk.
        features = fixture.features()
        for column in epf.FEATURE_DTYPES:
            assert missingness[column]["count"] == int(features[column].isna().sum()), column

        # Half the fixture's pairs have a blank token_df and the other half a blank
        # char_jaccard, so neither may be silently turned into 0.
        assert missingness["token_df"]["rate"] > 0.0
        assert missingness["char_jaccard"]["rate"] > 0.0
        # Integer features are never NaN: a blanked int is set to 0, deliberately,
        # because NaN is not representable in an integer column.
        for column, dtype in epf.FEATURE_DTYPES.items():
            if dtype != "float32":
                assert missingness[column]["count"] == 0, column
    finally:
        fixture.close()


def test_missingness_csv_lists_every_feature():
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0)
        csv = pd.read_csv(fixture.out / "feature_missingness.csv", sep="\t")
        assert list(csv["feature"]) == list(epf.FEATURE_DTYPES)
        assert set(csv.columns) >= {"feature", "dtype", "missing_rate", "missing_count"}
    finally:
        fixture.close()


def test_report_carries_every_required_measurement():
    fixture = _fixture()
    try:
        _, report = fixture.run(sample_fraction=1.0)
        assert report["sample"]["n_s1_entities_sampled"] > 0
        assert report["sample"]["n_candidate_pairs_sampled"] > 0
        assert report["timing"]["feature_pairs_per_sec"] > 0
        assert report["timing"]["total_seconds"] > 0
        assert report["memory"]["peak_rss_bytes"] is None or report["memory"]["peak_rss_bytes"] > 0
        assert report["memory"]["peak_rss_source"]
        assert report["features"]["dtypes"] and report["features"]["matrix_bytes"] > 0
        assert report["features"]["matrix_size"]
        assert report["features"]["missingness"]
        # The report must say which columns are features and which are diagnostics,
        # so a trainer never has to guess whether text_join_ok is trainable.
        assert set(report["features"]["non_feature_columns"]) == {"text_join_ok"}
        assert report["features"]["n_features"] == len(epf.FEATURE_DTYPES) - 1
        assert set(report["integrity"]["matrix_columns"]) == set(epf.FEATURE_DTYPES)
        assert not (set(report["features"]["non_feature_columns"])
                    & (set(epf.FEATURE_DTYPES) - {"text_join_ok"}))
        assert report["integrity"]["s1_join_failures"] == 0
        assert "duplicate_sampled_pairs" in report["integrity"]
        assert report["outputs"]["features"]["bytes"] > 0
        assert report["extrapolation"]["full_candidate_pairs"] == 336_056_756
        assert report["extrapolation"]["matrix_size_full"]
        assert report["extrapolation"]["matrix_fits_in_ram"] in (True, False)
        # The projections are labelled with how much they can be trusted, and the
        # caveats that make them upper bounds are stated, not implied.
        assert report["extrapolation"]["confidence"]
        assert len(report["extrapolation"]["assumptions"]) >= 2
        assert report["extrapolation"]["scan_timing_is_a_measurement"] is True
    finally:
        fixture.close()


def test_limit_rows_labels_the_scan_projection_as_an_extrapolation():
    fixture = _fixture()
    try:
        _, report = fixture.run(sample_fraction=1.0, limit_rows=40)
        assert report["extrapolation"]["scan_timing_is_a_measurement"] is False
        assert any("NOT a measurement" in text for text in report["extrapolation"]["assumptions"])
    finally:
        fixture.close()


def test_count_mismatch_is_detected():
    """The whole-entity invariant: phase 1's per-S1 count must equal phase 2's.

    Dropping a row out of the sample file is exactly what a partial entity would
    look like, and it must be reported rather than silently trained on.
    """
    fixture = _fixture()
    try:
        args = fixture.args(sample_fraction=1.0)
        config = fixture.config
        scan = epf.scan_and_sample(config, args, fixture.out, fixture.log)
        assert scan["rows_sampled"] > 1

        path = scan["sample_path"]
        frame = pd.read_csv(path, sep="\t", dtype=str)
        frame.iloc[:-1].to_csv(path, sep="\t", index=False)

        features = epf.extract_features(config, args, scan, fixture.out, fixture.log)
        assert features["integrity"]["count_mismatches"] == 1
        assert features["integrity"]["count_mismatch_examples"]
    finally:
        fixture.close()


def test_empty_sample_fails_loudly():
    fixture = _fixture()
    try:
        args = fixture.args(sample_fraction=0.0)
        try:
            epf.scan_and_sample(fixture.config, args, fixture.out, fixture.log)
        except RuntimeError as exc:
            assert "sample" in str(exc)
        else:  # pragma: no cover - the assertion below is the real check
            raise AssertionError("an empty sample must not pass silently")
    finally:
        fixture.close()


def test_two_runs_are_byte_identical():
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0, chunksize=7)
        first_sample = (fixture.out / "sample_candidates.tsv").read_bytes()
        first_features = (fixture.out / "features.tsv").read_bytes()
        fixture.run(sample_fraction=1.0, chunksize=11)
        # Different chunk sizes must not change a single byte of the output.
        assert (fixture.out / "sample_candidates.tsv").read_bytes() == first_sample
        assert (fixture.out / "features.tsv").read_bytes() == first_features
    finally:
        fixture.close()


def test_subsample_is_a_subset_of_the_full_sample():
    fixture = _fixture()
    try:
        fixture.run(sample_fraction=1.0)
        small_out = fixture.root / "out_small"
        epf.main(["--config", str(fixture.config_path), "--split", "train",
                  "--sample-fraction", "0.30", "--output-dir", str(small_out)])
        small = pd.read_csv(small_out / "sample_candidates.tsv", sep="\t", dtype=str)
        whole = fixture.sample()
        assert 0 < len(small) < len(whole)
        assert set(small[epf.CANDIDATE_S1_COLUMN]) <= set(whole[epf.CANDIDATE_S1_COLUMN])
    finally:
        fixture.close()


def test_output_directory_defaults_outside_the_candidate_directory():
    """The experiment must never write into outputs/candidates."""
    fixture = _fixture()
    try:
        args = epf.parse_args(["--config", str(fixture.config_path)])
        assert args.output_dir is None
        candidate_dir = Path(fixture.config["resolved"]["candidates_dir"])
        default = candidate_dir.parent / "experiments" / "step3_features"
        assert candidate_dir not in default.parents
        assert default != candidate_dir
    finally:
        fixture.close()


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
