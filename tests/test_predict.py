"""Submission tests: one row per test S1 entity, and the accounting that proves it.

What is being guarded, and why each guard exists
------------------------------------------------
``scripts/predict.py`` writes the file that is actually graded, and its failure mode is
*quiet*: an S1 entity that never gets a row, or a match that gets dropped on the way
out, produces a smaller file that still parses. The challenge metric is a macro average
over S1 entities, so a missing entity and a wrong answer cost the same thing - and
nothing downstream would notice the difference. Every test here pins one of the ways
that can happen.

The fixture is small enough to verify by hand. Its four test S1 entities are picked so
that one entity each covers the cases that actually break the mapping:

* ``S1-1`` has **four** candidate rows, three at/above the threshold (two of them the
  same ``(S1, target)`` pair) and one below, so its row must be exactly
  ``S2-1,S2-2``: multiple selected candidates kept, one duplicate collapsed, one
  rejection dropped, and the output order deterministic rather than insertion order.
* ``S1-2`` has one row above the threshold - the ordinary matched entity.
* ``S1-3`` has two rows and **both** fall below the threshold, so it is a real entity
  whose candidates were all rejected. Its required row is "no match" - not an omission,
  which is what silently dropping it would look like.
* ``S1-4`` has **no candidate rows at all**, so it exists only in the S1 file. No
  amount of care inside the feature table can produce its row; it is the case the
  ground truth's 123,247 empty lists correspond to.

Two more shapes the fixture is built to hit, both of which are load-bearing:

* ``--chunksize 2`` puts a chunk boundary through ``S1-1``, so its rows are scored in
  two different chunks and its selected targets have to be unioned across them.
* the model is the dependency-free ``threshold`` arm, so the whole path is testable
  without LightGBM (the ``lightgbm`` arm is tested separately, and skipped when the
  package is missing).

The end-to-end test asserts the submission's exact lines *and* the full accounting, so
"the row exists" and "the row is right" are both pinned - a test that only counted rows
would pass on a file full of empty match lists.

Runs standalone (``python tests/test_predict.py``) and under pytest.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.predict as predict_script  # noqa: E402

from src.evaluation import (  # noqa: E402
    CANDIDATE_S1_COLUMN,
    CANDIDATE_TARGET_COLUMN,
)
from src.matching_model import (  # noqa: E402
    FEATURE_COLUMNS,
    ID_COLUMNS,
    MODEL_THRESHOLD,
    NON_FEATURE_COLUMNS,
    ModelBundle,
    load_bundle,
    save_bundle,
)

_TEMP_DIRS: list[Path] = []

# The threshold arm scores ``name_token_set_ratio`` directly, so the fixture's
# above/below split is decided by this one column and nothing else.
SCORE_FEATURE = "name_token_set_ratio"
THRESHOLD = 0.5

# The test S1 population, in file order. S1-4 is the no-candidate entity.
TEST_S1_IDS = ["S1-1", "S1-2", "S1-3", "S1-4"]

# (s1, target, source, name_token_set_ratio)
FEATURE_ROWS = [
    ("S1-1", "S2-1", "S2", 0.90),  # selected
    ("S1-1", "S2-2", "S3", 0.80),  # selected
    # ---- chunk boundary (chunksize 2) ----
    ("S1-1", "S2-1", "S2", 0.95),  # selected again: same pair, must collapse
    ("S1-1", "S2-9", "S2", 0.10),  # rejected
    # ---- chunk boundary ----
    ("S1-2", "S3-1", "S3", 0.70),  # selected
    ("S1-3", "S2-3", "S2", 0.20),  # rejected
    ("S1-3", "S3-3", "S3", 0.40),  # rejected
]

EXPECTED_MATCHES = {
    "S1-1": "S2-1,S2-2",
    "S1-2": "S3-1",
    "S1-3": "",
    "S1-4": "",
}


def _temp_dir() -> Path:
    directory = Path(tempfile.mkdtemp(prefix="predict_test_"))
    _TEMP_DIRS.append(directory)
    return directory


def _write_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join("" if value is None else str(value) for value in row) + "\n")


def _has_lightgbm() -> bool:
    return importlib.util.find_spec("lightgbm") is not None


def _expect_argparse_exit(argv: list[str], why: str) -> None:
    """Assert argparse rejects ``argv``. Raises AssertionError (not SystemExit) if not.

    argparse prints its usage to stderr before exiting, which is noise in this
    runner's output rather than a failure, so it is swallowed here.
    """
    import contextlib
    import io

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            predict_script.parse_args(argv)
    except SystemExit:
        return
    raise AssertionError(why)


def _feature_header() -> list[str]:
    return list(ID_COLUMNS) + list(FEATURE_COLUMNS) + list(NON_FEATURE_COLUMNS)


def _feature_row(s1: str, target: str, source: str, ratio: float) -> list:
    """One feature row, every feature zero except the one the threshold arm scores.

    A zero is not the same as a blank here: this arm reads the column directly, so the
    fixture only has to make its value unambiguous.
    """
    values = {name: 0 for name in FEATURE_COLUMNS}
    values[SCORE_FEATURE] = ratio
    values["source_is_s2"] = 1 if source == "S2" else 0
    return [s1, target, source] + [values[name] for name in FEATURE_COLUMNS] + [1]


class _Fixture:
    """A temp tree with a feature table, a test S1 table and a saved threshold model."""

    def __init__(self) -> None:
        self.root = _temp_dir()
        self.prepared = self.root / "prepared"
        self.candidates = self.root / "candidates"
        self.run_dir = self.root / "experiments" / "v1"
        self.out = self.root / "submission"
        for path in (self.prepared, self.candidates, self.out):
            path.mkdir(parents=True, exist_ok=True)

        self.features_path = self.root / "features.tsv"
        _write_tsv(
            self.features_path,
            _feature_header(),
            [_feature_row(*row) for row in FEATURE_ROWS],
        )

        # The prepared test S1 table, so the default --s1 path is the one under test.
        # It carries the full prepared schema: predict must read only the id column out
        # of a table that has text columns too.
        _write_tsv(
            self.prepared / "test_source1_norm.tsv",
            ["entity_id", "business_name", "business_address", "country", "name_norm", "name_key"],
            [[s1, f"business {s1}", "1 main st", "in", f"business {s1}", f"business{s1}"]
             for s1 in TEST_S1_IDS],
        )

        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(
            "\n".join(
                [
                    "project: {name: er, seed: 42}",
                    f"paths: {{data_root: '{(self.root / 'raw').as_posix()}', "
                    f"test_data_root: '{(self.root / 'raw').as_posix()}', "
                    f"work_dir: '{(self.root / 'work').as_posix()}', "
                    f"prepared_dir: '{self.prepared.as_posix()}', "
                    f"index_dir: '{(self.root / 'indexes').as_posix()}', "
                    f"candidates_dir: '{self.candidates.as_posix()}', "
                    f"log_dir: '{(self.root / 'logs').as_posix()}'}}",
                    "io: {chunksize: 2, prepared_format: tsv, candidates_format: tsv}",
                    "columns: {entity_id: entity_id, gt_source1_id: source1_entity_id, "
                    "gt_matched_ids: matched_entity_ids}",
                ]
            ),
            encoding="utf-8",
        )

        save_bundle(
            ModelBundle(
                model=MODEL_THRESHOLD,
                feature_columns=tuple(FEATURE_COLUMNS),
                threshold=THRESHOLD,
                score_feature=SCORE_FEATURE,
            ),
            self.run_dir,
        )

        self.output_path = self.out / "matching_results.tsv"

    @property
    def report_path(self) -> Path:
        # Read through the script's own constants: a rename there must show up here as
        # a missing file, not as a test that quietly reads the wrong artifact.
        return self.out / predict_script.REPORT_FILE_NAME

    @property
    def smoke_report_path(self) -> Path:
        return self.out / predict_script.SMOKE_REPORT_FILE_NAME

    # -- command lines -----------------------------------------------------
    def argv(self, **overrides) -> list[str]:
        """A command line for this fixture.

        Built from a value map rather than a fixed list plus appended overrides, so an
        override *replaces* the default instead of silently losing to it (argparse takes
        the last occurrence, which would make ``output=None`` unable to switch the
        default path resolution on). ``None``/``False`` drop the flag entirely, and
        ``True`` turns it into a bare flag.
        """
        values: dict[str, Any] = {
            "config": self.config_path,
            "features": self.features_path,
            "model-dir": self.run_dir,
            "output": self.output_path,
            # 2 puts a chunk boundary through S1-1 in the default run, so the
            # cross-chunk union is exercised by every test rather than by one.
            "chunksize": 2,
            "log-level": "CRITICAL",
        }
        values.update({key.replace("_", "-"): value for key, value in overrides.items()})

        argv: list[str] = []
        for flag, value in values.items():
            name = f"--{flag}"
            if value is True:
                argv.append(name)
            elif value is False or value is None:
                continue
            else:
                argv += [name, str(value)]
        return argv

    def run(self, **overrides) -> int:
        return predict_script.main(self.argv(**overrides))

    # -- results -----------------------------------------------------------
    @property
    def report(self) -> dict:
        return json.loads(self.report_path.read_text(encoding="utf-8"))

    @property
    def lines(self) -> list[str]:
        return self.output_path.read_text(encoding="utf-8").splitlines()

    def rows(self) -> dict[str, str]:
        """The submission as an id -> match-list mapping, header dropped."""
        out: dict[str, str] = {}
        for line in self.lines[1:]:
            s1, _, targets = line.partition("\t")
            out[s1] = targets
        return out


# ---------------------------------------------------------------------------
# The fixture's own assumption
# ---------------------------------------------------------------------------
def test_fixture_rows_split_exactly_at_the_configured_threshold():
    """Guards the fixture, not the script: every case must be unambiguous.

    If a ratio ever drifted onto the threshold, ``decide`` (``>=``) would still be
    deterministic but the fixture would no longer be testing what its comments say.
    """
    for _s1, _target, _source, ratio in FEATURE_ROWS:
        assert ratio != THRESHOLD, f"ratio {ratio} sits exactly on the threshold"
    above = [row for row in FEATURE_ROWS if row[3] >= THRESHOLD]
    assert len(above) == 4, above
    assert len(set(TEST_S1_IDS)) == len(TEST_S1_IDS)
    assert {row[0] for row in FEATURE_ROWS} < set(TEST_S1_IDS), (
        "every S1 the feature table scores must also be in the S1 file, or the "
        "provenance check fires"
    )


# ---------------------------------------------------------------------------
# The happy path, end to end through the CLI
# ---------------------------------------------------------------------------
def test_submission_has_one_row_per_s1_entity_in_file_order():
    """The exact bytes of the graded artifact, including the two empty rows."""
    fixture = _Fixture()
    assert fixture.run() == 0

    assert fixture.lines == [
        "source1_entity_id\tmatched_entity_ids",
        "S1-1\tS2-1,S2-2",
        "S1-2\tS3-1",
        "S1-3\t",
        "S1-4\t",
    ], fixture.lines
    # No quoting: the id list is the field, and a quoted list would not be the shape
    # the ground truth uses.
    assert '"' not in fixture.output_path.read_text(encoding="utf-8")


def test_matched_singleton_rejection_and_no_candidate_entity():
    """One assertion per case the submission format has to get right."""
    fixture = _Fixture()
    assert fixture.run() == 0
    rows = fixture.rows()

    # multiple selected candidates, deduplicated, sorted - not insertion order
    assert rows["S1-1"] == "S2-1,S2-2"
    # the ordinary matched entity
    assert rows["S1-2"] == "S3-1"
    # every candidate rejected: a real entity with an empty row, not an omission
    assert rows["S1-3"] == ""
    # no candidate rows at all: it exists only in the S1 file
    assert rows["S1-4"] == ""
    # every test S1 entity, exactly once, in the order the S1 file lists them
    assert list(rows) == TEST_S1_IDS


def test_the_decision_threshold_is_the_one_frozen_in_the_bundle():
    """The operating point is the artifact's, not a constant in this script.

    Raising the saved threshold above the weaker rows must turn those entities into
    *singletons* - entities kept, matches gone. That is the shape a stricter frozen
    threshold is supposed to have, and it is why the earlier test's "S1-3 is empty"
    cannot be an accident of a hard-coded cutoff.
    """
    fixture = _Fixture()
    save_bundle(
        ModelBundle(
            model=MODEL_THRESHOLD,
            feature_columns=tuple(FEATURE_COLUMNS),
            threshold=0.93,
            score_feature=SCORE_FEATURE,
        ),
        fixture.run_dir,
    )

    assert fixture.run() == 0
    rows = fixture.rows()
    assert rows["S1-1"] == "S2-1", "only the 0.95 row is above 0.93"
    assert rows["S1-2"] == "", "0.70 is below the raised threshold"
    # the population is untouched by the stricter operating point
    assert list(rows) == TEST_S1_IDS
    assert fixture.report["ok"] is True
    assert fixture.report["accounting"]["n_matched"] == 1
    assert fixture.report["accounting"]["n_singletons"] == 3


def test_there_is_no_way_to_pass_a_threshold():
    """Structural, not stylistic: the threshold may only come from the artifact."""
    fixture = _Fixture()
    _expect_argparse_exit(
        fixture.argv(threshold=0.1),
        "predict.py must not accept --threshold: the operating point is frozen by "
        "train_model.py and read from model/model_meta.json",
    )


def test_accounting_covers_the_test_population_and_reports_the_dedup():
    fixture = _Fixture()
    assert fixture.run() == 0
    report = fixture.report
    accounting = report["accounting"]

    assert accounting["n_s1_input"] == 4
    assert accounting["n_s1_output"] == 4
    assert accounting["n_s1_output_measured"] == 4
    assert accounting["n_singletons"] == 2
    assert accounting["n_matched"] == 2
    assert accounting["n_duplicate_s1_ids"] == 0
    assert accounting["n_missing_s1_ids"] == 0
    # S2-1, S2-2 (S1-1) + S3-1 (S1-2) = 3 written pairs out of 4 selected rows: the
    # repeated (S1-1, S2-1) pair is one prediction, not two.
    assert accounting["n_predictions"] == 3
    assert accounting["n_selected_rows"] == 4
    assert accounting["n_candidate_rows_scored"] == len(FEATURE_ROWS) == 7
    assert accounting["n_s1_ids_scored"] == 3
    assert accounting["n_s1_ids_scored_but_below_threshold"] == 1  # S1-3
    assert accounting["n_target_ids_not_s2_or_s3"] == 0

    assert report["ok"] is True
    assert all(report["checks"].values()), report["checks"]
    assert report["threshold"] == THRESHOLD
    assert report["is_smoke"] is False
    assert report["inputs"]["model_dir"] == str(fixture.run_dir)
    assert predict_script.accounting_failures(accounting) == {}


def test_a_chunk_boundary_through_one_entity_keeps_all_of_its_matches():
    """S1-1's rows are split across chunks by the fixture's ``chunksize: 2``.

    Scoring is per chunk, so this is the case where a naive implementation would keep
    only the targets of whichever chunk happened to come last.
    """
    fixture = _Fixture()
    assert predict_script.parse_args(fixture.argv()).chunksize == 2
    assert fixture.run() == 0
    assert fixture.rows()["S1-1"] == "S2-1,S2-2"


def test_one_chunk_at_a_time_gives_the_same_submission():
    """The chunk size is a memory knob, not a policy knob."""
    sizes = []
    for chunksize in (1, 2, 3, 7, 1000):
        fixture = _Fixture()
        assert fixture.run(chunksize=chunksize) == 0
        sizes.append(fixture.lines)
    for other in sizes[1:]:
        assert other == sizes[0], f"chunksize changed the submission:\n{sizes[0]}\n{other}"


# ---------------------------------------------------------------------------
# Fail loudly
# ---------------------------------------------------------------------------
def test_a_scored_entity_missing_from_the_s1_file_is_an_accounting_failure():
    """The two inputs not being from the same split must stop the run.

    The feature table references ``S1-9``, which the S1 file does not list. Writing the
    run out anyway would publish a submission that silently omits every such entity -
    the exact failure this script exists to prevent.
    """
    fixture = _Fixture()
    _write_tsv(
        fixture.features_path,
        _feature_header(),
        [_feature_row(*row) for row in FEATURE_ROWS]
        + [_feature_row("S1-9", "S2-7", "S2", 0.99)],
    )

    assert fixture.run() == predict_script.EXIT_ACCOUNTING
    report = fixture.report
    assert report["ok"] is False
    assert report["checks"]["no_s1_entity_dropped"] is False
    assert report["accounting"]["n_missing_s1_ids"] == 1
    assert report["accounting"]["s1_ids_missing_from_the_s1_file"] == ["S1-9"]
    # the row count itself is still right, which is why the id check is separate
    assert report["checks"]["exactly_one_row_per_s1_entity"] is True
    # nothing was published, and the incomplete file is left for inspection
    assert not fixture.output_path.exists()
    assert (fixture.out / "matching_results.tsv.partial").exists()


def test_duplicate_ids_in_the_s1_file_are_an_accounting_failure():
    """An entity listed twice would produce two rows for one entity."""
    fixture = _Fixture()
    _write_tsv(
        fixture.prepared / "test_source1_norm.tsv",
        ["entity_id", "business_name"],
        [[s1, f"business {s1}"] for s1 in TEST_S1_IDS + ["S1-1"]],
    )

    assert fixture.run() == predict_script.EXIT_ACCOUNTING
    report = fixture.report
    assert report["accounting"]["n_duplicate_s1_ids"] == 1
    assert report["accounting"]["n_s1_input"] == 5
    assert report["checks"]["no_duplicate_s1_ids"] is False
    assert report["checks"]["exactly_one_row_per_s1_entity"] is True
    assert not fixture.output_path.exists()


def test_an_empty_s1_table_is_an_accounting_failure():
    """Zero rows "covers" zero entities, which must not read as success."""
    fixture = _Fixture()
    _write_tsv(fixture.prepared / "test_source1_norm.tsv", ["entity_id", "business_name"], [])

    assert fixture.run() == predict_script.EXIT_ACCOUNTING
    report = fixture.report
    assert report["accounting"]["n_s1_input"] == 0
    assert report["checks"]["the_s1_file_is_not_empty"] is False
    assert not fixture.output_path.exists()


def test_accounting_failures_names_the_broken_check():
    """The failure line has to say *which* invariant broke, not just that one did."""
    healthy = {
        "n_s1_input": 4, "n_s1_output": 4, "n_s1_output_measured": 4,
        "n_singletons": 2, "n_matched": 2, "n_duplicate_s1_ids": 0,
        "n_missing_s1_ids": 0, "n_predictions": 3, "n_selected_rows": 4,
        "n_candidate_rows_scored": 7, "n_target_ids_not_s2_or_s3": 0,
    }
    checks = predict_script.check_accounting(healthy)
    assert all(checks.values()), checks
    assert predict_script.accounting_failures({"checks": checks}) == {}

    broken = dict(healthy, n_missing_s1_ids=5)
    checks = predict_script.check_accounting(broken)
    assert checks["no_s1_entity_dropped"] is False
    assert predict_script.accounting_failures({"checks": checks}) == {"no_s1_entity_dropped": False}


def test_a_missing_model_is_not_silently_scored_with_nothing():
    fixture = _Fixture()
    shutil.rmtree(fixture.run_dir)
    assert fixture.run() == predict_script.EXIT_MODEL
    assert not fixture.output_path.exists()


def test_a_model_without_a_threshold_is_refused():
    """The threshold is frozen *from the bundle*; an unset one has no operating point."""
    fixture = _Fixture()
    meta_path = fixture.run_dir / "model" / "model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["threshold"] = None
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    assert fixture.run() == predict_script.EXIT_MODEL
    assert not fixture.output_path.exists()


def test_missing_inputs_exit_without_writing_anything():
    fixture = _Fixture()
    fixture.features_path.unlink()
    assert fixture.run() == predict_script.EXIT_INPUT
    assert not fixture.output_path.exists()

    fixture = _Fixture()
    (fixture.prepared / "test_source1_norm.tsv").unlink()
    assert fixture.run() == predict_script.EXIT_INPUT
    assert not fixture.output_path.exists()


def test_a_feature_file_the_matcher_does_not_know_is_refused():
    """Schema drift must fail here, not reach the model as a shorter matrix."""
    fixture = _Fixture()
    header = _feature_header() + ["surprise_feature"]
    rows = [
        _feature_row(*row) + [1]
        for row in FEATURE_ROWS
    ]
    _write_tsv(fixture.features_path, header, rows)

    assert fixture.run() == predict_script.EXIT_INPUT
    assert not fixture.output_path.exists()


# ---------------------------------------------------------------------------
# Paths, flags and the smoke run
# ---------------------------------------------------------------------------
def test_features_is_required_and_has_no_default():
    """The train and test feature tables share a file name but not a path."""
    _expect_argparse_exit(
        [],
        "--features must be required: there is no safe default for the test feature "
        "table, and a default could silently score the training one",
    )


def test_default_output_is_under_work_dir_and_smoke_gets_its_own_name():
    """A shakedown must not be able to land on the submission's file name."""
    fixture = _Fixture()
    config = predict_script.load_config(str(fixture.config_path))

    args = predict_script.parse_args(fixture.argv(output=None))
    default = predict_script.resolve_output_path(config, args)
    assert default == (fixture.root / "work" / "submission" / "matching_results.tsv").resolve()

    smoke_args = predict_script.parse_args(
        fixture.argv(output=None, sample_rows=3)
    )
    smoke = predict_script.resolve_output_path(config, smoke_args)
    assert smoke.name == "matching_results_sample.tsv"
    assert smoke != default
    # and the record of the shakedown cannot overwrite the real run's either
    assert predict_script.resolve_report_path(smoke_args, smoke) != predict_script.resolve_report_path(
        args, default
    )


def test_sample_rows_marks_the_report_and_still_covers_every_entity():
    """``--sample-rows`` truncates the *scoring*, never the population.

    The submission it writes is a real one for the rows it read - every test entity
    still gets exactly one row - which is what makes it a useful shakedown, and the
    report says is_smoke so it cannot be mistaken for the real thing.
    """
    fixture = _Fixture()
    assert fixture.run(sample_rows=3) == 0
    report = json.loads(fixture.smoke_report_path.read_text(encoding="utf-8"))
    assert not fixture.report_path.exists(), "a shakedown must not write the real report"
    assert report["is_smoke"] is True
    assert report["sample_rows"] == 3
    assert report["accounting"]["n_candidate_rows_scored"] == 3
    assert report["accounting"]["n_s1_input"] == 4
    assert sorted(fixture.rows()) == TEST_S1_IDS
    assert report["ok"] is True


def test_an_explicit_s1_table_is_read_as_a_plain_tsv():
    """``--s1`` is the escape hatch for a hand-made id list (and for fixtures)."""
    fixture = _Fixture()
    explicit = fixture.root / "s1_ids_only.tsv"
    _write_tsv(explicit, ["entity_id"], [[s1] for s1 in TEST_S1_IDS])

    assert fixture.run(s1=explicit) == 0
    assert fixture.lines[1:] == [
        "S1-1\tS2-1,S2-2",
        "S1-2\tS3-1",
        "S1-3\t",
        "S1-4\t",
    ]


def test_reruns_are_byte_identical():
    """The submission is regenerated on every run; it must not drift."""
    first = _Fixture()
    assert first.run() == 0
    before = first.output_path.read_bytes()
    assert first.run() == 0
    assert first.output_path.read_bytes() == before


# ---------------------------------------------------------------------------
# The production arm: a real LightGBM bundle round-trips through save/load
# ---------------------------------------------------------------------------
def test_a_lightgbm_bundle_scores_and_writes_a_covered_submission():
    """The real arm, not the dependency-free one.

    What is asserted is structural, not numeric: which pairs a 10-round model keeps is
    a property of the model, but *that* every entity is covered exactly once is a
    property of this script, and that is what is being tested.
    """
    if not _has_lightgbm():  # pragma: no cover - environment dependent
        return

    import lightgbm

    from src.matching_model import default_params

    fixture = _Fixture()
    # Train a booster on the fixture's own rows, then save it the way train_model.py
    # does, so load_bundle and the model_meta.json contract are exercised too.
    ratios = np.array([row[3] for row in FEATURE_ROWS], dtype=np.float32)
    labels = (ratios >= THRESHOLD).astype(np.float32)
    matrix = np.zeros((len(ratios), len(FEATURE_COLUMNS)), dtype=np.float32)
    matrix[:, list(FEATURE_COLUMNS).index(SCORE_FEATURE)] = ratios
    booster = lightgbm.train(
        default_params(seed=42, num_threads=1),
        lightgbm.Dataset(matrix, label=labels, feature_name=list(FEATURE_COLUMNS)),
        num_boost_round=10,
    )
    save_bundle(
        ModelBundle(
            feature_columns=tuple(FEATURE_COLUMNS),
            threshold=THRESHOLD,
            score_feature=SCORE_FEATURE,
            boosters=[booster],
        ),
        fixture.run_dir,
    )
    assert load_bundle(fixture.run_dir).is_ready()

    assert fixture.run() == 0
    report = fixture.report
    assert report["model"]["model"] == "lightgbm"
    assert report["model"]["n_boosters"] == 1
    assert report["ok"] is True
    assert list(fixture.rows()) == TEST_S1_IDS
    # the model must actually keep something on a fixture this separable
    assert report["accounting"]["n_matched"] >= 1
    assert report["accounting"]["n_s1_output_measured"] == 4


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _main() -> int:
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
