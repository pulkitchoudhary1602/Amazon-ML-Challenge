#!/usr/bin/env python
"""Stage 6 (milestone 2): write ``matching_results.tsv``.  **IMPLEMENTED (V1).**

Turns a trained matcher plus a *test* feature table into the graded artifact: one row
per test S1 entity, in S1 file order, with the selected target ids comma-separated.
That is the shape of ``train/train_ground_truth.tsv`` (``source1_entity_id`` +
``matched_entity_ids``), which is the only format evidence this repository has.

What one run does, in order:

1. **Loads the frozen matcher.** ``load_bundle`` restores the per-fold boosters, the
   feature-column order and the decision threshold that ``scripts/train_model.py``
   picked on out-of-fold validation scores. Nothing is retrained here and the
   threshold is *read*, never re-tuned - an operating point chosen against the test
   split would not be a frozen one, and this script deliberately has no ``--threshold``
   flag for the same reason.
2. **Reads the test S1 population** from the prepared test S1 table. That list is the
   universe the submission must cover. It is also the *only* place the entities no
   blocker proposed a candidate for exist: they have no row in the feature table, so
   nothing inside the feature table can restore them. Their required output row is
   "this entity, no match".
3. **Scores the test feature table** a chunk at a time:
   ``predict`` -> ``decide`` -> ``aggregate_matches``. The last one is the repository's
   own grouping rule, reused rather than reimplemented: every pair at or above the
   threshold is a match (one-to-many by design, there is no top-1 step), and target
   ids are deduplicated and sorted so reruns are byte-identical.
4. **Writes the submission** in S1 file order through ``ChunkWriter``, into a
   ``.partial`` that is atomically replaced only once the run is accounted for, then
   **verifies the result**: exactly one row per test S1 entity, no duplicate ids, and
   every S1 id the feature table scored present in the S1 file.

The accounting, and why it fails loudly
---------------------------------------
Every run reports ``n_s1_input``, ``n_s1_output``, ``n_singletons``, ``n_matched``,
``n_duplicate_s1_ids``, ``n_missing_s1_ids`` and ``n_predictions``, plus the number of
candidate rows read and selected. The script exits non-zero and *does not* publish the
submission if any of the invariants in :func:`check_accounting` fails - a submission
that quietly covers fewer entities than the test set is a zero, and it would look
exactly like a submission that works. ``n_missing_s1_ids`` is the sharp one: it counts
S1 ids the *feature table* scored that the S1 file does not list, i.e. the two inputs
came from different splits.

The output is left as ``<output>.partial`` on failure, so the incomplete file can be
inspected without being mistaken for a submission.

Format caveat (unchanged from the stub this replaces)
-----------------------------------------------------
The challenge's exact ``matching_results.tsv`` format was never provided. This writes
the ground-truth shape because that is the only evidence available, and it logs a
warning saying so on every run. The constraints the format rules do pin down are all
enforced above: every S1 exactly once, no-match entities present as an empty field,
deduplicated ids, deterministic ordering.

Examples::

    # the real test run (the feature table is a distinct artifact from the training one)
    python scripts/predict.py --config configs/config.yaml \
        --features outputs/experiments/step3_features_test/features.tsv

    # a shakedown: reads the first 2M feature rows, writes matching_results_sample.tsv
    python scripts/predict.py --features <test features.tsv> --sample-rows 2000000

    # no LightGBM needed: the same path on the dependency-free threshold arm
    python scripts/predict.py --model-dir outputs/experiments/v1_threshold \
        --features <test features.tsv>

Outputs: ``matching_results.tsv`` (the graded artifact, under ``<work_dir>/submission/``
by default) and ``prediction_report.json`` beside it. A ``--sample-rows`` shakedown
writes ``matching_results_sample.tsv`` and ``prediction_report_sample.json`` instead, so
a partial run can never overwrite - or be mistaken for - the real one.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (  # noqa: E402
    ChunkWriter,
    count_rows,
    describe_environment,
    iter_prepared,
    iter_tsv,
    load_config,
    prepared_path,
)
from src.evaluation import (  # noqa: E402
    CANDIDATE_S1_COLUMN,
    CANDIDATE_TARGET_COLUMN,
)
from src.matching_model import (  # noqa: E402
    DEFAULT_CHUNKSIZE,
    ID_COLUMNS,
    aggregate_matches,
    decide,
    describe_bundle,
    iter_feature_chunks,
    load_bundle,
    predict,
    resolve_feature_columns,
)
from src.utils import (  # noqa: E402
    describe_device,
    encode_entity_id,
    ensure_dir,
    fmt_int,
    log_memory,
    setup_logging,
    write_json,
)

LOG_NAME = "predict"

# The id column of a prepared table (``prepare_data.py`` writes ``entity_id`` first).
PREPARED_ID_COLUMN = "entity_id"

# The submission's column names, used when the config does not name them. These are
# the challenge's own names, taken from the training ground-truth header, and they are
# read from ``columns.gt_source1_id`` / ``columns.gt_matched_ids`` when the config
# declares them - the same keys ``load_ground_truth`` uses, so a rename has to happen
# in one place rather than two.
SUBMISSION_S1_COLUMN = "source1_entity_id"
SUBMISSION_TARGET_COLUMN = "matched_entity_ids"

# The prepared test S1 table, and the submission directory, both relative to work_dir.
PREPARED_SUBDIR = "prepared"
SUBMISSION_DIR_NAME = "submission"
SUBMISSION_FILE_NAME = "matching_results.tsv"
SMOKE_FILE_NAME = "matching_results_sample.tsv"
REPORT_FILE_NAME = "prediction_report.json"
SMOKE_REPORT_FILE_NAME = "prediction_report_sample.json"

# The trained run this script reads by default: scripts/train_model.py's own
# --output-dir default, so the common case needs no flag.
MODEL_SUBDIR = Path("experiments") / "v1"

# Exit codes. 2 and 4/5 mirror scripts/train_model.py; 3 is this script's own
# "the artifact does not account for the test population".
EXIT_INPUT = 2
EXIT_ACCOUNTING = 3
EXIT_MODEL = 4
EXIT_PREDICTION = 5

HOW_TO_BUILD_FEATURES = (
    "Build the TEST feature table first (it is a different file from the training "
    "one, which is why there is no default for --features):\n"
    "  python scripts/generate_candidates.py --name candidate_pairs_test\n"
    "  python scripts/extract_pair_features.py --split test --sample-fraction 1.0 "
    "--entities all --candidates candidate_pairs_test "
    "--output-dir <work_dir>/experiments/step3_features_test"
)

HOW_TO_TRAIN = (
    "Train the matcher first:\n"
    "  python scripts/train_model.py --features <train features.tsv> "
    "--output-dir <work_dir>/experiments/v1"
)

_UNSET = object()


# ---------------------------------------------------------------------------
# Argument parsing and path resolution
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a test feature table with the frozen matcher and write the submission.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--log-level", default="INFO")

    data = parser.add_argument_group("data")
    data.add_argument(
        "--features",
        required=True,
        help="the TEST candidate feature TSV, as written by extract_pair_features.py. "
        "Required on purpose: the training feature table has a different path but the "
        "same file name, so a default here could silently score the wrong one",
    )
    data.add_argument(
        "--model-dir",
        default=None,
        help=f"the trained run directory holding model/model_meta.json. Default: <work_dir>/{MODEL_SUBDIR.as_posix()}",
    )
    data.add_argument(
        "--s1",
        default=None,
        help="the test S1 id table: one row per S1 entity, column 'entity_id'. "
        f"Default: <prepared_dir>/test_source1_norm.tsv",
    )
    data.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help="feature rows per scoring chunk. Bounds peak memory at one chunk's "
        "matrix, never the whole table",
    )
    data.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="read only the first N feature rows - a shakedown of the plumbing, NOT a "
        "submission. The default output name gains a '_sample' suffix and the report is "
        "marked is_smoke",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--output",
        default=None,
        help=f"the submission path. Default: <work_dir>/{SUBMISSION_DIR_NAME}/{SUBMISSION_FILE_NAME}",
    )
    out.add_argument(
        "--report",
        default=None,
        help=f"the run's JSON report. Default: <output directory>/{REPORT_FILE_NAME} "
        f"({SMOKE_REPORT_FILE_NAME} with --sample-rows)",
    )
    return parser.parse_args(argv)


def resolve_model_dir(config: dict, args: argparse.Namespace) -> Path:
    if args.model_dir:
        return Path(args.model_dir).expanduser().resolve()
    return (Path(config["resolved"]["work_dir"]) / MODEL_SUBDIR).resolve()


def resolve_output_path(config: dict, args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()
    name = SMOKE_FILE_NAME if args.sample_rows else SUBMISSION_FILE_NAME
    return (Path(config["resolved"]["work_dir"]) / SUBMISSION_DIR_NAME / name).resolve()


def resolve_report_path(args: argparse.Namespace, output_path: Path) -> Path:
    if args.report:
        return Path(args.report).expanduser().resolve()
    # The smoke report gets its own name for the same reason the smoke submission
    # does: a shakedown must not be able to overwrite the real run's record.
    name = SMOKE_REPORT_FILE_NAME if args.sample_rows else REPORT_FILE_NAME
    return output_path.parent / name


def resolve_s1_source(config: dict, args: argparse.Namespace):
    """``(path, chunks)`` for the test S1 id table.

    The default is the prepared test S1 table, read through the repository's own
    ``iter_prepared`` so the configured prepared format and compression are honoured.
    An explicit ``--s1`` is read as a plain TSV, which is what a hand-made id list (or
    a fixture) is.
    """
    if args.s1:
        path = Path(args.s1).expanduser().resolve()
        return path, None
    return prepared_path(config, "test", "source1"), "prepared"


def iter_s1_ids(path: Path, id_column: str, chunksize: int, config: dict, mode: Optional[str]) -> Iterator[pd.Series]:
    """Stream the S1 id column, one Series per chunk.

    ``mode="prepared"`` means the path came from the config, so the file layout is the
    configured one and ``iter_prepared`` knows how to read it; anything else is a TSV.
    """
    if mode == "prepared":
        for chunk in iter_prepared(config, "test", "source1", columns=[id_column], chunksize=chunksize):
            yield chunk[id_column]
        return
    for chunk in iter_tsv(path, columns=[id_column], chunksize=chunksize):
        yield chunk[id_column]


def read_header(path: Path) -> list[str]:
    """The column names of a TSV, without reading its rows."""
    return list(pd.read_csv(path, sep="\t", nrows=0).columns)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_test_features(
    config: dict,
    bundle,
    features_path: Path,
    chunksize: int,
    sample_rows: Optional[int],
    log: logging.Logger,
) -> tuple[dict[str, Optional[set[str]]], dict[str, int]]:
    """Stream the test feature table through the frozen model.

    Returns ``(pending, counters)``. ``pending`` maps every S1 id the feature table
    scored to its selected target ids, or to ``None`` when its candidate rows all fell
    below the threshold. ``None`` and "absent from the mapping" are different facts and
    the submission needs both: ``None`` is a real entity whose correct row is empty,
    while an id still in ``pending`` after the write pass was scored by the feature
    table but is not in the S1 file - a provenance error, not a row.
    """
    pending: dict[str, Optional[set[str]]] = {}
    n_rows = 0
    n_selected = 0
    n_chunks = 0

    for chunk in iter_feature_chunks(
        features_path,
        columns=list(ID_COLUMNS) + list(bundle.feature_columns),
        chunksize=chunksize,
        sample_rows=sample_rows,
    ):
        # The model's own entry points, in the matcher's own order. ``aggregate_matches``
        # is the documented grouping rule (all pairs >= threshold, deduplicated, sorted);
        # it is called per chunk, and the union below is only needed because one S1
        # entity's rows can straddle a chunk boundary that the function cannot see.
        probabilities = predict(config, bundle, chunk)
        decisions = decide(config, bundle, probabilities)
        grouped = aggregate_matches(
            chunk[CANDIDATE_S1_COLUMN].to_numpy(),
            chunk[CANDIDATE_TARGET_COLUMN].to_numpy(),
            decisions,
        )

        for s1, targets in grouped.items():
            entry = pending.get(s1, _UNSET)
            if entry is _UNSET:
                pending[s1] = set(targets) if targets else None
            elif targets:
                if entry is None:
                    pending[s1] = set(targets)
                else:
                    entry.update(targets)

        n_rows += len(chunk)
        n_selected += int(decisions.sum())
        n_chunks += 1
        if n_chunks % 10 == 0:
            log_memory(log, f"predict chunk {n_chunks}")
            log.info(
                "scored %s chunks: %s candidate rows, %s selected, %s S1 entities seen",
                fmt_int(n_chunks),
                fmt_int(n_rows),
                fmt_int(n_selected),
                fmt_int(len(pending)),
            )

    counters = {
        "n_candidate_rows_scored": n_rows,
        "n_selected_rows": n_selected,
        "n_s1_ids_scored": len(pending),
        "n_s1_ids_scored_but_below_threshold": sum(1 for v in pending.values() if v is None),
        "n_scoring_chunks": n_chunks,
    }
    return pending, counters


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def write_submission(
    output_path: Path,
    universe_chunks: Iterator[pd.Series],
    pending: dict[str, Optional[set[str]]],
    s1_column: str,
    target_column: str,
    chunksize: int,
    log: logging.Logger,
) -> dict[str, Any]:
    """Write one row per S1 entity, in S1 file order, then measure what was written.

    ``pending`` is consumed: every id it still holds at the end was scored by the
    feature table but never appeared in the S1 file. The rows go to
    ``<output>.partial`` and are *not* published - moving the partial into place is
    ``main``'s decision, taken only once the accounting below says the file covers the
    test population. A run that fails its invariants therefore leaves no submission,
    only the partial it can be diagnosed from.
    """
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    ensure_dir(output_path.parent)

    n_input = n_output = n_singletons = n_matched = 0
    n_duplicates = n_predictions = n_bad_targets = 0
    seen: set[str] = set()
    block_s1: list[str] = []
    block_targets: list[str] = []

    def flush(writer: ChunkWriter) -> None:
        if block_s1:
            writer.append(pd.DataFrame({s1_column: block_s1, target_column: block_targets}))
            del block_s1[:]
            del block_targets[:]

    with ChunkWriter(partial) as writer:
        for ids in universe_chunks:
            for value in ids:
                s1 = str(value)
                n_input += 1
                if s1 in seen:
                    n_duplicates += 1
                seen.add(s1)

                targets = pending.pop(s1, _UNSET)
                if targets is _UNSET:
                    # The normal case for an entity no blocker proposed a candidate
                    # for: it has no feature rows at all, and its row is "no match".
                    joined = ""
                else:
                    if targets is None:
                        targets = ()
                    for target in targets:
                        try:
                            encode_entity_id(target)
                        except ValueError:
                            n_bad_targets += 1
                    joined = ",".join(sorted(targets))

                if joined:
                    n_matched += 1
                    n_predictions += len(targets)
                else:
                    n_singletons += 1

                block_s1.append(s1)
                block_targets.append(joined)
                n_output += 1
                if len(block_s1) >= chunksize:
                    flush(writer)
        flush(writer)

    written = n_output > 0
    measured = count_rows(partial) if written else 0

    return {
        "n_s1_input": n_input,
        "n_s1_output": n_output,
        "n_singletons": n_singletons,
        "n_matched": n_matched,
        "n_duplicate_s1_ids": n_duplicates,
        "n_missing_s1_ids": len(pending),
        # The ids the feature table scored that the S1 file does not list, so the
        # failure message can name them instead of only counting them.
        "s1_ids_missing_from_the_s1_file": sorted(pending)[:20],
        "n_predictions": n_predictions,
        "n_target_ids_not_s2_or_s3": n_bad_targets,
        # Measured on the file that was actually written, so a bug in the row loop
        # cannot hide behind the loop's own counter. It is the partial that is
        # measured: publishing is what the accounting gates.
        "n_s1_output_measured": measured,
        "output_path": str(output_path),
        "partial_path": str(partial),
        "written": written,
    }


def check_accounting(accounting: dict[str, Any]) -> dict[str, bool]:
    """The invariants the submission must satisfy, as named booleans.

    Named rather than inlined so a failure can print *which* one broke, and so
    ``tests/test_predict.py`` can assert on the name rather than on an arithmetic
    expression that a future edit could silently keep true.
    """
    return {
        # The required property: one row per test S1 entity, and nothing else.
        "exactly_one_row_per_s1_entity": accounting["n_s1_output"] == accounting["n_s1_input"],
        # The same property re-measured on the file that was actually written, so a
        # bug in the row loop cannot hide behind its own counter.
        "row_count_matches_the_s1_file": accounting["n_s1_output_measured"] == accounting["n_s1_input"],
        "the_s1_file_is_not_empty": accounting["n_s1_input"] > 0,
        "no_duplicate_s1_ids": accounting["n_duplicate_s1_ids"] == 0,
        # Every S1 entity the feature table scored is an entity the submission covers.
        "no_s1_entity_dropped": accounting["n_missing_s1_ids"] == 0,
        "every_row_is_singleton_or_matched": (
            accounting["n_singletons"] + accounting["n_matched"] == accounting["n_s1_output"]
        ),
        "predictions_are_valid_target_ids": accounting["n_target_ids_not_s2_or_s3"] == 0,
        # Nested, not equal: a (S1, target) pair proposed twice collapses to one
        # prediction, so the prediction count can only be below the selected rows.
        "predictions_within_selected_rows": accounting["n_predictions"] <= accounting["n_selected_rows"],
        "selected_within_scored_rows": accounting["n_selected_rows"] <= accounting["n_candidate_rows_scored"],
    }


def accounting_failures(accounting: dict[str, Any]) -> dict[str, bool]:
    """The checks that came out ``False``, for a log line that names them."""
    return {name: ok for name, ok in (accounting.get("checks") or {}).items() if ok is False}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(
        args.config, overrides={"data_root": args.data_root, "work_dir": args.work_dir}
    )
    log = setup_logging(
        LOG_NAME,
        log_dir=config["resolved"]["log_dir"],
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
    )

    features_path = Path(args.features).expanduser().resolve()
    model_dir = resolve_model_dir(config, args)
    output_path = resolve_output_path(config, args)
    report_path = resolve_report_path(args, output_path)
    id_column = config.get("columns", {}).get("entity_id", PREPARED_ID_COLUMN)
    s1_column = config.get("columns", {}).get("gt_source1_id", SUBMISSION_S1_COLUMN)
    target_column = config.get("columns", {}).get("gt_matched_ids", SUBMISSION_TARGET_COLUMN)
    s1_path, s1_mode = resolve_s1_source(config, args)

    log.info("=" * 78)
    log.info("predict: test submission (%s -> %s)", SUBMISSION_FILE_NAME, output_path.name)
    log.info(describe_environment(config))
    describe_device(log, config)
    log.info("=" * 78)
    log.info("features      : %s", features_path)
    log.info("model dir     : %s", model_dir)
    log.info("test S1 table : %s (column %r)", s1_path, id_column)
    log.info("submission    : %s", output_path)
    log.info("report        : %s", report_path)
    log.info("=" * 78)
    log.warning(
        "submission format: writing the ground-truth shape (%s + comma-separated %s). "
        "Confirm it against the challenge brief before submitting - it is the only "
        "format evidence this repository has.",
        s1_column,
        target_column,
    )
    if args.sample_rows:
        log.warning(
            "SAMPLE ROWS: %s - a shakedown of the plumbing, NOT a submission. The rows "
            "beyond this prefix are scored nowhere and the report is marked is_smoke",
            fmt_int(args.sample_rows),
        )

    # ---- inputs, checked in the order that fails cheapest first -------------
    if not features_path.is_file():
        log.error("feature file not found: %s", features_path)
        log.error("%s", HOW_TO_BUILD_FEATURES)
        return EXIT_INPUT
    if not s1_path.is_file():
        log.error("test S1 table not found: %s", s1_path)
        log.error("  run `python scripts/prepare_data.py` (it writes the test split too), "
                  "or pass --s1 with an explicit id list")
        return EXIT_INPUT

    try:
        header = read_header(features_path)
    except pd.errors.EmptyDataError:
        log.error("the feature file is empty (no header): %s", features_path)
        log.error("%s", HOW_TO_BUILD_FEATURES)
        return EXIT_INPUT
    try:
        resolve_feature_columns(header)
    except ValueError as exc:
        log.error("the feature file does not match the matcher's schema: %s", exc)
        return EXIT_INPUT
    if CANDIDATE_S1_COLUMN not in header or CANDIDATE_TARGET_COLUMN not in header:
        log.error(
            "the feature file has no %r / %r column, so its rows cannot be mapped back "
            "to entities: %s",
            CANDIDATE_S1_COLUMN,
            CANDIDATE_TARGET_COLUMN,
            features_path,
        )
        return EXIT_INPUT

    # ---- the frozen model ---------------------------------------------------
    try:
        bundle = load_bundle(model_dir)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        log.error("%s", HOW_TO_TRAIN)
        return EXIT_MODEL
    except ImportError as exc:
        log.error("%s", exc)
        return EXIT_MODEL
    except (ValueError, TypeError, KeyError) as exc:
        # An unreadable model_meta.json is a ValueError (json.JSONDecodeError is one),
        # a null threshold is a TypeError, a missing key is a KeyError. All three mean
        # "this is not a usable model", and none of them may be scored through.
        log.error("the saved model at %s is not usable: %s: %s", model_dir, type(exc).__name__, exc)
        log.error("%s", HOW_TO_TRAIN)
        return EXIT_MODEL

    missing_features = [c for c in bundle.feature_columns if c not in set(header)]
    if missing_features:
        log.error(
            "the model at %s was trained on columns this feature file does not have: %s",
            model_dir,
            missing_features,
        )
        return EXIT_MODEL
    if not bundle.is_ready() or not np.isfinite(bundle.threshold):
        log.error(
            "the saved model at %s cannot score pairs: %s", model_dir, describe_bundle(bundle)
        )
        log.error("%s", HOW_TO_TRAIN)
        return EXIT_MODEL

    log.info("model         : %s", describe_bundle(bundle))
    log.info(
        "threshold     : %.6g (frozen - read from model/model_meta.json, never re-tuned here)",
        bundle.threshold,
    )

    started = time.time()

    # ---- score --------------------------------------------------------------
    try:
        pending, scoring = score_test_features(
            config, bundle, features_path, args.chunksize, args.sample_rows, log
        )
    except ValueError as exc:
        log.error("scoring failed: %s", exc)
        return EXIT_PREDICTION
    except ImportError as exc:  # pragma: no cover - raised by lightgbm itself
        log.error("%s", exc)
        return EXIT_MODEL

    log.info(
        "scored %s candidate rows over %s chunks: %s at/above the threshold, %s S1 "
        "entities scored, %s of those with nothing selected",
        fmt_int(scoring["n_candidate_rows_scored"]),
        fmt_int(scoring["n_scoring_chunks"]),
        fmt_int(scoring["n_selected_rows"]),
        fmt_int(scoring["n_s1_ids_scored"]),
        fmt_int(scoring["n_s1_ids_scored_but_below_threshold"]),
    )

    # ---- write, then verify -------------------------------------------------
    try:
        # ``mode == "prepared"`` means the path came from the config, so the file
        # layout is the configured one and ``iter_prepared`` knows how to read it.
        universe = iter_s1_ids(s1_path, id_column, args.chunksize, config, s1_mode)
        written = write_submission(
            output_path, universe, pending, s1_column, target_column, args.chunksize, log
        )
    except ValueError as exc:
        log.error("could not read the test S1 table %s: %s", s1_path, exc)
        return EXIT_INPUT

    accounting: dict[str, Any] = {**scoring, **written}
    accounting["checks"] = check_accounting(accounting)
    accounting["ok"] = all(accounting["checks"].values())
    elapsed = time.time() - started

    # Publish only what the accounting has vouched for. A submission that covers fewer
    # entities than the test set is a zero, and it would look exactly like one that
    # works, so the failed run is the one that leaves no submission behind.
    if accounting["ok"]:
        os.replace(accounting["partial_path"], accounting["output_path"])
    accounting["published"] = accounting["ok"]

    log.info("-" * 78)
    log.info("S1 accounting")
    log.info("  n_s1_input                     : %s", fmt_int(accounting["n_s1_input"]))
    log.info("  n_s1_output                    : %s", fmt_int(accounting["n_s1_output"]))
    log.info("  n_singletons                   : %s", fmt_int(accounting["n_singletons"]))
    log.info("  n_matched                      : %s", fmt_int(accounting["n_matched"]))
    log.info("  n_duplicate_s1_ids             : %s", fmt_int(accounting["n_duplicate_s1_ids"]))
    log.info("  n_missing_s1_ids               : %s", fmt_int(accounting["n_missing_s1_ids"]))
    log.info("  n_predictions                  : %s", fmt_int(accounting["n_predictions"]))
    log.info(
        "  n_s1_ids_scored                : %s (%s with candidate rows and nothing selected)",
        fmt_int(accounting["n_s1_ids_scored"]),
        fmt_int(accounting["n_s1_ids_scored_but_below_threshold"]),
    )
    log.info(
        "  n_candidate_rows_scored        : %s (%s selected)",
        fmt_int(accounting["n_candidate_rows_scored"]),
        fmt_int(accounting["n_selected_rows"]),
    )
    log.info(
        "  mean predictions per matched S1: %.2f",
        (accounting["n_predictions"] / accounting["n_matched"]) if accounting["n_matched"] else 0.0,
    )
    log.info("-" * 78)

    payload = {
        "generated_by": "scripts/predict.py",
        "is_smoke": bool(args.sample_rows),
        "sample_rows": args.sample_rows,
        "inputs": {
            "features": str(features_path),
            "model_dir": str(model_dir),
            "s1_table": str(s1_path),
            "s1_id_column": id_column,
            "submission_columns": [s1_column, target_column],
            "output": str(output_path),
        },
        "model": bundle.summary(),
        "threshold": bundle.threshold,
        "threshold_source": "model/model_meta.json - frozen, this script has no way to re-tune it",
        "submission_format": {
            "columns": [s1_column, target_column],
            "empty_match": "empty field",
            "target_id_order": "deduplicated, sorted (matching_model.aggregate_matches)",
            "row_order": "test S1 file order",
            "evidence": "the header and rows of train/train_ground_truth.tsv",
            "confirmed_against_the_brief": False,
        },
        "accounting": {k: v for k, v in accounting.items() if k != "checks"},
        "checks": accounting["checks"],
        "ok": accounting["ok"],
        "seconds": round(elapsed, 1),
    }
    write_json(report_path, payload)
    log.info("report written: %s", report_path)

    if not accounting["ok"]:
        log.error(
            "ACCOUNTING FAILURE: the submission does not cover the test S1 population "
            "exactly once. %s",
            accounting_failures(accounting),
        )
        if accounting["n_missing_s1_ids"]:
            log.error(
                "%s S1 ids in the feature table are not in %s (first few: %s) - the two "
                "inputs are not from the same split.",
                fmt_int(accounting["n_missing_s1_ids"]),
                s1_path,
                accounting["s1_ids_missing_from_the_s1_file"],
            )
        log.error(
            "the incomplete file is left at %s (nothing was published to %s)",
            accounting["partial_path"],
            accounting["output_path"],
        )
        return EXIT_ACCOUNTING

    log.info(
        "done: %s rows, %s matched / %s singletons, %s predictions -> %s (%.1f s)",
        fmt_int(accounting["n_s1_output"]),
        fmt_int(accounting["n_matched"]),
        fmt_int(accounting["n_singletons"]),
        fmt_int(accounting["n_predictions"]),
        accounting["output_path"],
        elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
