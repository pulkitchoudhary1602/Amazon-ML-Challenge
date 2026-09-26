#!/usr/bin/env python
"""Read-only A/B comparison of two completed V1 matcher runs.

Scores the *same saved artifacts* of two runs at *one* threshold and reports the
challenge metric for each, so an experiment arm can be compared against the frozen
baseline at the baseline's own operating point - not at whatever threshold its own
sweep would have chosen. Nothing is retrained, no model is loaded, no feature file
is read: this only re-reads ``oof_probabilities.npy`` and the label artifacts beside
it.

This exists because ``train_model.py`` has no ``--threshold`` input: it always tunes
its own operating point. So "same threshold, new candidates" is not expressible as a
training flag, but it *is* expressible as one pass over two saved runs.

Everything metric-shaped is delegated to the repository's own definitions -
``src.matching_model.evaluate_at_threshold`` (which calls ``macro_f05``, which calls
``CandidateEvaluation._macro_f05``), ``src.evaluation.split_mask_for`` for the val
population, and ``GroundTruth.lengths()`` for the per-entity true-match counts. This
script introduces no second definition of a true pair or of F0.5.

    # B at A's chosen threshold (the default: the baseline defines the operating point)
    python scripts/compare_v1_runs.py \
        --config configs/config.yaml \
        --baseline outputs/experiments/v1 \
        --candidate outputs/experiments/v1_suffix_stripped

    # both at one threshold pinned by hand
    python scripts/compare_v1_runs.py --config configs/config.yaml \
        --baseline outputs/experiments/v1 \
        --candidate outputs/experiments/v1_suffix_stripped \
        --threshold 0.53

Writes ``v1_ab_comparison.json`` next to the candidate run (or to ``--output``).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_config, load_ground_truth  # noqa: E402
from src.evaluation import split_mask_for  # noqa: E402
from src.matching_model import (  # noqa: E402
    OTHER_POLICY,
    ZERO_MATCH_POLICY,
    evaluate_at_threshold,
)
from src.utils import (  # noqa: E402
    ensure_dir,
    fmt_int,
    read_json,
    setup_logging,
    write_json,
)

LOG_NAME = "compare_v1_runs"

# The artifacts every completed run leaves behind, all row-aligned with each other.
PROBABILITIES_FILE = "oof_probabilities.npy"
LABELS_FILE = "val_labels.npy"
OWNERS_FILE = "val_owner_index.npy"
METRICS_FILE = "v1_metrics.json"

# The metrics `evaluate_at_threshold` reports per policy, and where the sweep parks
# its choice. Read by name so a rename upstream fails loudly here rather than
# silently comparing the wrong column.
POLICY_COLUMNS = {
    ZERO_MATCH_POLICY: "macro_f05_score_zero",
    OTHER_POLICY: "macro_f05_exclude",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two completed V1 matcher runs at a shared threshold (read-only)."
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--baseline", required=True, help="run directory of arm A")
    parser.add_argument("--candidate", required=True, help="run directory of arm B")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="threshold applied to BOTH runs. Default: the baseline's own chosen "
        "operating point (operating_point.chosen_threshold in its v1_metrics.json)",
    )
    parser.add_argument(
        "--policy",
        choices=(ZERO_MATCH_POLICY, OTHER_POLICY, "both"),
        default="both",
        help="zero_match_policy to lead with; 'both' reports each and names the "
        "config's evaluation.zero_match_policy as primary",
    )
    parser.add_argument("--output", default=None, help="where to write v1_ab_comparison.json")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def load_run(directory: Path, log: logging.Logger) -> dict[str, Any]:
    """Load one run's saved probabilities, labels, owners and metrics.

    Raises rather than guessing: a missing artifact means the run did not finish, and
    comparing a finished run against a partial one would produce a number that looks
    like a result.
    """
    missing = [
        name
        for name in (PROBABILITIES_FILE, LABELS_FILE, OWNERS_FILE)
        if not (directory / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{directory} is missing {', '.join(missing)} - the run did not complete a "
            f"labelled pass, so there is nothing to score"
        )
    metrics_path = directory / METRICS_FILE
    if not metrics_path.is_file():
        raise FileNotFoundError(f"{directory} has no {METRICS_FILE}")
    metrics = read_json(metrics_path)

    probabilities = np.load(directory / PROBABILITIES_FILE).astype(np.float32)
    is_true = np.load(directory / LABELS_FILE).astype(bool)
    owners = np.load(directory / OWNERS_FILE).astype(np.int64)
    if not (len(probabilities) == len(is_true) == len(owners)):
        raise ValueError(
            f"{directory}: artifacts are not row-aligned - "
            f"{len(probabilities)} probabilities, {len(is_true)} labels, {len(owners)} owners"
        )
    log.info(
        "[%s] %s rows, %s positive, threshold %s",
        directory.name,
        fmt_int(len(probabilities)),
        fmt_int(int(is_true.sum())),
        metrics.get("operating_point", {}).get("chosen_threshold"),
    )
    return {
        "dir": directory,
        "metrics": metrics,
        "probabilities": probabilities,
        "is_true": is_true,
        "owners": owners,
    }


def chosen_threshold(run: dict[str, Any]) -> float:
    """The baseline's own operating point, from the sweep that recorded it."""
    point = (run["metrics"].get("operating_point") or {}).get("chosen_threshold")
    if point is None:
        raise ValueError(
            f"{run['dir']} has no operating_point.chosen_threshold; pass --threshold explicitly"
        )
    return float(point)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(
        args.config, overrides={"data_root": args.data_root, "work_dir": args.work_dir}
    )
    log = setup_logging(
        LOG_NAME,
        log_dir=config["resolved"]["log_dir"],
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    baseline_dir = Path(args.baseline).expanduser().resolve()
    candidate_dir = Path(args.candidate).expanduser().resolve()
    log.info("=" * 78)
    log.info("compare_v1_runs: A/B at a shared threshold")
    log.info("  A (baseline) : %s", baseline_dir)
    log.info("  B (candidate): %s", candidate_dir)
    log.info("=" * 78)

    # ---- the val population, from the repo's own pure split ------------------
    ground_truth = load_ground_truth(config, log=log)
    lengths = ground_truth.lengths()
    entity_mask = split_mask_for(ground_truth, config, "val")
    log.info(
        "val population: %s of %s S1 entities, %s true pairs",
        fmt_int(int(entity_mask.sum())),
        fmt_int(len(lengths)),
        fmt_int(int(lengths[entity_mask].sum())),
    )

    try:
        baseline = load_run(baseline_dir, log)
        candidate = load_run(candidate_dir, log)
    except (FileNotFoundError, ValueError) as error:
        log.error("%s", error)
        return 2

    for run in (baseline, candidate):
        # Owners are ground-truth row indices, so the vector they index must be the
        # same one. A run built against a different ground truth would otherwise be
        # scored against the wrong per-entity lengths - a silent, plausible wrong
        # answer rather than an error.
        if len(run["owners"]) and run["owners"].max() >= len(lengths):
            log.error(
                "%s: owner index %s exceeds the %s ground-truth entities - this run was "
                "built against a different ground truth",
                run["dir"].name,
                fmt_int(int(run["owners"].max())),
                fmt_int(len(lengths)),
            )
            return 2

    threshold = float(args.threshold) if args.threshold is not None else chosen_threshold(baseline)
    log.info("shared threshold: %.6g%s", threshold, "" if args.threshold is not None else " (A's own)")

    # ---- score both runs at that one threshold ------------------------------
    scored: dict[str, dict[str, Any]] = {}
    for label, run in (("A", baseline), ("B", candidate)):
        result = evaluate_at_threshold(
            threshold,
            run["probabilities"],
            run["is_true"],
            run["owners"],
            lengths,
            entity_mask,
        )
        scored[label] = result
        log.info(
            "[%s] predicted %s pairs | macro F0.5 score_zero %.6f | exclude %.6f",
            label,
            fmt_int(result["predicted_pairs"]),
            result["macro_f05_score_zero"],
            result["macro_f05_exclude"],
        )

    # ---- self-check: A at A's own threshold must reproduce what A reported ----
    # Skipped when the threshold was pinned by hand, because then the two numbers
    # describe different operating points and disagreeing is expected.
    reproduced: dict[str, Any] = {"checked": args.threshold is None}
    if args.threshold is None:
        reported = (baseline["metrics"].get("val_metrics") or {}).get("macro_f05_score_zero")
        measured = scored["A"]["macro_f05_score_zero"]
        reproduced.update(
            {
                "reported_by_baseline_run": reported,
                "measured_here": measured,
                "agrees": reported is not None and abs(float(reported) - measured) < 1e-9,
            }
        )
        if reproduced["agrees"]:
            log.info("self-check: reproduced A's reported macro F0.5 exactly")
        else:
            log.warning(
                "self-check FAILED: A reported %s, this script measured %s at the same "
                "threshold. Do not read the delta below until this agrees - the two runs "
                "are not on the same population.",
                reported,
                measured,
            )

    primary_policy = config.get("evaluation", {}).get("zero_match_policy", ZERO_MATCH_POLICY)
    if primary_policy not in POLICY_COLUMNS:
        log.warning("config zero_match_policy %r is not a known policy; leading with %s",
                    primary_policy, ZERO_MATCH_POLICY)
        primary_policy = ZERO_MATCH_POLICY

    comparisons = {}
    for policy, column in POLICY_COLUMNS.items():
        delta = scored["B"][column] - scored["A"][column]
        comparisons[policy] = {
            "column": column,
            "baseline": scored["A"][column],
            "candidate": scored["B"][column],
            "delta": delta,
            "primary": policy == primary_policy,
        }

    order = [p for p in POLICY_COLUMNS if p == primary_policy] + [
        p for p in POLICY_COLUMNS if p != primary_policy
    ]
    log.info("-" * 78)
    log.info("MACRO F0.5 AT THRESHOLD %.6g  (val split, same entities both arms)", threshold)
    log.info("%-14s %12s %12s %12s", "policy", "A", "B", "B - A")
    for policy in order:
        entry = comparisons[policy]
        log.info(
            "%-14s %12.6f %12.6f %+12.6f%s",
            policy,
            entry["baseline"],
            entry["candidate"],
            entry["delta"],
            "   <- primary" if entry["primary"] else "",
        )
    log.info("-" * 78)
    log.info("predicted pairs  A %s -> B %s (%+d)",
             fmt_int(scored["A"]["predicted_pairs"]),
             fmt_int(scored["B"]["predicted_pairs"]),
             scored["B"]["predicted_pairs"] - scored["A"]["predicted_pairs"])
    for key in ("pair_recall", "macro_recall_entity", "pair_precision"):
        if key in scored["A"] and key in scored["B"]:
            log.info("%-16s A %.6f -> B %.6f (%+.6f)", key,
                     scored["A"][key], scored["B"][key], scored["B"][key] - scored["A"][key])
    log.info(
        "NOTE: the primary number is the metric on this val population only; it is not "
        "the competition score, which also scores entities the feature file never covers."
    )

    payload = {
        "generated_by": "scripts/compare_v1_runs.py",
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "threshold": threshold,
        "threshold_source": "argument" if args.threshold is not None else "baseline operating point",
        "val_population": {
            "split": "val",
            "n_s1_entities": int(entity_mask.sum()),
            "n_true_pairs": int(lengths[entity_mask].sum()),
        },
        # The shape of each arm, so the size of the experiment is on the record
        # beside its result.
        "run_shape": {
            label: {
                "dir": str(run["dir"]),
                "rows": int(len(run["probabilities"])),
                "positive_labels": int(run["is_true"].sum()),
                "n_s1_entities_with_rows": int(len(np.unique(run["owners"]))),
                "chosen_threshold": (run["metrics"].get("operating_point") or {}).get(
                    "chosen_threshold"
                ),
                "reported_macro_f05_score_zero": (run["metrics"].get("val_metrics") or {}).get(
                    "macro_f05_score_zero"
                ),
            }
            for label, run in (("A", baseline), ("B", candidate))
        },
        "baseline_val_metrics": baseline["metrics"].get("val_metrics"),
        "candidate_val_metrics": candidate["metrics"].get("val_metrics"),
        "baseline_operating_point": baseline["metrics"].get("operating_point"),
        "candidate_operating_point": candidate["metrics"].get("operating_point"),
        "at_shared_threshold": {"baseline": scored["A"], "candidate": scored["B"]},
        "comparisons": comparisons,
        "primary_policy": primary_policy,
        "self_check": reproduced,
    }
    output_dir = ensure_dir(Path(args.output) if args.output else candidate_dir)
    output_path = output_dir / "v1_ab_comparison.json"
    write_json(output_path, payload)
    log.info("comparison written: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
