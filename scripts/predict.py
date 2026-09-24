#!/usr/bin/env python
"""Stage 6 (milestone 2): write ``matching_results.tsv``.  **NOT IMPLEMENTED.**

Deliberately not implemented, for two reasons:

1. **The submission format is unknown.** The brief says to follow the challenge's
   exact required format, and that format has not been provided. Guessing it
   would produce a file that looks plausible and scores zero. The only format
   evidence available is the training ground truth
   (``source1_entity_id, matched_entity_ids`` with comma-separated ids), which
   suggests the submission mirrors it - but "suggests" is not good enough for the
   artifact that is actually graded. Please confirm the spec, then this becomes
   a small script.

2. **There is nothing to predict with yet.** No matcher exists (see
   ``train_model.py``), and the exact-name blocker alone is not a submission.

What the format rules already constrain, and what this script must guarantee
once implemented:

* every S1 entity appears **exactly once**, including the ones with no match
  (123,247 of the training S1 entities have an empty ground-truth list, so the
  "no match" case is a real, required output row - not an omission)
* matched ids are **deduplicated** and contain only valid S2/S3 ids
* ordering: emit rows in S1 file order and, within a row, in a deterministic id
  order, so reruns are byte-identical

The streaming shape is already proven by the other stages; once the format is
confirmed this is a merge-join between ``candidate_pairs.tsv`` and the S1 id list,
written through ``ChunkWriter`` so peak memory stays bounded.

    python scripts/predict.py        # raises NotImplementedError by design
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import describe_environment, load_config  # noqa: E402
from src.utils import describe_device, setup_logging  # noqa: E402

LOG_NAME = "predict"

NOT_IMPLEMENTED_MESSAGE = (
    "scripts/predict.py is a milestone-2 stub.\n"
    "Blocked on two things:\n"
    "  1. the challenge's exact matching_results.tsv format (please confirm - I did not want to guess it)\n"
    "  2. a trained matcher (see scripts/train_model.py)\n"
    "The training ground truth uses 'source1_entity_id' + comma-separated 'matched_entity_ids',\n"
    "which is the likely submission shape, but this must be confirmed against the challenge spec."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write matching_results.tsv (stage 6, milestone 2).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config, overrides={"data_root": args.data_root, "work_dir": args.work_dir})
    log = setup_logging(LOG_NAME, log_dir=config["resolved"]["log_dir"])

    log.info("=" * 78)
    log.info("predict: not implemented in this milestone")
    log.info(describe_environment(config))
    describe_device(log, config)
    log.info("=" * 78)
    log.error("%s", NOT_IMPLEMENTED_MESSAGE)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
