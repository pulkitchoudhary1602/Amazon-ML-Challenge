#!/usr/bin/env python
"""Stage 5 (milestone 2): train the match classifier.  **NOT IMPLEMENTED.**

Deliberately not implemented yet. The brief says not to jump to transformer
training, and the measured ceiling of the current exact-name blocker is far too
low (see the blocking evaluation report) for a trained matcher to be meaningful:
the model would learn to rank a candidate set that is missing most true matches,
and its F0.5 would mostly measure the blocker, not the model.

What is in place for this stage:

* ``candidate_pairs.tsv`` with a ``blockers`` provenance column
* a validation split defined by S1 entity, so train/val never share an entity
* ``src/evaluation.py`` with the per-entity macro F0.5 implementation, so any
  trained model can be scored the way the challenge scores it
* ``src/utils.resolve_device()``, which returns ``"cpu"`` on this machine and
  ``"cuda"`` automatically on an HPC node with a GPU

Recommended order once the blockers are strong enough:

1. token / character n-gram blockers, then re-read the blocking report
2. lexical features + a threshold on one score (CPU, no model needed)
3. gradient-boosted trees on lexical + address features
4. embeddings, then cross-encoder re-ranking - only if they still pay off

    python scripts/train_model.py        # raises NotImplementedError by design
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import describe_environment, load_config  # noqa: E402
from src.matching_model import NOT_IMPLEMENTED_MESSAGE  # noqa: E402
from src.utils import describe_device, setup_logging  # noqa: E402

LOG_NAME = "train_model"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the match model (stage 5, milestone 2).")
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
    log.info("train_model: not implemented in this milestone")
    log.info(describe_environment(config))
    describe_device(log)
    log.info("=" * 78)
    log.error("%s", NOT_IMPLEMENTED_MESSAGE)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
