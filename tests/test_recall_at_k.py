"""Regression test for the recall@K split-masking bug in ``src.evaluation``.

The bug
-------
``CandidateEvaluation.evaluate_file`` accumulated ``_hits_at_k`` as ONE GLOBAL
SCALAR per K, counted over every entity in the candidate file. But
``compute_metrics(s1_mask=...)`` masks the denominator (``n_true_pairs``) down to
a single split. The ``--split val`` and ``--split train`` reports therefore
divided all-entity hits by split-only true pairs and printed figures above 100%
(300% for val and 150% for train were observed on a smoke run).

The fix
-------
``_hits_at_k`` is now a per-entity accumulator, reduced under the same mask as
every other counter, so numerator and denominator always describe the same S1
split. The definition of recall@K is unchanged: among the first K candidate rows
that an S1 entity owns, in file order, how many are true matches - divided by the
number of true matches that entity has.

The fixture
-----------
Built so the OLD formulation genuinely exceeds 100%, which is what makes this a
regression test rather than a tautology: the two val entities retrieve almost
nothing (their only true match sits at group position 3), the four train entities
retrieve everything inside the first K rows, and the val denominator is small.

Runs standalone (``python tests/test_recall_at_k.py``) and under pytest. Nothing
here touches the dataset: all data is synthetic, in memory or in a temp file.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import GroundTruth  # noqa: E402
from src.evaluation import (  # noqa: E402
    CANDIDATE_S1_COLUMN,
    CANDIDATE_TARGET_COLUMN,
    CandidateEvaluation,
)
from src.utils import encode_entity_id  # noqa: E402

K_VALUES = (1, 2, 3, 4)

# S1 id -> (true target ids, candidate target ids in file order).
# Groups are contiguous in the file, as the real generator writes them.
FIXTURE: dict[str, tuple[list[str], list[str]]] = {
    # --- val: 1 true pair each, retrieved only at group position 3 ---
    "S1-1": (["S2-100"], ["S2-901", "S2-902", "S2-903", "S2-100"]),
    "S1-2": (["S2-200"], ["S2-904", "S2-905", "S2-906", "S2-200"]),
    # --- train: everything retrieved at the very front of the group ---
    "S1-3": (["S2-300", "S3-301", "S2-302"], ["S2-300", "S3-301", "S2-302"]),
    "S1-4": (["S2-400", "S3-401", "S2-402"], ["S2-400", "S3-401", "S2-402"]),
    "S1-5": (["S2-500", "S3-501"], ["S2-500", "S3-501"]),
    "S1-6": (["S3-600"], ["S3-600"]),
    # --- train: a true match the blocker never proposed ---
    "S1-7": (["S2-700"], []),
}

VAL_SPLIT = ("S1-1", "S1-2")
TRAIN_SPLIT = ("S1-3", "S1-4", "S1-5", "S1-6", "S1-7")


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _ground_truth() -> GroundTruth:
    """Build the ground truth in CSR layout from ``FIXTURE``."""
    entity_ids = np.array(list(FIXTURE), dtype=object)
    flat: list[str] = []
    lengths: list[int] = []
    for trues, _ in FIXTURE.values():
        flat.extend(trues)
        lengths.append(len(trues))
    codes = np.array([encode_entity_id(t) for t in flat], dtype=np.int64)
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(np.array(lengths, dtype=np.int64), out=offsets[1:])
    return GroundTruth(entity_ids, offsets, codes)


def _mask(entity_ids: tuple[str, ...]) -> np.ndarray:
    """Boolean mask over ground-truth rows selecting ``entity_ids``."""
    return np.array([e in set(entity_ids) for e in FIXTURE], dtype=bool)


def _candidate_rows() -> list[tuple[str, str, bool, int]]:
    """``(s1, target, is_true, position_within_group)`` for every candidate row."""
    rows = []
    for s1, (trues, candidates) in FIXTURE.items():
        true_set = set(trues)
        for position, target in enumerate(candidates):
            rows.append((s1, target, target in true_set, position))
    return rows


def _reference() -> dict[str, tuple[dict[int, int], int]]:
    """Brute-force recall@K numerator and denominator, per split.

    Deliberately written as a plain python loop over rows so it shares no code
    with the accumulator path under test. The numerator counts true candidate
    rows inside the first K of the entity's own group; the denominator counts the
    selected entities' true pairs. Both are restricted to the same split.
    """
    rows = _candidate_rows()
    reference = {}
    for label, selected in (
        ("all", set(FIXTURE)),
        ("val", set(VAL_SPLIT)),
        ("train", set(TRAIN_SPLIT)),
    ):
        denominator = sum(len(FIXTURE[s][0]) for s in selected)
        numerator = {
            k: sum(1 for s1, _, is_true, pos in rows if s1 in selected and is_true and pos < k)
            for k in K_VALUES
        }
        reference[label] = (numerator, denominator)
    return reference


def _evaluate() -> dict[str, dict]:
    """Run the real streaming evaluator over the fixture, once per split."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate_pairs.tsv"
        lines = [f"{CANDIDATE_S1_COLUMN}\t{CANDIDATE_TARGET_COLUMN}\tsource\tscore"]
        for s1, (trues, candidates) in FIXTURE.items():
            true_set = set(trues)
            for target in candidates:
                source = "source2" if target.startswith("S2") else "source3"
                score = "1.0" if target in true_set else "0.1"
                lines.append(f"{s1}\t{target}\t{source}\t{score}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        evaluator = CandidateEvaluation(_ground_truth(), n_target_records=100, k_values=K_VALUES)
        # chunksize=3 straddles the 4-row and 3-row groups deliberately, so the
        # per-group position carry across a chunk boundary is exercised too.
        all_metrics = evaluator.evaluate_file(path, chunksize=3)
        return {
            "all": all_metrics,
            "val": evaluator.compute_metrics(s1_mask=_mask(VAL_SPLIT), split_label="val"),
            "train": evaluator.compute_metrics(s1_mask=_mask(TRAIN_SPLIT), split_label="train"),
        }


def _at_k(metrics: dict) -> dict[int, float]:
    return {int(k): v for k, v in metrics["recall_at_k_file_order"].items()}


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_recall_at_k_never_exceeds_one() -> None:
    """recall@K <= 1.0 for every split and every K, including val and train."""
    for label, metrics in _evaluate().items():
        values = _at_k(metrics)
        for k, value in values.items():
            assert 0.0 <= value <= 1.0 + 1e-12, f"{label} split, K={k}: recall@{k} = {value}"


def test_split_numerator_and_denominator_use_same_split() -> None:
    """Each split's recall@K equals its own numerator over its own denominator.

    Also checks the splits partition the whole: the three numerators sum to the
    all-split numerator at every K, which can only hold if every numerator is
    restricted to exactly the entities its denominator counts.
    """
    reference = _reference()
    metrics = _evaluate()

    for label in ("val", "train", "all"):
        expected_numerator, expected_denominator = reference[label]
        assert metrics[label]["n_true_pairs"] == expected_denominator, label
        for k, value in _at_k(metrics[label]).items():
            expected = expected_numerator[k] / expected_denominator
            assert abs(value - expected) < 1e-12, (
                f"{label} split, K={k}: got {value}, expected {expected_numerator[k]}/{expected_denominator}"
            )

    all_numerator, _ = reference["all"]
    val_numerator, _ = reference["val"]
    train_numerator, _ = reference["train"]
    for k in K_VALUES:
        assert val_numerator[k] + train_numerator[k] == all_numerator[k]
        # The val figure must be a genuinely different number from the all-split
        # figure, i.e. the split restriction is doing something.
        if k in (1, 2, 3, 4):
            assert _at_k(metrics["val"])[k] != _at_k(metrics["all"])[k] or all_numerator[k] == 0


def test_all_split_results_remain_correct() -> None:
    """The unmasked path still reports the true global recall@K."""
    reference = _reference()
    values = _at_k(_evaluate()["all"])
    numerator, denominator = reference["all"]
    for k, value in values.items():
        assert abs(value - numerator[k] / denominator) < 1e-12

    # Spot-check the absolute numbers so a silently-zeroed accumulator fails.
    assert numerator == {1: 4, 2: 7, 3: 9, 4: 11}
    assert denominator == 12
    assert abs(values[1] - 4 / 12) < 1e-12
    assert abs(values[4] - 11 / 12) < 1e-12


def test_full_mask_agrees_with_no_mask() -> None:
    """compute_metrics(s1_mask=all True) must equal compute_metrics(None)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate_pairs.tsv"
        lines = [f"{CANDIDATE_S1_COLUMN}\t{CANDIDATE_TARGET_COLUMN}\tsource\tscore"]
        for s1, (_, candidates) in FIXTURE.items():
            for target in candidates:
                lines.append(f"{s1}\t{target}\tsource2\t1.0")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        evaluator = CandidateEvaluation(_ground_truth(), k_values=K_VALUES)
        unmasked = _at_k(evaluator.evaluate_file(path, chunksize=3))
        masked = _at_k(evaluator.compute_metrics(s1_mask=np.ones(len(FIXTURE), dtype=bool)))
    assert unmasked == masked


def test_old_formulation_would_have_exceeded_one() -> None:
    """The fixture reproduces the bug: the old global-numerator form breaks 100%.

    This is what makes the test above meaningful. If someone reintroduces a
    scalar numerator, the val figure goes back to these values.
    """
    rows = _candidate_rows()
    global_numerator = {
        k: sum(1 for _, _, is_true, pos in rows if is_true and pos < k) for k in K_VALUES
    }
    _, val_denominator = _reference()["val"]
    old_val = {k: global_numerator[k] / val_denominator for k in K_VALUES}

    assert any(value > 1.0 for value in old_val.values()), old_val
    assert old_val[1] == 2.0 and old_val[3] == 4.5, old_val

    # ...and the fixed evaluator does NOT agree with it.
    assert _at_k(_evaluate()["val"]) != old_val


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
