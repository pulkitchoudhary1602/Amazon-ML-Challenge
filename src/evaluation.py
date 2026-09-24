"""Evaluation of blocking quality and of the final matching metric.

Two different questions get answered here, and it is worth keeping them apart:

1. **Blocking quality** - did the candidate generator retrieve the true matches?
   Measured by recall (pair-level and S1-level) at a given candidate volume.
   A blocker with high recall and an unusable number of candidates is not
   progress, so volume metrics are reported alongside every recall number.

2. **Match quality** - the challenge metric. F0.5, computed per S1 entity and
   then macro-averaged. Because the average is over entities, a single S1 with
   many matches does not dominate, and because beta is 0.5, precision is worth
   four times as much as recall.

Both are computed from the same streaming pass, without materializing the
candidate table:

``build_true_pair_codes`` packs every ground-truth pair into one sorted int64
array (~61MB for 7.6M pairs). A candidate pair is a true positive iff its packed
form is present in that array, which turns "is this the right answer for this
S1?" into a vectorized ``searchsorted`` instead of a per-entity python loop over
2.2M groups.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .blocking import PAIR_MULTIPLIER
from .data_loader import GroundTruth, assign_splits
from .utils import fmt_int, write_json

logger = logging.getLogger(__name__)

F_BETA = 0.5

CANDIDATE_S1_COLUMN = "source1_entity_id"
CANDIDATE_TARGET_COLUMN = "matched_entity_id"
CANDIDATE_SOURCE_COLUMN = "source"
CANDIDATE_SCORE_COLUMN = "score"


def build_true_pair_codes(ground_truth: GroundTruth) -> np.ndarray:
    """Pack and sort every ground-truth pair as ``owner * MULT + entity_code``.

    The result is the ground-truth pair set in the same coordinate system as the
    candidates, so membership testing is a binary search rather than a dict of
    sets. Sorted output is what ``searchsorted`` requires.

    Memory: ``8 * n_true_pairs`` bytes - ~61MB for the full training set.
    """
    lengths = ground_truth.lengths()
    owners = np.repeat(np.arange(ground_truth.n_entities, dtype=np.int64), lengths)
    packed = owners * PAIR_MULTIPLIER + ground_truth.codes
    packed.sort()
    return packed


class CandidateEvaluation:
    """Streaming evaluator comparing a candidate file against the ground truth.

    Usage::

        gt = load_ground_truth(config)
        evaluation = CandidateEvaluation(gt, n_target_records=10_320_219)
        metrics = evaluation.evaluate_file(candidates_path)
        print(format_report(metrics))

    The evaluator never holds the candidate table: it accumulates per-entity
    counters with ``np.bincount``, so RAM is O(n_S1) regardless of how many
    candidate pairs the file contains.
    """

    def __init__(
        self,
        ground_truth: GroundTruth,
        n_target_records: Optional[int] = None,
        k_values: Sequence[int] = (10, 25, 50, 100, 200),
        log: Optional[logging.Logger] = None,
    ) -> None:
        self.ground_truth = ground_truth
        self.n_entities = ground_truth.n_entities
        self.n_target_records = int(n_target_records) if n_target_records else None
        self.k_values = [int(k) for k in k_values]
        self.log = log or logger

        self.true_lengths = ground_truth.lengths()
        self.n_true_pairs = int(self.true_lengths.sum())
        self.true_pair_codes = build_true_pair_codes(ground_truth)

        # Per-entity accumulators, filled during evaluate_file.
        self._candidate_counts = np.zeros(self.n_entities, dtype=np.int64)
        self._hit_counts = np.zeros(self.n_entities, dtype=np.int64)
        self._candidate_counts_by_source = {2: np.zeros(self.n_entities, dtype=np.int64), 3: np.zeros(self.n_entities, dtype=np.int64)}
        self._hit_counts_by_source = {2: np.zeros(self.n_entities, dtype=np.int64), 3: np.zeros(self.n_entities, dtype=np.int64)}
        self._hits_at_k = {k: 0 for k in self.k_values}
        self._unknown_s1 = 0
        self._n_candidate_rows = 0

    # -- streaming ----------------------------------------------------------
    def evaluate_file(
        self,
        path: str | os.PathLike,
        chunksize: int = 500_000,
        max_rows: Optional[int] = None,
    ) -> dict:
        """Consume a candidate TSV and return the metric dict.

        Args:
            path: candidate pairs file (columns: S1 id, target id, source, ...).
            chunksize: rows per read - tune against available RAM.
            max_rows: stop after this many candidate rows (smoke tests).

        Returns:
            Metrics dict, also usable with :func:`format_report`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"candidate file not found: {path}\n  Run: python scripts/generate_candidates.py")

        self.log.info("evaluating candidates from %s", path)
        rows_seen = 0
        trailing_s1: Optional[int] = None
        trailing_position = 0

        for chunk in _iter_candidate_chunks(path, chunksize):
            if max_rows is not None and rows_seen >= max_rows:
                break
            if max_rows is not None and rows_seen + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - rows_seen]
            rows_seen += len(chunk)

            owners = self.ground_truth.positions_of(chunk[CANDIDATE_S1_COLUMN])
            target_codes = _encode_target_codes(chunk[CANDIDATE_TARGET_COLUMN])

            unknown = owners < 0
            if unknown.any():
                self._unknown_s1 += int(unknown.sum())
                owners = owners[~unknown]
                target_codes = target_codes[~unknown]

            packed = owners * PAIR_MULTIPLIER + target_codes
            is_true = _contains_sorted(self.true_pair_codes, packed)

            # Position of each candidate within its own S1 group, needed for the
            # recall@K sweep. The union writes S1 groups contiguously, but a
            # group can straddle a chunk boundary, so carry the trailing group's
            # count into the next chunk.
            position_in_group = _position_within_groups(owners, trailing_s1, trailing_position)
            if len(owners):
                trailing_s1 = int(owners[-1])
                trailing_position = int(position_in_group[-1]) + 1

            np.add.at(self._candidate_counts, owners, 1)
            self._hit_counts[owners[is_true]] += 1

            source_codes = target_codes // 10**10
            for source_code, counts in self._candidate_counts_by_source.items():
                mask = source_codes == source_code
                if mask.any():
                    np.add.at(counts, owners[mask], 1)
                    self._hit_counts_by_source[source_code][owners[mask & is_true]] += 1

            for k in self.k_values:
                in_top_k = position_in_group < k
                self._hits_at_k[k] += int(np.count_nonzero(is_true & in_top_k))

            self._n_candidate_rows = rows_seen
            self.log.info("  evaluated %s candidate rows", fmt_int(rows_seen))

        return self.compute_metrics()

    # -- aggregation --------------------------------------------------------
    def compute_metrics(self, s1_mask: Optional[np.ndarray] = None, split_label: str = "all") -> dict:
        """Turn the accumulators into a metrics dict.

        Args:
            s1_mask: optional boolean mask over ground-truth entities, used to
                report validation metrics only. Splitting is by S1 entity, so
                masking entities can never leak pairs across the split.
            split_label: label recorded in the output.

        Returns:
            Metrics dict.
        """
        if s1_mask is None:
            lengths = self.true_lengths
            candidates = self._candidate_counts
            hits = self._hit_counts
            by_source = {
                code: (self._candidate_counts_by_source[code], self._hit_counts_by_source[code])
                for code in self._candidate_counts_by_source
            }
        else:
            mask = np.asarray(s1_mask, dtype=bool)
            lengths = self.true_lengths[mask]
            candidates = self._candidate_counts[mask]
            hits = self._hit_counts[mask]
            by_source = {
                code: (
                    self._candidate_counts_by_source[code][mask],
                    self._hit_counts_by_source[code][mask],
                )
                for code in self._candidate_counts_by_source
            }

        n_entities = len(lengths)
        n_with_matches = int(np.count_nonzero(lengths > 0))
        n_true_pairs = int(lengths.sum())
        n_candidate_pairs = int(candidates.sum())
        n_hits = int(hits.sum())

        has_candidates = candidates > 0
        # An entity is "fully retrieved" when every true match appeared as a
        # candidate. This is the ceiling on that entity's F0.5: no matcher can
        # score well on matches the blocker never proposed.
        fully_retrieved = (lengths > 0) & (hits >= lengths)
        partially_retrieved = (lengths > 0) & (hits > 0)

        metrics: dict[str, Any] = {
            "split": split_label,
            "n_s1_entities": n_entities,
            "n_s1_with_true_matches": n_with_matches,
            "n_s1_with_zero_true_matches": n_entities - n_with_matches,
            "n_true_pairs": n_true_pairs,
            "n_candidate_pairs": n_candidate_pairs,
            "candidate_rows_read": self._n_candidate_rows,
            "unknown_s1_in_candidates": self._unknown_s1,
            # --- recall ---
            "true_pairs_retrieved": n_hits,
            "blocking_recall_pair": _safe_div(n_hits, n_true_pairs),
            "s1_full_recall_rate": _safe_div(int(fully_retrieved.sum()), n_with_matches),
            "s1_partial_recall_rate": _safe_div(int(partially_retrieved.sum()), n_with_matches),
            # --- volume ---
            "avg_candidates_per_s1": _safe_div(n_candidate_pairs, n_entities),
            "avg_candidates_per_s1_with_matches": _safe_div(n_candidate_pairs, n_with_matches),
            "median_candidates_per_s1": float(np.median(candidates)) if n_entities else 0.0,
            "p99_candidates_per_s1": float(np.percentile(candidates, 99)) if n_entities else 0.0,
            "max_candidates_per_s1": int(candidates.max()) if n_entities else 0,
            "n_s1_with_zero_candidates": int((~has_candidates).sum()),
            "fraction_s1_with_zero_candidates": _safe_div(int((~has_candidates).sum()), n_entities),
            # --- precision ---
            "candidate_precision": _safe_div(n_hits, n_candidate_pairs),
            # --- challenge metric, if every candidate were accepted ---
            "f05_accept_all_macro": self._macro_f05(lengths, candidates, hits, policy="exclude"),
            "f05_accept_all_macro_score_zero": self._macro_f05(lengths, candidates, hits, policy="score_zero"),
        }

        # --- reduction ---
        if self.n_target_records:
            possible = n_entities * self.n_target_records
            metrics["n_possible_pairs"] = int(possible)
            metrics["reduction_ratio"] = _safe_div(possible, n_candidate_pairs)
            metrics["candidate_rate"] = _safe_div(n_candidate_pairs, possible)

        # --- recall@K (file order) ---
        # Until blocker scores exist, "top K" means the first K rows in file
        # order, which is arbitrary. It is reported because it bounds what any
        # future re-ranking can achieve, and because it makes the cost of
        # capping candidate volume explicit.
        metrics["recall_at_k_file_order"] = {
            str(k): _safe_div(self._hits_at_k[k], n_true_pairs) for k in self.k_values
        }

        # --- per-source breakdown ---
        per_source = {}
        for source_code, (counts, source_hits) in by_source.items():
            prefix = "S2" if source_code == 2 else "S3"
            true_for_source = _true_counts_per_entity(self.ground_truth, source_code, s1_mask)
            n_true_source = int(true_for_source.sum())
            n_hits_source = int(source_hits.sum())
            # Same S1-level definition as above, restricted to one target source:
            # the entity retrieved every true match that lives in this source.
            source_full = (true_for_source > 0) & (source_hits >= true_for_source)
            per_source[prefix] = {
                "n_true_pairs": n_true_source,
                "n_candidates": int(counts.sum()),
                "true_pairs_retrieved": n_hits_source,
                "blocking_recall_pair": _safe_div(n_hits_source, n_true_source),
                "s1_full_recall_rate": _safe_div(int(source_full.sum()), int((true_for_source > 0).sum())),
                "s1_partial_recall_rate": _safe_div(
                    int(((true_for_source > 0) & (source_hits > 0)).sum()), int((true_for_source > 0).sum())
                ),
                "candidate_precision": _safe_div(n_hits_source, int(counts.sum())),
            }
        metrics["per_source"] = per_source

        return metrics

    @staticmethod
    def _macro_f05(lengths: np.ndarray, candidates: np.ndarray, hits: np.ndarray, policy: str) -> float:
        """Macro-averaged F0.5 assuming every candidate is accepted as a match.

        This is the score the pipeline would get with a perfect matcher given the
        current blockers - i.e. an upper bound on the current candidate set, and
        a direct measure of how much precision the ranking model must add.

        Args:
            lengths: true match counts per entity.
            candidates: candidate counts per entity.
            hits: retrieved true match counts per entity.
            policy: ``exclude`` (average over entities with >=1 true match) or
                ``score_zero`` (entities with no true match score 0 if anything
                was predicted, 1 otherwise).
        """
        true_positive = hits.astype(np.float64)
        false_positive = (candidates - hits).astype(np.float64)
        false_negative = (lengths - hits).astype(np.float64)

        beta_squared = F_BETA**2
        denominator = (1.0 + beta_squared) * true_positive + beta_squared * false_negative + false_positive
        scores = np.where(denominator > 0, (1.0 + beta_squared) * true_positive / np.where(denominator > 0, denominator, 1.0), np.nan)

        if policy == "score_zero":
            zero_match = lengths == 0
            scores = np.where(zero_match, np.where(candidates > 0, 0.0, 1.0), np.nan_to_num(scores, nan=0.0))
            return float(scores.mean()) if len(scores) else 0.0

        eligible = lengths > 0
        if not eligible.any():
            return 0.0
        return float(np.nanmean(scores[eligible]))


def _true_counts_per_entity(
    ground_truth: GroundTruth,
    source_code: int,
    s1_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-entity true match counts restricted to one target source (2 or 3)."""
    owners = np.repeat(np.arange(ground_truth.n_entities, dtype=np.int64), ground_truth.lengths())
    belongs = (ground_truth.codes // 10**10) == source_code
    per_entity = np.bincount(
        owners[belongs], minlength=ground_truth.n_entities
    ).astype(np.int64)
    if s1_mask is not None:
        per_entity = per_entity[np.asarray(s1_mask, dtype=bool)]
    return per_entity


def _position_within_groups(owners: np.ndarray, trailing_s1: Optional[int], trailing_position: int) -> np.ndarray:
    """Index of each row within its own S1 group.

    Assumes ``owners`` is non-decreasing (true for candidate files written by
    ``generate_candidates.py``). Handles a group split across chunk boundaries by
    seeding the first group's counter from the previous chunk.
    """
    if len(owners) == 0:
        return np.empty(0, dtype=np.int64)

    is_new_group = np.empty(len(owners), dtype=bool)
    is_new_group[0] = True
    if len(owners) > 1:
        np.not_equal(owners[1:], owners[:-1], out=is_new_group[1:])
    group_starts = np.flatnonzero(is_new_group)

    positions = np.arange(len(owners), dtype=np.int64)
    offsets = np.zeros(len(group_starts), dtype=np.int64)
    if trailing_s1 is not None and group_starts[0] == 0 and owners[0] == trailing_s1:
        # First group continues from the previous chunk.
        offsets[0] = trailing_position
    positions -= np.repeat(group_starts - offsets, np.diff(np.append(group_starts, len(owners))))
    return positions


def _contains_sorted(sorted_codes: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Vectorized membership test in a sorted int64 array."""
    if len(query) == 0:
        return np.empty(0, dtype=bool)
    if len(sorted_codes) == 0:
        return np.zeros(len(query), dtype=bool)
    idx = np.searchsorted(sorted_codes, query)
    np.clip(idx, 0, len(sorted_codes) - 1, out=idx)
    return sorted_codes[idx] == query


def _encode_target_codes(target_ids: pd.Series) -> np.ndarray:
    """Encode ``"S2-123"`` strings into packed int64 codes without a dict lookup."""
    prefixes = target_ids.str.slice(0, 2)
    numerics = target_ids.str.slice(3).astype("int64").to_numpy()
    source_codes = prefixes.map({"S2": 2, "S3": 3}).to_numpy()
    if np.isnan(source_codes).any():
        bad = target_ids[prefixes.isin(["S2", "S3"]) == False].head(5).tolist()
        raise ValueError(f"candidate target ids must start with S2-/S3-, found: {bad}")
    return source_codes.astype(np.int64) * 10**10 + numerics


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 instead of raising / returning inf."""
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _iter_candidate_chunks(path: Path, chunksize: int):
    """Stream the candidate TSV with only the columns evaluation needs."""
    from .data_loader import iter_tsv

    columns = [CANDIDATE_S1_COLUMN, CANDIDATE_TARGET_COLUMN]
    available = _peek_columns(path)
    columns = [c for c in columns if c in available]
    yield from iter_tsv(path, columns=columns, chunksize=chunksize)


def _peek_columns(path: Path) -> list[str]:
    """Read just the header of a (possibly compressed) TSV."""
    import gzip

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline()
    return header.rstrip("\n").split("\t")


def split_mask_for(
    ground_truth: GroundTruth,
    config: dict,
    split: str = "val",
) -> np.ndarray:
    """Boolean mask selecting the ground-truth entities belonging to ``split``.

    Splitting is done with :func:`~src.data_loader.assign_splits`, a pure
    function of the S1 id, so the candidate generator and the evaluator always
    agree on which entities are held out - without passing any state between
    them.
    """
    section = config.get("evaluation", {}).get("split", {}) or {}
    labels = assign_splits(
        pd.Series(ground_truth.entity_ids),
        val_fraction=section.get("val_fraction", 0.2),
        mode=section.get("mode", "hash"),
        seed=config.get("project", {}).get("seed", 42),
    )
    return labels == split


def format_report(metrics: dict) -> str:
    """Render a metrics dict as a readable text block."""
    lines = []
    add = lines.append
    add("=" * 78)
    add(f"BLOCKING EVALUATION  (split={metrics.get('split')})")
    add("=" * 78)
    add("")
    add("-- recall " + "-" * 67)
    add(f"  true pairs (ground truth)        : {fmt_int(metrics['n_true_pairs'])}")
    add(f"  true pairs retrieved             : {fmt_int(metrics['true_pairs_retrieved'])}")
    add(f"  blocking recall (pair-level)     : {100 * metrics['blocking_recall_pair']:.2f}%")
    add(
        f"  S1 entities with ALL matches     : {100 * metrics['s1_full_recall_rate']:.2f}%"
        f"  ({fmt_int(round(metrics['s1_full_recall_rate'] * metrics['n_s1_with_true_matches']))}"
        f" of {fmt_int(metrics['n_s1_with_true_matches'])})"
    )
    add(f"  S1 entities with >=1 match       : {100 * metrics['s1_partial_recall_rate']:.2f}%")
    add("")
    add("-- candidate volume " + "-" * 58)
    add(f"  candidate pairs                  : {fmt_int(metrics['n_candidate_pairs'])}")
    add(f"  avg candidates per S1            : {metrics['avg_candidates_per_s1']:.3f}")
    add(f"  median / p99 / max per S1        : {metrics['median_candidates_per_s1']:.0f} / "
        f"{metrics['p99_candidates_per_s1']:.0f} / {fmt_int(metrics['max_candidates_per_s1'])}")
    add(
        f"  S1 with zero candidates          : {fmt_int(metrics['n_s1_with_zero_candidates'])}"
        f"  ({100 * metrics['fraction_s1_with_zero_candidates']:.2f}%)"
    )
    if "n_possible_pairs" in metrics:
        add(f"  full cross product (avoided)     : {metrics['n_possible_pairs']:.3e}")
        add(f"  reduction ratio                  : {metrics['reduction_ratio']:.3e}x")
    add("")
    add("-- candidate quality " + "-" * 57)
    add(f"  candidate precision              : {100 * metrics['candidate_precision']:.3f}%")
    add("  F0.5 if EVERY candidate accepted:")
    add(f"      macro over S1 with matches   : {metrics['f05_accept_all_macro']:.4f}   (ceiling for a perfect matcher)")
    add(f"      zero-match S1 scored 0/1     : {metrics['f05_accept_all_macro_score_zero']:.4f}")
    add("")
    add("-- recall@K (first K in file order) " + "-" * 41)
    for k, value in metrics["recall_at_k_file_order"].items():
        add(f"  K={k:<5} : {100 * value:.2f}%")
    add("")
    add("-- per source " + "-" * 64)
    for prefix, stats in metrics.get("per_source", {}).items():
        add(
            f"  {prefix}: true={fmt_int(stats['n_true_pairs'])} "
            f"candidates={fmt_int(stats['n_candidates'])} "
            f"recall={100 * stats['blocking_recall_pair']:.2f}%"
        )
    add("")
    if metrics.get("unknown_s1_in_candidates"):
        add(f"  WARNING: {fmt_int(metrics['unknown_s1_in_candidates'])} candidate rows referenced unknown S1 ids")
    add("=" * 78)
    return "\n".join(lines)


def save_metrics(metrics: dict, path: str | os.PathLike) -> None:
    """Persist metrics as JSON next to the run's other outputs."""
    write_json(path, metrics)
