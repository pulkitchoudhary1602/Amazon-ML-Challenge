#!/usr/bin/env python
"""Measure the Phase 0.2-0.5 hot paths so the HPC run is configured from data.

This script answers one question with numbers instead of intuition: *for this
workload, on this machine, which implementation is actually fastest?* It is
deliberately separate from the pipeline. The analyzer keeps Phase 0.1's reference
character-3-gram implementation unconditionally, because that is the definition
the report is written against; this harness is where alternatives - including the
GPU one - are measured, so the measurement can happen without any GPU-shaped code
entering the production path.

What it times, on real pairs sampled from the prepared data:

1. **The char-3-gram arms.** ``reference`` (Phase 0.1's python sets), ``numpy``
   (one flat vectorized pass per batch), ``torch`` (the same flattening on the
   GPU). CPU seconds, GPU seconds, speedup, peak GPU memory and peak CPU RSS.
2. **The knobs that decide the HPC command**: worker counts and chunk sizes,
   measured by running the *real* per-pair statistic, not a proxy for it.

Every arm is checked against the reference for equality before its time is
reported, so a fast-but-wrong arm can never be mistaken for a win. A mismatch is
printed as an error and the arm's timing is still shown - visibly disqualified
rather than silently believed.

Usage::

    python scripts/benchmark_phase0.py --sample 200000
    python scripts/benchmark_phase0.py --sample 500000 --arms reference,numpy,torch
    python scripts/benchmark_phase0.py --skip-workers          # char arms only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_config, load_ground_truth, prepared_path  # noqa: E402
from src.normalization import NAME_KEY  # noqa: E402
from src.utils import (  # noqa: E402
    current_rss_bytes,
    detect_hardware,
    format_hardware_report,
    human_bytes,
    peak_rss_bytes,
    setup_logging,
)

LOG_NAME = "benchmark_phase0"

ALL_ARMS = ("reference", "numpy", "torch")

# Collision-free trigram key: three uint32 code points viewed as one 12-byte void.
# Using the code points themselves rather than a hash is what makes the vectorized
# path exactly equal to the python set version, with no collision risk to reason
# about - a 64-bit hash would need a "collisions are unlikely" argument here, and
# this arm's job is to be provably identical or provably wrong.
TRIGRAM_DTYPE = np.dtype((np.void, 12))

# Difference above which a char-similarity arm is treated as disagreeing with the
# reference. Both sides store float32, so this is "not the same float".
TOLERANCE = 1e-6


def _reference_module() -> Any:
    """Import the Phase 0.1 script so the reference arm is the real definition."""
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import analyze_name_differences as module  # noqa: PLC0415 - deliberate lazy import

    return module


# ---------------------------------------------------------------------------
# Flattening: the CPU-side work every vectorized arm must pay
# ---------------------------------------------------------------------------
def _code_points(text: str) -> np.ndarray:
    """UTF-32 code points of ``text`` as a uint32 array."""
    return np.frombuffer(text.encode("utf-32-le"), dtype="<u4")


def _trigram_matrix(text: str) -> Optional[np.ndarray]:
    """Unique trigrams of ``text`` as a void12 array, or ``None`` if too short."""
    codes = _code_points(text)
    if codes.size < 3:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(codes, 3)
    return np.unique(np.ascontiguousarray(windows).view(TRIGRAM_DTYPE))


def _char_matrix(text: str) -> np.ndarray:
    """Unique characters of ``text`` as a void12 array, for the short-string branch."""
    codes = _code_points(text)
    if codes.size == 0:
        return np.empty(0, dtype=TRIGRAM_DTYPE)
    return np.unique(np.repeat(codes, 3).reshape(-1, 3).view(TRIGRAM_DTYPE))


def _flatten(
    keys_a: Sequence[str], keys_b: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Flatten both sides into one trigram matrix plus a per-row pair id.

    Phase 0.1 branches on the *pair*, not per string: if either side is shorter
    than three characters then BOTH sides use character sets. Reproducing that
    here is what keeps the arms exactly equal rather than merely similar.

    Returns:
        ``(rows, group, n_pairs)`` - ``rows`` is ``(n_total, 3)`` uint32, ``group``
        is the pair id of each row (``0..n-1`` for side A, ``n..2n-1`` for side B).
    """
    n = len(keys_a)
    sizes = np.zeros((2, n), dtype=np.int64)
    # Indexed by ``side * n + pair`` so the block order matches ``group`` below
    # exactly - concatenating pair-major while emitting group ids side-major
    # would tag every block after the first pair with the wrong pair.
    blocks: list[Optional[np.ndarray]] = [None] * (2 * n)
    for index, (a, b) in enumerate(zip(keys_a, keys_b)):
        short = len(a) < 3 or len(b) < 3
        for side, text in ((0, a), (1, b)):
            matrix = _char_matrix(text) if short else _trigram_matrix(text)
            sizes[side, index] = 0 if matrix is None else matrix.size
            blocks[side * n + index] = matrix

    parts = [block for block in blocks if block is not None and block.size]
    if not parts:
        return np.empty((0, 3), dtype=np.uint32), np.empty(0, dtype=np.int64), n

    stacked = np.concatenate(parts).view("u4").reshape(-1, 3)
    # Side-major, so the two sides of one pair decode as ``id`` and ``id - n``.
    group = np.repeat(np.concatenate((np.arange(n), np.arange(n) + n)), sizes.ravel())
    return stacked, group, n


# ---------------------------------------------------------------------------
# Arm 1: the numpy kernel
# ---------------------------------------------------------------------------
def _numpy_arm(keys_a: Sequence[str], keys_b: Sequence[str]) -> np.ndarray:
    """Char-3-gram Jaccard, one flat vectorized pass over the whole batch.

    Per-pair numpy is a trap here: two sorts over ~11 elements each lose to a
    python ``set`` intersection by more than an order of magnitude. Flattening the
    batch and sorting once is the only shape with a chance, which is why this arm
    exists in this form.
    """
    rows, group, n = _flatten(keys_a, keys_b)
    out = np.zeros(n, dtype=np.float32)
    if rows.size == 0:
        return out

    # Dedupe (pair, trigram): sort by group, then by the three code points.
    order = np.lexsort((rows[:, 0], rows[:, 1], rows[:, 2], group))
    ordered_group = group[order]
    ordered_rows = rows[order]
    first = np.empty(len(ordered_group), dtype=bool)
    first[0] = True
    first[1:] = (ordered_group[1:] != ordered_group[:-1]) | (
        ordered_rows[1:] != ordered_rows[:-1]
    ).any(axis=1)
    index = np.nonzero(first)[0]
    unique_group = ordered_group[index]
    unique_rows = ordered_rows[index]

    size_a = np.bincount(unique_group[unique_group < n], minlength=n)[:n]
    size_b = np.bincount(unique_group[unique_group >= n] - n, minlength=n)[:n]

    # Each (pair, trigram) now appears at most once per side, so a run of length 2
    # in the combined dedupe is exactly a shared trigram.
    pair_group = np.where(unique_group < n, unique_group, unique_group - n)
    order2 = np.lexsort((unique_rows[:, 0], unique_rows[:, 1], unique_rows[:, 2], pair_group))
    g2 = pair_group[order2]
    r2 = unique_rows[order2]
    start = np.empty(len(g2), dtype=bool)
    start[0] = True
    start[1:] = (g2[1:] != g2[:-1]) | (r2[1:] != r2[:-1]).any(axis=1)
    starts = np.nonzero(start)[0]
    run_length = np.diff(np.append(starts, len(g2)))
    shared = np.bincount(g2[starts], weights=(run_length == 2), minlength=n)[:n]

    union = size_a.astype(np.int64) + size_b.astype(np.int64) - shared.astype(np.int64)
    nonzero = union > 0
    out[nonzero] = (shared[nonzero] / union[nonzero]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Arm 2: the same algorithm on the GPU
# ---------------------------------------------------------------------------
def _torch_arm(keys_a: Sequence[str], keys_b: Sequence[str], device: str) -> np.ndarray:
    """The numpy kernel moved to the GPU. Exists to be measured, not to be shipped.

    The flattening stays on the CPU because it is pure python string work - which
    is the crux: this arm pays that cost *and* the transfer cost. PyTorch has no
    lexsort, so the four-key ordering is four stable argsorts applied in reverse
    key priority, which is the standard equivalent.
    """
    import torch  # noqa: PLC0415 - only imported when this arm is requested

    rows, group, n = _flatten(keys_a, keys_b)
    out = np.zeros(n, dtype=np.float32)
    if rows.size == 0:
        return out

    def move(values: np.ndarray) -> Any:
        return torch.from_numpy(np.ascontiguousarray(values.astype(np.int64))).to(device)

    g = move(group)
    c1, c2, c3 = move(rows[:, 0]), move(rows[:, 1]), move(rows[:, 2])

    def lexsort(*keys: Any) -> Any:
        """Stable argsort applied in reverse key priority == a lexsort."""
        order = torch.arange(g.numel(), device=device)
        for key in reversed(keys):
            order = order[torch.argsort(key[order], stable=True)]
        return order

    order = lexsort(g, c1, c2, c3)
    og, o1, o2, o3 = g[order], c1[order], c2[order], c3[order]
    first = torch.ones_like(og, dtype=torch.bool)
    first[1:] = (og[1:] != og[:-1]) | (o1[1:] != o1[:-1]) | (o2[1:] != o2[:-1]) | (o3[1:] != o3[:-1])
    index = torch.nonzero(first).squeeze(1)
    ug, u1, u2, u3 = og[index], o1[index], o2[index], o3[index]

    size_a = torch.bincount(ug[ug < n], minlength=n)[:n]
    size_b = torch.bincount(ug[ug >= n] - n, minlength=n)[:n]

    pair_group = torch.where(ug < n, ug, ug - n)
    order2 = lexsort(pair_group, u1, u2, u3)
    g2, k1, k2, k3 = pair_group[order2], u1[order2], u2[order2], u3[order2]
    start = torch.ones_like(g2, dtype=torch.bool)
    start[1:] = (g2[1:] != g2[:-1]) | (k1[1:] != k1[:-1]) | (k2[1:] != k2[:-1]) | (k3[1:] != k3[:-1])
    starts = torch.nonzero(start).squeeze(1)
    ends = torch.cat([starts[1:], torch.tensor([g2.numel()], device=device, dtype=starts.dtype)])
    run_length = ends - starts
    shared = torch.bincount(
        g2[starts], weights=(run_length == 2).to(torch.float64), minlength=n
    )[:n]

    union = (size_a + size_b - shared).to(torch.float64)
    nonzero = union > 0
    values = torch.where(
        nonzero, shared.to(torch.float64) / torch.clamp(union, min=1.0), torch.zeros_like(union)
    )
    out = values.to(torch.float32).cpu().numpy()
    # Deliberately no ``empty_cache()`` here: it sits inside the timed region, and
    # it would both charge the GPU arm for a driver round-trip it does not need in
    # a steady-state pipeline and force the *next* run to reallocate from a cold
    # cache. Peak VRAM is reported from ``max_memory_allocated``, which is a
    # high-water mark and is unaffected by holding the cache warm.
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _sorted_key_table(config: dict, split: str, source: str) -> tuple[np.ndarray, np.ndarray]:
    """Numeric entity suffixes and ``name_key`` values, sorted by suffix.

    Sorted because the lookup below is a ``searchsorted``; the prepared file is in
    file order, which is not sorted, so relying on it would silently produce
    wrong pairs.
    """
    table = pd.read_csv(
        prepared_path(config, split, source),
        sep="\t",
        usecols=["entity_id", NAME_KEY],
        dtype=str,
    )
    numeric = pd.to_numeric(table["entity_id"].str.slice(3), errors="coerce").to_numpy()
    keys = table[NAME_KEY].to_numpy(dtype=object)
    order = np.argsort(numeric, kind="stable")
    return numeric[order], keys[order]


def _load_pairs(
    config: dict, sample: int, split: str, logger: logging.Logger
) -> tuple[list[str], list[str]]:
    """Sample ``sample`` real pairs' name keys from the prepared tables.

    Real data, not synthetic: the whole question is how these arms behave on the
    actual string lengths and scripts in the corpus. Keys rather than raw names
    because the char-3-gram signal is defined on ``name_key``.
    """
    ground_truth = load_ground_truth(config, log=logger)
    lengths = ground_truth.lengths()
    n_pairs = len(ground_truth.codes)

    if sample < n_pairs:
        # Evenly spaced choice, seeded so a re-run reproduces the timing exactly.
        generator = np.random.default_rng(42)
        picked = np.sort(generator.choice(n_pairs, size=sample, replace=False))
    else:
        picked = np.arange(n_pairs)

    owners = np.repeat(np.arange(len(ground_truth.entity_ids)), lengths)
    owner_of_pair = owners[picked]
    target_codes = ground_truth.codes[picked]
    source_of_pair = (target_codes // 10**10).astype(np.int8)

    s1_ids, s1_keys = _sorted_key_table(config, split, "source1")
    keys_a = np.full(len(picked), "", dtype=object)
    wanted = ground_truth.entity_ids[owner_of_pair] % 10**10
    slots = np.searchsorted(s1_ids, wanted)
    np.clip(slots, 0, max(len(s1_ids) - 1, 0), out=slots)
    keys_a[slots < len(s1_ids)] = s1_keys[slots[slots < len(s1_ids)]]
    del s1_ids, s1_keys

    keys_b = np.full(len(picked), "", dtype=object)
    for source, code in (("source2", 2), ("source3", 3)):
        mask = source_of_pair == code
        if not mask.any():
            continue
        table_ids, table_keys = _sorted_key_table(config, split, source)
        wanted_b = target_codes[mask] % 10**10
        slots_b = np.searchsorted(table_ids, wanted_b)
        np.clip(slots_b, 0, max(len(table_ids) - 1, 0), out=slots_b)
        hit = table_ids[slots_b] == wanted_b
        resolved = np.full(len(wanted_b), "", dtype=object)
        resolved[hit] = table_keys[slots_b[hit]]
        keys_b[mask] = resolved
        del table_ids, table_keys

    return [str(value) for value in keys_a], [str(value) for value in keys_b]


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def _synchronize() -> None:
    """Block until any queued CUDA work has actually finished.

    CUDA launches are asynchronous, so without this a GPU arm that returns a
    device tensor is timed at kernel *launch* cost and can appear arbitrarily
    fast while doing nothing. A benchmark that can be fooled that way is worse
    than no benchmark, because it would send the pipeline the wrong way.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _benchmark_arm(
    function: Callable[[], np.ndarray],
    repeat: int,
    logger: logging.Logger,
    label: str,
) -> tuple[np.ndarray, float]:
    """Run ``function`` once to warm up, then ``repeat`` times; report the best.

    The best rather than the mean: the fastest observed run is the least polluted
    by whatever else the machine was doing, and this is a "how fast can this go"
    question, not a latency-distribution one.
    """
    function()
    _synchronize()
    best = float("inf")
    result = None
    for _ in range(max(1, repeat)):
        started = time.perf_counter()
        result = function()
        _synchronize()
        best = min(best, time.perf_counter() - started)
    logger.info("  %-10s %8.3f s", label, best)
    return result if result is not None else np.zeros(0, dtype=np.float32), best


def _run_pair_pass(
    analyzer: Any,
    names_a: np.ndarray,
    names_b: np.ndarray,
    empty: np.ndarray,
    n: int,
    chunk_pairs: int,
    workers: int,
    window_size: int,
) -> float:
    """Time the real per-pair statistic over the sample.

    Mirrors the analyzer's own bounded sliding window rather than using
    ``Executor.map``, which would queue every chunk at once; the number measured
    here has to be the number the pipeline will actually see.
    """
    chunks = analyzer._iter_pair_chunks(
        names_a, names_b, names_a, names_b, empty, empty, n, chunk_pairs
    )
    started = time.perf_counter()
    if workers == 1:
        for _block, payload in chunks:
            analyzer._pair_statistics(payload, True)
        return time.perf_counter() - started

    window: deque = deque()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _block, payload in chunks:
            window.append(pool.submit(analyzer._pair_statistics, payload, True))
            if len(window) >= window_size:
                window.popleft().result()
        while window:
            window.popleft().result()
    return time.perf_counter() - started


def _worker_sweep(
    keys_a: list[str],
    keys_b: list[str],
    config: dict,
    auto_workers: int,
    logger: logging.Logger,
) -> None:
    """Measure the real pair pass across worker counts and chunk sizes.

    This is the sweep that decides the HPC command. The char signal alone is far
    too cheap to stand in for it, so the whole per-pair statistic runs.
    """
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import analyze_blocking_statistics as analyzer  # noqa: PLC0415

    n = len(keys_a)
    names_a = np.asarray(keys_a, dtype=object)
    names_b = np.asarray(keys_b, dtype=object)
    # Addresses are not the subject of this sweep; empty ones keep the address
    # signal out of the timing without changing the code path's shape.
    empty = np.full(n, "", dtype=object)

    counts = sorted({1, auto_workers})
    for workers in counts:
        for chunk_pairs in (10_000, 50_000):
            if workers > 1 and n <= chunk_pairs:
                continue
            elapsed = _run_pair_pass(
                analyzer,
                names_a,
                names_b,
                empty,
                n,
                chunk_pairs,
                workers,
                max(2, workers * 4),
            )
            logger.info(
                "  workers=%-4d chunk_pairs=%-7s %8.3f s  (%.0f pairs/s)",
                workers,
                f"{chunk_pairs:,}",
                elapsed,
                n / max(elapsed, 1e-9),
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Phase 0.2-0.5 hot paths on real pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sample", type=int, default=200_000, help="pairs to sample")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--arms",
        default="reference,numpy,torch",
        help="comma-separated subset of reference,numpy,torch",
    )
    parser.add_argument("--repeat", type=int, default=1, help="timed runs per arm")
    parser.add_argument("--skip-workers", action="store_true", help="skip the worker sweep")
    parser.add_argument("--device", default=None, help="torch device for the GPU arm")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    logger = setup_logging(
        LOG_NAME,
        log_dir=config["resolved"]["log_dir"],
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )

    requested = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    unknown = [arm for arm in requested if arm not in ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}; expected a subset of {ALL_ARMS}")

    logger.info("=" * 78)
    logger.info("Phase 0.2-0.5 benchmark")
    hardware = detect_hardware()
    for line in format_hardware_report(hardware).splitlines():
        logger.info("  %s", line)
    logger.info("=" * 78)

    keys_a, keys_b = _load_pairs(config, args.sample, args.split, logger)
    n = len(keys_a)
    mean_length = float(np.mean([len(value) for value in keys_a])) if n else 0.0
    logger.info(
        "sampled %s real pairs (mean name_key length %.1f chars)", f"{n:,}", mean_length
    )
    if n == 0:
        raise SystemExit("no pairs sampled; nothing to benchmark")

    reference = _reference_module()
    baseline = np.array(
        [reference._trigram_jaccard(a, b) for a, b in zip(keys_a, keys_b)], dtype=np.float32
    )

    arms: dict[str, Callable[[], np.ndarray]] = {}
    if "reference" in requested:
        arms["reference"] = lambda: np.array(
            [reference._trigram_jaccard(a, b) for a, b in zip(keys_a, keys_b)], dtype=np.float32
        )
    if "numpy" in requested:
        arms["numpy"] = lambda: _numpy_arm(keys_a, keys_b)
    if "torch" in requested:
        device = args.device or "cuda"
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning(
                    "torch GPU arm skipped: torch %s cannot see a CUDA device. A CPU-only "
                    "torch build cannot, even when nvidia-smi lists a GPU.",
                    torch.__version__,
                )
            else:
                arms["torch"] = lambda: _torch_arm(keys_a, keys_b, device)
        except ImportError:
            logger.warning("torch GPU arm skipped: torch is not installed")

    logger.info("-" * 78)
    logger.info("char-3-gram arms (%s pairs)", f"{n:,}")
    timings: dict[str, float] = {}
    mismatches: dict[str, int] = {}
    for name, function in arms.items():
        result, best = _benchmark_arm(function, max(1, args.repeat), logger, name)
        timings[name] = best
        bad = int(np.sum(np.abs(np.asarray(result, dtype=np.float64) - baseline) > TOLERANCE))
        mismatches[name] = bad
        if bad:
            logger.error(
                "  %s DISAGREES with the reference on %s pairs - its timing is not usable",
                name,
                f"{bad:,}",
            )

    logger.info("-" * 78)
    logger.info("verdict")
    baseline_seconds = timings.get("reference")
    for name, seconds in sorted(timings.items(), key=lambda item: item[1]):
        verdict = "OK" if not mismatches.get(name) else f"{mismatches[name]:,} MISMATCHES"
        if baseline_seconds and name != "reference":
            logger.info(
                "  %-10s %8.3f s  %.2fx vs reference  [%s]",
                name,
                seconds,
                baseline_seconds / seconds,
                verdict,
            )
        else:
            logger.info(
                "  %-10s %8.3f s  (%.0f pairs/s)  [%s]",
                name,
                seconds,
                n / max(seconds, 1e-9),
                verdict,
            )

    clean = {name: seconds for name, seconds in timings.items() if not mismatches.get(name)}
    fastest = min(clean, key=clean.get) if clean else None
    if fastest and fastest != "reference":
        logger.info(
            "note: %r wins the char signal alone, but the analyzer still runs the reference "
            "definition. Moving it would need a bit-identity test and a reason to prefer it.",
            fastest,
        )
    elif fastest == "reference":
        logger.info(
            "note: the reference implementation is the fastest correct arm here, which is "
            "what the CPU-first policy predicts for short strings. No GPU path is warranted."
        )

    logger.info("peak CPU RSS: %s", human_bytes(peak_rss_bytes() or current_rss_bytes() or 0))
    if "torch" in arms:
        import torch

        logger.info("peak GPU memory: %s", human_bytes(int(torch.cuda.max_memory_allocated())))

    if not args.skip_workers:
        from src.utils import resolve_workers

        auto = resolve_workers(
            0, config.get("compute", {}).get("num_workers", 0), n, logger=None
        )
        logger.info("-" * 78)
        logger.info("worker / chunk sweep (full per-pair statistics; auto=%d)", auto)
        _worker_sweep(keys_a, keys_b, config, auto, logger)

    logger.info("=" * 78)
    logger.info("done")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
