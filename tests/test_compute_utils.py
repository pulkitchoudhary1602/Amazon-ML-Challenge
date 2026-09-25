"""Synthetic tests for the Phase 0.2-0.5 compute plumbing.

These cover the infrastructure the wall-clock optimisation added, not the
analytical definitions - ``tests/test_blocking_statistics.py`` remains the
contract for E/T/C/A, the signal union, the census and the zero-match analysis.

What is asserted here:

* **Hardware detection** parses ``nvidia-smi`` output, survives a node with
  neither ``nvidia-smi`` nor torch, and prefers nvidia-smi over torch so a
  CPU-only torch build does not erase a GPU that physically exists.
* **Worker resolution** honours CLI > config > auto, clamps to the chunk count,
  and never returns 0 (a zero-worker pool would hang, not fail).
* **Window planning** bounds queued payload bytes under a RAM budget, logs the
  adjustment, and still returns a usable ``(window, chunk_pairs)`` when the
  budget is absurdly small - the brief is explicit that a too-large default
  batch must be reduced and reported, never allowed to crash the run.
* **Checkpointing** round-trips, refuses a stale fingerprint, and needs both the
  payload and its marker before it will skip a phase.

Everything is in memory or in a temp directory. Nothing here touches the
dataset, the GPU, or the network.

Runs standalone (``python tests/test_compute_utils.py``) and under pytest.
"""

from __future__ import annotations

import builtins
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.utils as utils  # noqa: E402
from scripts import analyze_blocking_statistics as abs_  # noqa: E402

MIB = 1024 * 1024

# ``nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits``
# Two usable devices, one junk line, one device whose memory is unreadable.
NVIDIA_SMI_CSV = """\
0, NVIDIA A100-SXM4-80GB, 81920, 81100
1, NVIDIA A100-SXM4-80GB, 81920, 40000
this line is not a gpu
2, Tesla V100-SXM2-32GB, [N/A], [N/A]
"""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
class _Capture(logging.Handler):
    """A logging handler that keeps the messages it sees."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _capture_log() -> tuple[logging.Logger, _Capture]:
    logger = logging.getLogger("test_compute_utils")
    logger.setLevel(logging.INFO)
    handler = _Capture()
    logger.addHandler(handler)
    return logger, handler


def _without_torch(function, *args, **kwargs):
    """Call ``function`` with ``import torch`` forced to raise ImportError."""
    real_import = builtins.__import__

    def fake_import(name, *rest, **more):  # noqa: ANN001 - import hook
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is not installed (simulated)")
        return real_import(name, *rest, **more)

    builtins.__import__ = fake_import
    try:
        return function(*args, **kwargs)
    finally:
        builtins.__import__ = real_import


def _patch(module, name: str, value) -> object:
    """Temporarily set ``module.name``, returning a restore callable."""
    original = getattr(module, name)
    setattr(module, name, value)
    return lambda: setattr(module, name, original)


# ---------------------------------------------------------------------------
# hardware detection
# ---------------------------------------------------------------------------
def test_parse_nvidia_smi_reads_devices_and_skips_junk():
    gpus = utils.parse_nvidia_smi(NVIDIA_SMI_CSV)

    assert [gpu["index"] for gpu in gpus] == [0, 1, 2], gpus
    assert gpus[0]["name"] == "NVIDIA A100-SXM4-80GB"
    assert gpus[0]["memory_total"] == 81920 * MIB
    assert gpus[0]["memory_free"] == 81100 * MIB
    # The unparseable line is dropped; the two real devices survive.
    assert len(gpus) == 3


def test_parse_nvidia_smi_keeps_device_with_unreadable_memory():
    """A ``[N/A]`` VRAM figure must not delete the device from the report."""
    gpus = utils.parse_nvidia_smi(NVIDIA_SMI_CSV)

    v100 = gpus[2]
    assert v100["name"] == "Tesla V100-SXM2-32GB"
    assert v100["memory_total"] is None
    assert v100["memory_free"] is None


def test_parse_nvidia_smi_tolerates_empty_and_unusable_lines():
    assert utils.parse_nvidia_smi("") == []
    assert utils.parse_nvidia_smi("\n\n") == []
    # A line with no parseable index is not a device record.
    assert utils.parse_nvidia_smi("not an index, Some GPU, 1024, 512") == []
    # A name-only line IS a device, just one whose memory we could not read -
    # the same treatment the "[N/A]" case gets.
    only_name = utils.parse_nvidia_smi("0, Some GPU")
    assert len(only_name) == 1
    assert only_name[0]["memory_total"] is None


def test_detect_hardware_survives_missing_nvidia_smi_and_torch():
    restore = _patch(utils, "_nvidia_smi_gpus", lambda: [])
    try:
        info = _without_torch(utils.detect_hardware)
    finally:
        restore()

    assert info["gpus"] == []
    assert info["torch"] is None
    assert info["cuda_available"] is False
    assert info["cuda_device_count"] == 0
    # The keys the report and the config plumbing read must all be present.
    for key in (
        "cpu_logical",
        "cpu_physical",
        "ram_total",
        "ram_available",
        "gpus",
        "gpu_source",
        "torch",
        "cuda_available",
        "cuda_device_count",
    ):
        assert key in info, key
    assert info["cpu_logical"] and info["cpu_logical"] >= 1


def test_detect_hardware_prefers_nvidia_smi_over_torch():
    """A CPU-only torch build must not erase a GPU that nvidia-smi can see."""
    listed = utils.parse_nvidia_smi(NVIDIA_SMI_CSV)
    restore = _patch(utils, "_nvidia_smi_gpus", lambda: list(listed))
    try:
        info = utils.detect_hardware()
    finally:
        restore()

    assert [gpu["index"] for gpu in info["gpus"]] == [0, 1, 2]
    assert info["gpu_source"] == "nvidia-smi"
    # This is the whole point: the devices are reported even when torch cannot
    # use them. Asserting the report distinguishes the two is what proves the
    # distinction is actually carried through to the human-readable output.
    if not info["cuda_available"]:
        assert "present but not usable by torch" in utils.format_hardware_report(info)


def test_format_hardware_report_distinguishes_gpu_present_but_unusable():
    info = {
        "cpu_logical": 12,
        "cpu_physical": 6,
        "ram_total": 8 * 1024**3,
        "ram_available": 900 * MIB,
        "gpus": [{"index": 0, "name": "NVIDIA RTX 3050", "memory_total": 4 * 1024**3, "memory_free": None}],
        "gpu_source": "nvidia-smi",
        "torch": "2.14.0+cpu",
        "cuda_available": False,
        "cuda_device_count": 0,
    }
    report = utils.format_hardware_report(info)

    assert "6 physical / 12 logical" in report
    assert "NVIDIA RTX 3050" in report
    assert "present but not usable by torch" in report
    assert "2.14.0+cpu" in report


def test_format_hardware_report_renders_a_gpureless_node():
    info = {
        "cpu_logical": 64,
        "cpu_physical": 32,
        "ram_total": 256 * 1024**3,
        "ram_available": None,
        "gpus": [],
        "gpu_source": "nvidia-smi",
        "torch": None,
        "cuda_available": False,
        "cuda_device_count": 0,
    }
    report = utils.format_hardware_report(info)

    assert "gpus: none detected" in report
    assert "cuda: no (torch not installed)" in report
    assert "usable by torch" not in report


# ---------------------------------------------------------------------------
# worker resolution
# ---------------------------------------------------------------------------
def test_auto_worker_count_prefers_physical_cores():
    restore = _patch(utils, "_physical_cpu_count", lambda: 6)
    try:
        assert utils.auto_worker_count() == 6
    finally:
        restore()

    # Falls back to logical cores when psutil cannot tell us.
    restore = _patch(utils, "_physical_cpu_count", lambda: None)
    try:
        assert utils.auto_worker_count() == max(1, __import__("os").cpu_count() or 1)
    finally:
        restore()


def test_resolve_workers_precedence_and_clamping():
    restore = _patch(utils, "_physical_cpu_count", lambda: 8)
    try:
        # auto -> physical cores, when there is enough work to fill them.
        assert utils.resolve_workers(0, 0, 1000) == 8
        # config wins over auto.
        assert utils.resolve_workers(0, 3, 1000) == 3
        # CLI wins over config.
        assert utils.resolve_workers(5, 3, 1000) == 5
        # Never more workers than chunks.
        assert utils.resolve_workers(64, 0, 4) == 4
        # A single chunk still gets one worker.
        assert utils.resolve_workers(0, 0, 1) == 1
        assert utils.resolve_workers(0, 0, 0) == 1
    finally:
        restore()


def test_resolve_workers_never_returns_zero():
    restore = _patch(utils, "_physical_cpu_count", lambda: 4)
    try:
        for requested in (0, -1, 1, 4, 999):
            for n_chunks in (0, 1, 2, 1000):
                assert utils.resolve_workers(requested, 0, n_chunks) >= 1
    finally:
        restore()


def test_resolve_workers_logs_the_choice_with_both_core_counts():
    restore = _patch(utils, "_physical_cpu_count", lambda: 8)
    logger, handler = _capture_log()
    try:
        utils.resolve_workers(0, 0, 100, logger=logger, label="pair workers")
    finally:
        restore()

    assert len(handler.messages) == 1
    message = handler.messages[0]
    assert "pair workers" in message
    assert "physical cpu=8" in message


# ---------------------------------------------------------------------------
# window / chunk sizing
# ---------------------------------------------------------------------------
def test_plan_inflight_window_keeps_the_default_when_the_budget_is_large():
    window, chunk_pairs = utils.plan_inflight_window(
        workers=8, chunk_pairs=100_000, bytes_per_pair=200, budget_bytes=64 * 1024**3
    )

    assert window == 8 * 4
    assert chunk_pairs == 100_000


def test_plan_inflight_window_shrinks_under_a_small_budget_and_logs_it():
    """The payload is reduced to fit; the window is only cut if that is not enough."""
    logger, handler = _capture_log()
    window, chunk_pairs = utils.plan_inflight_window(
        workers=64,
        chunk_pairs=100_000,
        bytes_per_pair=200,
        budget_bytes=64 * MIB,
        logger=logger,
        label="pair chunks",
    )

    assert chunk_pairs < 100_000, chunk_pairs
    assert window >= 2 and chunk_pairs >= 1
    # Whatever it chose, the queued payload now fits the budget it was given.
    assert window * chunk_pairs * 200 <= 64 * MIB
    assert handler.messages, "the adjustment must be reported, not made silently"
    assert "chunk_pairs" in handler.messages[0]


def test_plan_inflight_window_cuts_the_window_when_the_chunk_floor_is_hit():
    """With chunk_pairs already at 1, the window is the only thing left to shrink."""
    logger, handler = _capture_log()
    window, chunk_pairs = utils.plan_inflight_window(
        workers=64, chunk_pairs=1, bytes_per_pair=8 * MIB, budget_bytes=64 * MIB, logger=logger
    )

    assert chunk_pairs == 1
    assert window < 64 * 4, window
    assert window == 8, window  # 64 MiB / 8 MiB per pair
    assert handler.messages


def test_plan_inflight_window_stays_usable_under_a_degenerate_budget():
    """A budget smaller than one pair must not produce a zero-sized pool."""
    logger, handler = _capture_log()
    for budget in (0, 1, 100, 1000):
        window, chunk_pairs = utils.plan_inflight_window(
            workers=64, chunk_pairs=100_000, bytes_per_pair=10_000, budget_bytes=budget, logger=logger
        )
        assert window >= 2, (budget, window)
        assert chunk_pairs >= 1, (budget, chunk_pairs)
    assert handler.messages


def test_plan_inflight_window_handles_zero_valued_inputs():
    window, chunk_pairs = utils.plan_inflight_window(0, 0, 0, 0)
    assert window >= 2 and chunk_pairs >= 1


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------
def _checkpoint(directory: Path, fingerprint: str = "fp-1", enabled: bool = True):
    logger, _handler = _capture_log()
    return abs_._PhaseCheckpoint(directory, fingerprint, enabled, logger)


def test_checkpoint_round_trip_restores_arrays():
    """The real payload's dtypes, because they are what make the load work."""
    with tempfile.TemporaryDirectory() as tmp:
        writer = _checkpoint(Path(tmp))
        writer.save(
            "pair_statistics",
            name_matrix=np.arange(12, dtype=np.float32).reshape(6, 2),
            name_equal=np.array([True, False, True, True, False, True]),
            category=np.full(6, -1, dtype=np.int16),
            n_unknown_df=np.asarray(41, dtype=np.int64),
        )

        # The payload and its marker both land, and no scratch file survives.
        assert (Path(tmp) / "_ckpt" / "pair_statistics.npz").exists()
        assert (Path(tmp) / "_ckpt" / "pair_statistics.done.json").exists()
        assert not list((Path(tmp) / "_ckpt").glob("*.tmp"))

        restored = _checkpoint(Path(tmp)).load("pair_statistics")
        assert restored is not None
        assert np.array_equal(restored["name_matrix"], np.arange(12, dtype=np.float32).reshape(6, 2))
        assert list(restored["name_equal"]) == [True, False, True, True, False, True]
        assert restored["category"].dtype == np.int16
        assert int(restored["n_unknown_df"]) == 41


def test_checkpoint_refuses_an_object_payload_instead_of_crashing():
    """``allow_pickle=False`` means an object array cannot silently round-trip.

    The pair pass stores only numeric arrays, so this cannot happen today; the
    point is that if someone later adds a string array the checkpoint degrades to
    "recompute" rather than raising out of the middle of a long run.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _checkpoint(Path(tmp))
        ckpt.save("pair_statistics", categories=np.array(["a", "b"], dtype=object))

        assert ckpt.load("pair_statistics") is None


def test_checkpoint_ignores_a_stale_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        _checkpoint(Path(tmp), fingerprint="fp-1").save("pair_statistics", x=np.arange(4))

        # Same inputs -> resumable.
        assert _checkpoint(Path(tmp), fingerprint="fp-1").load("pair_statistics") is not None
        # Different inputs (new chunk size, new pair count, new corpus) -> recompute.
        assert _checkpoint(Path(tmp), fingerprint="fp-2").load("pair_statistics") is None


def test_checkpoint_needs_both_payload_and_marker():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _checkpoint(Path(tmp))
        # Nothing written at all.
        assert ckpt.load("pair_statistics") is None

        ckpt.save("pair_statistics", x=np.arange(4))
        # A marker without its payload is not a completed phase.
        (Path(tmp) / "_ckpt" / "pair_statistics.npz").unlink()
        assert ckpt.load("pair_statistics") is None

        # ...and a payload without its marker is not one either.
        ckpt.save("pair_statistics", x=np.arange(4))
        (Path(tmp) / "_ckpt" / "pair_statistics.done.json").unlink()
        assert ckpt.load("pair_statistics") is None


def test_checkpoint_disabled_writes_nothing_and_loads_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _checkpoint(Path(tmp), enabled=False)
        ckpt.save("pair_statistics", x=np.arange(4))

        assert not (Path(tmp) / "_ckpt").exists()
        assert ckpt.load("pair_statistics") is None

        # And a run without --resume must not consume an existing checkpoint.
        _checkpoint(Path(tmp)).save("pair_statistics", x=np.arange(4))
        assert _checkpoint(Path(tmp), enabled=False).load("pair_statistics") is None


def test_checkpoint_fingerprint_tracks_every_input():
    args = abs_.parse_args(
        ["--config", "configs/config.yaml", "--split", "train", "--limit-pairs", "1000", "--chunk-pairs", "500"]
    )
    config = {"resolved": {"prepared_dir": "outputs/prepared"}}
    base = abs_.checkpoint_fingerprint(args, ["S1", "S2"], config, 7)

    # Deterministic for identical inputs...
    assert abs_.checkpoint_fingerprint(args, ["S1", "S2"], config, 7) == base
    # ...and every input the pair pass depends on moves it.
    assert abs_.checkpoint_fingerprint(args, ["S1"], config, 7) != base
    assert abs_.checkpoint_fingerprint(args, ["S1", "S2"], config, 8) != base
    assert (
        abs_.checkpoint_fingerprint(args, ["S1", "S2"], {"resolved": {"prepared_dir": "other"}}, 7) != base
    )

    other_split = abs_.parse_args(["--config", "configs/config.yaml", "--split", "val"])
    assert abs_.checkpoint_fingerprint(other_split, ["S1", "S2"], config, 7) != base

    other_chunk = abs_.parse_args(["--config", "configs/config.yaml", "--chunk-pairs", "999"])
    assert abs_.checkpoint_fingerprint(other_chunk, ["S1", "S2"], config, 7) != base


def test_checkpoint_fingerprint_moves_when_name_categories_toggle():
    with_cats = abs_.parse_args(["--config", "configs/config.yaml"])
    without = abs_.parse_args(["--config", "configs/config.yaml", "--no-name-categories"])
    config = {"resolved": {"prepared_dir": "outputs/prepared"}}

    assert abs_.checkpoint_fingerprint(with_cats, ["S1"], config, 7) != abs_.checkpoint_fingerprint(
        without, ["S1"], config, 7
    )


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
