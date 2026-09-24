"""Shared utilities: logging, progress reporting, hashing, memory helpers.

Kept deliberately small and dependency-light so the HPC environment only needs
numpy + pandas + pyyaml (+ optional tqdm/psutil) to run the preprocessing,
indexing and candidate-generation stages.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, TypeVar

import numpy as np

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Optional dependencies. The pipeline must run without them.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import guard
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:  # pragma: no cover
    _tqdm = None
    _HAS_TQDM = False

try:  # pragma: no cover
    import psutil as _psutil
except ImportError:  # pragma: no cover
    _psutil = None


_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    name: str,
    log_dir: Optional[str | os.PathLike] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure a logger that writes to stderr and (optionally) a log file.

    Idempotent: calling it twice with the same name does not duplicate handlers,
    which matters because scripts call it at import time and again in ``main``.

    Args:
        name: logger name, conventionally the module or script name.
        log_dir: directory for the log file. Created if missing.
        level: logging level.
        log_file: explicit file name; defaults to ``<name>.log`` in ``log_dir``.

    Returns:
        The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (log_file or f"{name}.log")
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def track(
    iterable: Iterable[T],
    total: Optional[int] = None,
    desc: str = "",
    logger: Optional[logging.Logger] = None,
    log_every: Optional[int] = None,
    unit: str = "rows",
) -> Iterator[T]:
    """Iterate with progress reporting that degrades gracefully.

    Uses tqdm when it is installed and stderr is a terminal (a progress bar is
    noise in an HPC batch log). Otherwise, logs every ``log_every`` items, or
    every 10% when ``log_every`` is not given.

    Args:
        iterable: the iterable to wrap.
        total: expected number of items, if known.
        desc: label shown in the progress bar / log line.
        logger: logger used for the non-tqdm path.
        log_every: item interval between log lines in the non-tqdm path.
        unit: unit name for the progress output.
    """
    if _HAS_TQDM and sys.stderr.isatty():
        yield from _tqdm(iterable, total=total, desc=desc, unit=unit)
        return

    if logger is None:
        logger = logging.getLogger("progress")

    if log_every is None:
        log_every = max(1, (total // 10) if total else 100_000)

    start = time.time()
    count = 0
    for item in iterable:
        yield item
        count += 1
        if count % log_every == 0:
            elapsed = time.time() - start
            rate = count / elapsed if elapsed > 0 else 0.0
            if total:
                pct = 100.0 * count / total
                eta = (total - count) / rate if rate > 0 else float("nan")
                logger.info(
                    "%s: %s/%s (%.1f%%) | %.0f %s/s | ETA %.1f min",
                    desc or "progress",
                    f"{count:,}",
                    f"{total:,}",
                    pct,
                    rate,
                    unit,
                    eta / 60.0,
                )
            else:
                logger.info(
                    "%s: %s %s | %.0f %s/s | %.1f min",
                    desc or "progress",
                    f"{count:,}",
                    unit,
                    rate,
                    unit,
                    elapsed / 60.0,
                )
    logger.info("%s: done, %s %s in %.1f min", desc or "progress", f"{count:,}", unit, (time.time() - start) / 60.0)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python/random/numpy. Deliberately does not touch torch.

    Importing torch here would slow down the CPU-only stages for no benefit.
    Training scripts call ``torch.manual_seed`` themselves when torch is present.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device string without ever requiring a GPU.

    The pipeline must run to completion on a CPU-only machine (the dev laptop has
    no GPU). Embedding, dense retrieval and transformer stages call this and get
    ``"cpu"`` when torch is absent or CUDA is unavailable, so nothing has to be
    special-cased further down.

    Args:
        preference: ``"auto"`` (detect), or an explicit device such as
            ``"cpu"``/``"cuda"``/``"cuda:1"``/``"mps"``.

    Returns:
        A device string suitable for ``torch.device(...)``.
    """
    if preference and preference != "auto":
        return preference
    try:
        import torch  # imported lazily: CPU-only stages never pay for it
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(logger: Optional[logging.Logger] = None) -> dict:
    """Report the compute environment. Logged at the start of GPU-capable stages."""
    info: dict = {"device": resolve_device(), "torch": None, "gpu": None, "gpu_memory": None}
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = human_bytes(torch.cuda.get_device_properties(0).total_memory)
    except ImportError:
        pass
    if logger:
        logger.info(
            "compute: device=%s torch=%s gpu=%s%s",
            info["device"],
            info["torch"] or "not installed",
            info["gpu"] or "none",
            f" ({info['gpu_memory']})" if info["gpu_memory"] else "",
        )
    return info


# ---------------------------------------------------------------------------
# Stable hashing
# ---------------------------------------------------------------------------
def stable_hash64(values: str | Sequence[str]) -> np.ndarray | int:
    """Hash strings to 64-bit unsigned ints.

    Deliberately NOT python's builtin ``hash()``: that is salted per process
    (PYTHONHASHSEED), so an index built in one run would not be readable in the
    next. blake2b is stable across processes, machines and python versions.

    Collisions are possible in principle (64-bit space); every consumer here
    verifies the actual string after a hash lookup, so a collision degrades to a
    miss rather than a wrong answer. With ~4M keys the birthday probability is
    ~4e-7.

    Args:
        values: a single string, or an iterable/pandas Series of strings.

    Returns:
        A python int for a single string, else a uint64 numpy array.
    """
    if isinstance(values, str):
        return int.from_bytes(hashlib.blake2b(values.encode("utf-8"), digest_size=8).digest(), "little")

    import pandas as pd  # local import keeps this module importable without pandas

    if isinstance(values, pd.Series):
        raw = values.array
    else:
        raw = np.asarray(values, dtype=object)

    out = np.empty(len(raw), dtype=np.uint64)
    blake = hashlib.blake2b
    for i, value in enumerate(raw):
        out[i] = int.from_bytes(blake(value.encode("utf-8"), digest_size=8).digest(), "little")
    return out


# ---------------------------------------------------------------------------
# Memory / diagnostics
# ---------------------------------------------------------------------------
def human_bytes(num_bytes: float) -> str:
    """Format a byte count as a human-readable string."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def nbytes_of(*arrays: np.ndarray) -> str:
    """Total size of the given arrays, formatted."""
    return human_bytes(sum(a.nbytes for a in arrays if a is not None))


def log_memory(logger: logging.Logger, label: str = "") -> Optional[str]:
    """Log current RSS if psutil is available. Returns the formatted string."""
    if _psutil is None:
        return None
    rss = _psutil.Process(os.getpid()).memory_info().rss
    text = human_bytes(rss)
    logger.info("RSS%s: %s", f" ({label})" if label else "", text)
    return text


# ---------------------------------------------------------------------------
# Entity id codec
# ---------------------------------------------------------------------------
# Challenge ids look like "S2-166376419": a source prefix plus a numeric suffix
# of 2-9 digits (verified against the full training set). We pack them into
# int64 as ``source_code * 10**10 + numeric``.
#
# Why bother: the ground truth holds 7.6M matched ids. As a numpy unicode array
# ('<U13') that is ~400MB; as int64 it is ~61MB. Integer codes also let the
# evaluation stage do set comparisons with np.isin instead of python string sets,
# which is several times faster on 7.6M pairs.
ID_SOURCE_CODES = {"S1": 1, "S2": 2, "S3": 3}
ID_SOURCE_NAMES = {code: name for name, code in ID_SOURCE_CODES.items()}
ID_NUMERIC_MODULUS = 10**10
_ID_MAX_NUMERIC = ID_NUMERIC_MODULUS - 1


def split_entity_id(entity_id: str) -> tuple[str, int]:
    """Split ``"S2-166376419"`` into ``("S2", 166376419)``."""
    prefix, _, numeric = entity_id.partition("-")
    if not numeric or not numeric.isdigit():
        raise ValueError(f"malformed entity id: {entity_id!r}")
    return prefix, int(numeric)


def encode_entity_id(entity_id: str) -> int:
    """Pack a single entity id into an int64-safe integer code."""
    prefix, numeric = split_entity_id(entity_id)
    try:
        source_code = ID_SOURCE_CODES[prefix]
    except KeyError:
        raise ValueError(f"unknown source prefix {prefix!r} in {entity_id!r}") from None
    if numeric > _ID_MAX_NUMERIC:  # pragma: no cover - guarded by the codec design
        raise ValueError(f"numeric part too large to pack: {entity_id!r}")
    return source_code * ID_NUMERIC_MODULUS + numeric


def decode_entity_id(code: int) -> str:
    """Inverse of :func:`encode_entity_id`."""
    code = int(code)
    source_code, numeric = divmod(code, ID_NUMERIC_MODULUS)
    try:
        prefix = ID_SOURCE_NAMES[source_code]
    except KeyError:
        raise ValueError(f"unknown source code {source_code} in packed id {code}") from None
    return f"{prefix}-{numeric}"


def decode_entity_ids(codes: np.ndarray) -> np.ndarray:
    """Vectorized :func:`decode_entity_id`. Returns an object-dtype string array."""
    codes = np.asarray(codes, dtype=np.int64)
    source_codes, numerics = np.divmod(codes, ID_NUMERIC_MODULUS)
    if len(codes) and not np.isin(source_codes, list(ID_SOURCE_NAMES)).all():
        raise ValueError("packed ids contain an unknown source code")
    prefixes = np.array([ID_SOURCE_NAMES[int(s)] for s in np.unique(source_codes)], dtype=object)
    lookup = {int(s): ID_SOURCE_NAMES[int(s)] for s in np.unique(source_codes)}
    out = np.empty(len(codes), dtype=object)
    # Group by source so the f-string loop is not re-deciding the prefix each row.
    for source_code, prefix in lookup.items():
        mask = source_codes == source_code
        if mask.any():
            out[mask] = [f"{prefix}-{n}" for n in numerics[mask]]
    del prefixes
    return out


def encode_entity_ids(values) -> np.ndarray:
    """Vectorized :func:`encode_entity_id` over a Series / array / list."""
    import pandas as pd

    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    prefixes = series.str.slice(0, 2)
    numerics = series.str.slice(3)
    if not numerics.str.isdigit().all():
        bad = series[~numerics.str.isdigit()].head(5).tolist()
        raise ValueError(f"malformed entity ids (expected e.g. 'S2-123'), got: {bad}")
    numbers = numerics.astype("int64").to_numpy()
    if len(numbers) and numbers.max() > _ID_MAX_NUMERIC:  # pragma: no cover
        raise ValueError("numeric part too large to pack")
    source_codes = prefixes.map(ID_SOURCE_CODES)
    if source_codes.isna().any():
        bad = series[source_codes.isna()].head(5).tolist()
        raise ValueError(f"unknown source prefix in ids: {bad}")
    return source_codes.to_numpy(dtype=np.int64) * ID_NUMERIC_MODULUS + numbers


# ---------------------------------------------------------------------------
# Tiny IO helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | os.PathLike, payload: Any) -> None:
    """Write JSON atomically (temp file + replace) so partial runs can't corrupt."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, target)


def read_json(path: str | os.PathLike) -> Any:
    """Read a JSON file produced by :func:`write_json`."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def fmt_int(value: int | float) -> str:
    """Thousands-separated integer string for log messages."""
    return f"{int(value):,}"
