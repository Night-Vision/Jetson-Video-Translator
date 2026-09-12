from __future__ import annotations

import glob
import logging
import os

import psutil

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger("video_translator.memory_monitor")


def log_vram(label: str, config: Config) -> None:
    """Log GPU VRAM usage to debug output.

    Uses nvidia-ml-py (pynvml) to read unified memory on Jetson.
    Silently skipped if pynvml is not installed or fails.
    """
    if not config.debug:
        return
    try:
        import pynvml  # noqa: PLC0415
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = info.used / 1024 ** 3
        total_gb = info.total / 1024 ** 3
        logger.debug("[VRAM] %s: %.2f GB used / %.2f GB total", label, used_gb, total_gb)
    except Exception:
        pass  # pynvml unavailable — silently skip


def log_memory(label: str, config: Config) -> None:
    """Log RAM usage to stdout when debug mode is enabled."""
    if not config.debug:
        return
    vm = psutil.virtual_memory()
    used_gb = vm.used / 1024 ** 3
    avail_gb = vm.available / 1024 ** 3
    logger.debug("[MEM] %s: %.2f GB used / %.2f GB available", label, used_gb, avail_gb)


def check_memory(min_free_gb: float, label: str = "") -> None:
    """Raise MemoryError if available RAM is below the threshold.

    Args:
        min_free_gb: Minimum acceptable free RAM in gigabytes.
        label:       Optional context label included in the error message.
    """
    avail_gb = psutil.virtual_memory().available / 1024 ** 3
    if avail_gb < min_free_gb:
        ctx = f" ({label})" if label else ""
        raise MemoryError(
            f"Insufficient free RAM{ctx}: "
            f"{avail_gb:.1f} GB available, {min_free_gb} GB required."
        )


# Jetson SoC overcurrent event counters.  These are cumulative since boot and
# carry no timestamp, so the only way to attribute an event to a pipeline stage
# is to sample them at each stage boundary and report the delta.  Overcurrent
# is a power-rail alarm, unrelated to the thermal trip points -- it can fire on
# a cold board when CPU and GPU load spike together under an uncapped nvpmodel
# profile.
_OC_COUNTERS = sorted(glob.glob("/sys/class/hwmon/hwmon*/oc*_event_cnt"))
_oc_previous: dict[str, int] = {}


def _read_oc_counters() -> dict[str, int]:
    counts = {}
    for path in _OC_COUNTERS:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                counts[os.path.basename(path)] = int(fh.read().strip())
        except (OSError, ValueError):
            continue
    return counts


def log_oc(label: str, config: Config | None = None) -> None:
    """Report SoC overcurrent events that fired since the previous call.

    The first call establishes the baseline.  Increments are logged at WARNING
    regardless of debug mode -- an overcurrent event means the board throttled,
    which is worth surfacing on every run.
    """
    current = _read_oc_counters()
    if not current:
        return

    for rail, count in current.items():
        before = _oc_previous.get(rail)
        if before is not None and count > before:
            logger.warning(
                "[OC] %s: %s fired %d time(s) during this stage (total %d) "
                "— board throttled on an overcurrent alarm",
                label, rail, count - before, count,
            )
    _oc_previous.update(current)

    if config is not None and config.debug:
        logger.debug(
            "[OC] %s: %s", label,
            ", ".join(f"{r}={c}" for r, c in sorted(current.items())),
        )
