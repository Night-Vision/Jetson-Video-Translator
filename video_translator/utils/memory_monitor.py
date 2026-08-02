from __future__ import annotations

import logging
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
