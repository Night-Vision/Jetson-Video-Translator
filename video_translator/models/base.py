from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator


class BaseModel(ABC):
    """Abstract base enforcing an explicit load → use → unload lifecycle.

    All GPU/RAM-heavy models MUST be used via the lifecycle() context manager
    so memory cleanup is guaranteed even when exceptions are raised.

    Example::

        with WhisperModel(config).lifecycle() as model:
            segments = model.transcribe(audio_path)
    """

    @abstractmethod
    def _load(self) -> None:
        """Allocate the model into memory or start its backing service."""

    @abstractmethod
    def _unload(self) -> None:
        """Release the model and reclaim memory / stop its backing service."""

    @contextmanager
    def lifecycle(self) -> Generator[BaseModel, None, None]:
        """Context manager that calls _load on enter and _unload + gc on exit."""
        self._load()
        try:
            yield self
        finally:
            self._unload()
            gc.collect()
