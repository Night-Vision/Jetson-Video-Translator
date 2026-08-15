from __future__ import annotations

import psutil

from .base import BaseModel


class WhisperModel(BaseModel):
    """Wraps faster-whisper with a RAM guard and an explicit load/unload lifecycle.

    Supports two backends:
      - "faster_whisper" (default): CTranslate2 INT8 via the faster-whisper package.
      - "whisper_trt" (experimental): TensorRT engine via the whisper_trt package.

    torch and faster_whisper are intentionally NOT imported at module level.
    Keeping heavy CUDA imports inside _load() prevents GPU context contamination
    in the parent process — the transcriber runs Whisper in a child subprocess
    precisely so the CUDA context is fully released when inference is done.
    """

    def __init__(
        self,
        model_size: str,
        min_free_ram_gb: float,
        backend: str = "faster_whisper",
        engine_dir: str = "",
    ) -> None:
        self.model_size = model_size
        self.min_free_ram_gb = min_free_ram_gb
        self._model = None

    def _load(self) -> None:
        available_gb = psutil.virtual_memory().available / 1024 ** 3
        if available_gb < self.min_free_ram_gb:
            raise MemoryError(
                f"Insufficient RAM to load Whisper: "
                f"{available_gb:.1f} GB available, "
                f"{self.min_free_ram_gb} GB required."
            )
        from faster_whisper import WhisperModel as _WhisperModel  # noqa: PLC0415
        self._model = _WhisperModel(
            self.model_size, device="cuda", compute_type="int8"
        )

    def _unload(self) -> None:
        del self._model
        self._model = None
        try:
            import torch  # noqa: PLC0415
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def transcribe(
        self,
        audio_path: str,
        *,
        vad_filter: bool,
        word_timestamps: bool,
    ):
        """Transcribe audio via the selected backend.

        Returns the backend's raw output — the caller normalises into Segment objects.
        Callers MUST iterate generators lazily — do not wrap in list().
        """
        if self._model is None:
            raise RuntimeError(
                "WhisperModel.transcribe() called outside a lifecycle() context."
            )
        return self._model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
        )
