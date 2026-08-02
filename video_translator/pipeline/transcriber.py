"""Whisper transcription pipeline stage.

Design note — subprocess isolation
-----------------------------------
Whisper (via CTranslate2) allocates a CUDA context that cannot be cleanly
released inside the same process.  To guarantee that 100% of VRAM is returned
to the OS before the vLLM container starts, transcription is run inside a
fresh child process.  When the subprocess exits, the OS reclaims its CUDA
context automatically.

This module doubles as both the pipeline stage class (Transcriber) and the
standalone subprocess worker (__main__ block).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger("video_translator.transcriber")

# Project root is three levels up: pipeline/ → video_translator/ → project root
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


class Transcriber:
    """Pipeline stage: transcribe tmp_audio → yield Segment objects.

    Whisper runs in a CUDA (Jetson) isolated subprocess to guarantee full CUDA context teardown.
    The subprocess writes segments.json; this class reads it and yields Segments.
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def transcribe(self) -> Generator:
        """Run the whisper subprocess, then yield Segments from the JSON output."""
        logger.info("Transcribing (INT8 in isolated CUDA (Jetson) subprocess)...")
        self._run_worker_subprocess()
        yield from self._load_segments()

    def _run_worker_subprocess(self) -> None:
        try:
            subprocess.run(
                [
                    sys.executable, "-m", "video_translator.pipeline.transcriber",
                    "--audio", self.config.tmp_audio,
                    "--segments", self.config.tmp_segments,
                    "--model", self.config.whisper_model,
                    "--min-ram", str(self.config.min_free_ram_gb),
                    "--backend", self.config.whisper_backend,
                    "--engine-dir", self.config.whisper_trt_engine_dir,
                    "--vad", str(self.config.vad_filter),
                    "--word-timestamps", str(self.config.word_timestamps),
                ],
                check=True,
                timeout=1800,  # 30 minutes timeout
                cwd=_PROJECT_ROOT,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Transcription subprocess failed with code %d", e.returncode)
            raise RuntimeError(f"Transcription subprocess failed: {e}")
        except subprocess.TimeoutExpired as e:
            logger.error("Transcription subprocess timed out: %s", e)
            raise RuntimeError(f"Transcription subprocess timed out: {e}")

    def _load_segments(self) -> Generator:
        # Local import — only needed in the parent process after subprocess exits
        from ..models.segment import Segment  # noqa: PLC0415

        with open(self.config.tmp_segments, "r") as fh:
            records = json.load(fh)

        for rec in records:
            yield Segment(
                start_time=rec["start_time"],
                end_time=rec["end_time"],
                text=rec["text"],
                language=rec["language"],
                confidence=rec["confidence"],
            )


# ── Subprocess worker entry point ────────────────────────────────────────────
# Run as: python -m video_translator.pipeline.transcriber --audio ... --segments ...
if __name__ == "__main__":
    import argparse

    from video_translator.models.whisper_model import WhisperModel

    parser = argparse.ArgumentParser(description="Whisper transcription worker")
    parser.add_argument("--audio",           required=True)
    parser.add_argument("--segments",        required=True)
    parser.add_argument("--model",           required=True)
    parser.add_argument("--min-ram",         type=float, required=True)
    parser.add_argument("--backend",         default="faster_whisper")
    parser.add_argument("--engine-dir",      default="")
    parser.add_argument("--vad",             type=lambda x: x.lower() == "true", required=True)
    parser.add_argument("--word-timestamps", type=lambda x: x.lower() == "true", required=True)
    args = parser.parse_args()

    whisper = WhisperModel(args.model, args.min_ram, backend=args.backend, engine_dir=args.engine_dir)
    with whisper.lifecycle():
        segments_iter, info = whisper.transcribe(
            args.audio,
            vad_filter=args.vad,
            word_timestamps=args.word_timestamps,
        )
        language = info.language
        records = [
            {
                "start_time": seg.start,
                "end_time":   seg.end,
                "text":       seg.text,
                "language":   language,
                "confidence": seg.avg_logprob,
            }
            for seg in segments_iter  # iterate lazily; do not buffer
        ]

    with open(args.segments, "w") as fh:
        json.dump(records, fh)
