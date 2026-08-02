from __future__ import annotations

import logging
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger("video_translator.extractor")


class AudioExtractor:
    """Extracts a mono 22050 Hz PCM WAV from the downloaded video file via ffmpeg."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def extract(self) -> None:
        """Write tmp_audio from tmp_video; raises subprocess.CalledProcessError on failure."""
        logger.info("Extracting audio track...")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", self.config.tmp_video,
                "-ar", "22050",   # 22050 Hz is Piper's native sample rate
                "-ac", "1",       # mono
                "-c:a", "pcm_s16le",
                self.config.tmp_audio,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
