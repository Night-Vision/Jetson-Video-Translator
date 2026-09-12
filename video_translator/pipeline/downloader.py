from __future__ import annotations

import logging
import shutil
import subprocess
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger("video_translator.downloader")


class Downloader:
    """Downloads a video URL to the configured tmp_video path using yt-dlp."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def download(self, url: str) -> bool:
        """Return True on success, False on failure (never raises)."""
        logger.info("Downloading media...")

        # YouTube presents JS challenges that require the yt-dlp-ejs scripts
        # plus an enabled JS runtime; without them yt-dlp falls back to the
        # increasingly fragile jless clients. Node is not enabled by default
        # (only Deno is), so it must be requested explicitly.
        if shutil.which("node") is None:
            logger.warning(
                "node not found on PATH — yt-dlp-ejs JS challenge solving "
                "unavailable; YouTube extraction will fall back to jless clients"
            )

        cmd = [
            "yt-dlp",
            "--extractor-args", "youtube:player_client=default",
            "--js-runtimes", "node",
            # H.264 first: Orin Nano's NVDEC has no AV1 path, so an AV1 stream
            # decodes all-core on the CPU in every later ffmpeg pass.  YouTube
            # ships no avc1 above 1080p, so the codec filter also caps the
            # download size -- do NOT add a [height<=N] clause, it rejects
            # vertical formats (a 1080p Short is 1080x1920, height 1920).
            "-f", (
                "bv*[vcodec^=avc1]+ba[ext=m4a]/"  # H.264 + AAC: NVDEC, clean mp4 mux
                "bv*[vcodec^=avc1]+ba/"           # H.264 + whatever audio exists
                "bv*+ba/b"                        # last resort: any codec
            ),
            "--merge-output-format", "mp4",
            "-o", self.config.tmp_video,
        ]
        cmd.append(url)

        max_attempts = 3
        delay = 2.0
        for attempt in range(1, max_attempts + 1):
            try:
                subprocess.run(cmd, check=True)
                logger.info("Download completed successfully.")
                return True
            except subprocess.CalledProcessError as e:
                logger.warning(
                    "Download attempt %d failed: %s",
                    attempt,
                    e
                )
                if attempt == max_attempts:
                    logger.error("All download attempts failed.")
                    return False
                
                logger.info("Retrying in %gs...", delay)
                time.sleep(delay)
                delay *= 2.0
        return False
