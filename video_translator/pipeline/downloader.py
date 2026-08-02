from __future__ import annotations

import logging
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
        cmd = [
            "yt-dlp",
            "--extractor-args", "youtube:player_client=default",
            "-f", "bestvideo+bestaudio/best",
            "--merge-output-format", "mp4",
            "-o", self.config.tmp_video,
        ]
        # Only inject cookie auth when a Firefox profile path is configured
        if self.config.browser_cookies_path:
            cmd.extend([
                "--cookies-from-browser",
                f"firefox:{self.config.browser_cookies_path}",
            ])
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
