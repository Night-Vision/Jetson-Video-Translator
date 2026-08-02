"""VideoTranslator — AI-powered video dubbing pipeline for Jetson Orin.

Entry point: orchestrates pipeline stages only, zero business logic.
"""
import argparse
import sys
import logging

from video_translator.config import Config
from video_translator.pipeline.downloader import Downloader
from video_translator.pipeline.extractor import AudioExtractor
from video_translator.pipeline.transcriber import Transcriber
from video_translator.pipeline.translator import Translator
from video_translator.pipeline.dubber import AudioDubber
from video_translator.pipeline.writer import SubtitleWriter
from video_translator.utils.memory_monitor import log_memory, log_vram

logger = logging.getLogger("video_translator")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoTranslator — AI-powered video dubbing")
    parser.add_argument("url", help="URL of the video to translate")
    parser.add_argument("--lang", default="Russian", help="Target language (default: Russian)")
    parser.add_argument(
        "--format",
        choices=["dubbed", "srt", "vtt", "json"],
        default="dubbed",
        help="Output format (default: dubbed)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug memory logging")
    parser.add_argument(
        "--bg-volume", type=float, default=None,
        help="Original audio volume during speech (0.0=mute, 1.0=full, default=0.15)",
    )
    return parser.parse_args()


def _cleanup(config: Config) -> None:
    """Remove intermediate files from the RAM disk on exit."""
    import os  # noqa: PLC0415
    import glob  # noqa: PLC0415
    logger.info("Cleaning up intermediate /dev/shm/ files...")
    
    # Clean up specific temp files and glob segment files
    paths_to_remove = [
        config.tmp_video,
        config.tmp_audio,
        config.tmp_segments,
        "/dev/shm/dub_track_main.wav",
    ]
    paths_to_remove.extend(glob.glob("/dev/shm/seg_*.wav"))
    
    for path in paths_to_remove:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            logger.debug("Failed to remove clean-up file %s: %s", path, e)


def main() -> None:
    args = _parse_args()
    
    # Configure root logger to output to stdout with configured level
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    config = Config(
        target_lang=args.lang,
        output_format=args.format,
        debug=args.debug,
    )
    if args.bg_volume is not None:
        config.bg_volume = args.bg_volume

    downloader  = Downloader(config)
    extractor   = AudioExtractor(config)
    transcriber = Transcriber(config)
    translator  = Translator(config)

    # Unload any stale models before starting to maximise RAM for Whisper
    translator.reset()

    try:
        if not downloader.download(args.url):
            logger.error("Download failed, aborting.")
            sys.exit(1)

        extractor.extract()
        log_memory("post-extract", config)
        log_vram("post-extract", config)

        segments   = transcriber.transcribe()
        translated = translator.translate_segments(segments)
        log_memory("post-translate", config)
        log_vram("post-translate", config)

        if config.output_format == "dubbed":
            AudioDubber(config).dub(translated)
        else:
            SubtitleWriter(config).write(translated)

        log_memory("post-output", config)
        log_vram("post-output", config)

        output = (
            config.output_video
            if config.output_format == "dubbed"
            else f"{config.output_subtitles}.{config.output_format}"
        )
        logger.info("Done! Output: %s", output)

    finally:
        translator.reset()
        _cleanup(config)


if __name__ == "__main__":
    main()