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
    parser.add_argument(
        "--max-tempo", type=float, default=None,
        help="Max dub speech-rate speedup when translation is longer than its window "
             "(1.0=never speed up, default=1.35)",
    )
    args = parser.parse_args()
    if args.max_tempo is not None and args.max_tempo < 1.0:
        parser.error("--max-tempo must be >= 1.0 (1.0 = never speed up)")
    return args


def _ramdisk_paths(config: Config) -> list[str]:
    """All intermediate paths the pipeline may leave on the RAM disk.

    Includes yt-dlp partial-stream artifacts (e.g. ``input_vid.f399.mp4``,
    ``input_vid.f251.webm``, ``*.part``, ``*.ytdl``) that survive an
    interrupted download while ``bestvideo+bestaudio`` streams are being
    merged.  Artifacts are derived from the tmp_video basename so the sweep
    works for any configured path.
    """
    import glob  # noqa: PLC0415
    import os  # noqa: PLC0415

    shm_dir = os.path.dirname(config.tmp_video) or "/dev/shm"
    prefix = os.path.basename(config.tmp_video).rsplit(".", 1)[0]

    paths = [
        config.tmp_video,
        config.tmp_audio,
        config.tmp_segments,
        "/dev/shm/dub_track_main.wav",
        "/dev/shm/silent_base.wav",
    ]
    # yt-dlp per-stream temp files + in-progress fragments for this video
    paths.extend(glob.glob(os.path.join(shm_dir, f"{prefix}.f*")))
    paths.extend(glob.glob(os.path.join(shm_dir, f"{prefix}.*.part")))
    paths.extend(glob.glob(os.path.join(shm_dir, f"{prefix}.*.ytdl")))
    # Per-segment Piper outputs (also removed by the dubber on success)
    paths.extend(glob.glob(os.path.join(shm_dir, "seg_*.wav")))
    paths.extend(glob.glob(os.path.join(shm_dir, "seg_*_trimmed.wav")))
    return paths


def _remove_paths(paths_to_remove: list[str]) -> None:
    import os  # noqa: PLC0415

    for path in paths_to_remove:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            logger.debug("Failed to remove clean-up file %s: %s", path, e)


def _cleanup(config: Config) -> None:
    """Remove intermediate files from the RAM disk on exit."""
    logger.info("Cleaning up intermediate /dev/shm/ files...")
    _remove_paths(_ramdisk_paths(config))


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
    if args.max_tempo is not None:
        config.max_tempo = args.max_tempo

    downloader  = Downloader(config)
    extractor   = AudioExtractor(config)
    transcriber = Transcriber(config)
    translator  = Translator(config)

    # Unload any stale models before starting to maximise RAM for Whisper
    translator.reset()

    # Sweep leftovers from an interrupted previous run (e.g. yt-dlp
    # partial-stream fragments) before starting fresh — the exit-time
    # _cleanup() cannot run if the process was SIGKILLed.
    _remove_paths(_ramdisk_paths(config))

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
            produced = AudioDubber(config).dub(translated)
        else:
            produced = SubtitleWriter(config).write(translated)

        log_memory("post-output", config)
        log_vram("post-output", config)

        if not produced:
            logger.warning("No speech or dubbable segments — no output file produced.")
            sys.exit(0)

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