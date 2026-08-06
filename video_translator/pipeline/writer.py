"""Subtitle output stage — writes translated segments to SRT, VTT, or JSON files."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..models.segment import Segment

logger = logging.getLogger("video_translator.writer")


class SubtitleWriter:
    """Pipeline stage: serialise translated Segments to a subtitle file.

    Supported formats (controlled by config.output_format):
      - "srt"  → SubRip (.srt)
      - "vtt"  → WebVTT (.vtt)
      - "json" → JSON array with start/end/source/translated fields
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def write(self, segment_iter) -> bool:
        """Serialize translated Segments to a subtitle file; False when empty."""
        segments: list[Segment] = list(segment_iter)
        if not segments:
            logger.info("No segments to write.")
            return False

        fmt = self.config.output_format
        output_path = f"{self.config.output_subtitles}.{fmt}"

        writers = {
            "srt":  self._write_srt,
            "vtt":  self._write_vtt,
            "json": self._write_json,
        }
        writer_fn = writers.get(fmt)
        if writer_fn is None:
            raise ValueError(f"Unsupported subtitle format: {fmt!r}")

        writer_fn(segments, output_path)
        logger.info("Subtitles written to %s", output_path)
        return True

    # ── Format implementations ───────────────────────────────────────────────

    def _write_srt(self, segments: list[Segment], path: str) -> None:
        lines: list[str] = []
        for idx, seg in enumerate(segments, start=1):
            start = self._srt_timestamp(seg.start_time)
            end   = self._srt_timestamp(seg.end_time)
            lines.append(f"{idx}\n{start} --> {end}\n{seg.translated_text}\n")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def _write_vtt(self, segments: list[Segment], path: str) -> None:
        lines: list[str] = ["WEBVTT", ""]
        for seg in segments:
            start = self._vtt_timestamp(seg.start_time)
            end   = self._vtt_timestamp(seg.end_time)
            lines.append(f"{start} --> {end}\n{seg.translated_text}\n")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))

    def _write_json(self, segments: list[Segment], path: str) -> None:
        records = [
            {
                "start":      seg.start_time,
                "end":        seg.end_time,
                "source":     seg.text,
                "translated": seg.translated_text,
                "language":   seg.language,
                "confidence": seg.confidence,
            }
            for seg in segments
        ]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=2)

    # ── Timestamp formatters ─────────────────────────────────────────────────

    @staticmethod
    def _srt_timestamp(seconds: float) -> str:
        """Format seconds as HH:MM:SS,mmm (SubRip)."""
        h, rem = divmod(int(seconds), 3600)
        m, s   = divmod(rem, 60)
        ms     = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _vtt_timestamp(seconds: float) -> str:
        """Format seconds as HH:MM:SS.mmm (WebVTT)."""
        h, rem = divmod(int(seconds), 3600)
        m, s   = divmod(rem, 60)
        ms     = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
