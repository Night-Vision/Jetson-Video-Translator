from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import TYPE_CHECKING

from ..utils.audio_utils import estimate_gender, get_wav_duration

if TYPE_CHECKING:
    from ..config import Config
    from ..models.segment import Segment

logger = logging.getLogger("video_translator.dubber")

# TTS model filenames per target language and speaker gender.
_TTS_MODELS: dict[str, dict[str, str]] = {
    "English": {"female": "en_US-lessac-medium.onnx",   "male": "en_US-lessac-medium.onnx"},
    "Russian": {"female": "ru_RU-irina-medium.onnx",    "male": "ru_RU-dmitri-medium.onnx"},
    "Spanish": {"female": "es_ES-sharvard-medium.onnx", "male": "es_ES-sharvard-medium.onnx"},
    "French":  {"female": "fr_FR-upmc-medium.onnx",     "male": "fr_FR-upmc-medium.onnx"},
    "German":  {"female": "de_DE-thorsten-medium.onnx", "male": "de_DE-thorsten-medium.onnx"},
}
_DEFAULT_MODELS = {"female": "en_US-lessac-medium.onnx", "male": "en_US-lessac-medium.onnx"}


class AudioDubber:
    """Pipeline stage: generate Piper TTS audio and mux it with the source video.

    Processing order:
      1. Materialise + overlap-resolve segments
      2. Estimate speaker gender per segment via pitch autocorrelation
      3. Batch-generate TTS WAVs via Piper (two passes: male / female)
      4. Mix all segment WAVs into a single dub track with ffmpeg
      5. Mux dub track with the original video, muting source speech windows
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    def _get_model_sample_rate(self) -> int:
        """Resolve sample rate from target model config JSON; defaults to 22050 Hz."""
        lang_models = _TTS_MODELS.get(self.config.target_lang, _DEFAULT_MODELS)
        female_model = lang_models.get("female", _DEFAULT_MODELS["female"])
        model_path_candidate = os.path.join(self.config.tts_models_dir, female_model)
        model_path = model_path_candidate if os.path.exists(model_path_candidate) else female_model

        json_path = f"{model_path}.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
                sr = cfg.get("audio", {}).get("sample_rate")
                if sr:
                    logger.info("Detected target model sample rate from config: %d Hz", sr)
                    return int(sr)
            except Exception as e:
                logger.warning("Failed to parse model JSON %s: %s", json_path, e)
        return 22050

    def _get_video_audio_rate(self) -> int | None:
        """Probe the original video's audio stream sample rate."""
        try:
            probe = subprocess.check_output([
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=sample_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                self.config.tmp_video,
            ])
            rate = int(probe)
            logger.info("Detected video audio sample rate: %d Hz", rate)
            return rate
        except Exception as exc:
            logger.warning("Could not read video audio rate: %s", exc)
            return None

    def dub(self, segment_iter) -> None:
        segments: list[Segment] = list(segment_iter)
        if not segments:
            logger.info("No segments to dub.")
            return

        logger.info("Generating dubbed audio & synchronizing...")
        video_duration = self._get_video_duration()
        self._resolve_overlaps(segments, video_duration)
        self._assign_genders(segments)
        self._run_tts_for_all_genders(segments)

        sample_rate = self._get_video_audio_rate() or self._get_model_sample_rate()

        dub_track = self._mix_segments_to_track(segments, sample_rate)
        try:
            self._mux_final_video(segments, dub_track)
        finally:
            self._cleanup_dub_track(dub_track)

    # ── Step helpers ─────────────────────────────────────────────────────────

    def _get_video_duration(self) -> float:
        try:
            probe = subprocess.check_output([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                self.config.tmp_video,
            ])
            return float(probe)
        except Exception as exc:
            logger.warning("Could not read video duration: %s", exc)
            return 0.0

    def _resolve_overlaps(self, segments: list[Segment], video_duration: float) -> None:
        """Ensure no two segments overlap; clamp last segment to video duration."""
        logger.info("Resolving overlapping segments...")
        min_dur = 0.5

        # First pass: shrink previous segment when overlap occurs
        for i in range(1, len(segments)):
            if segments[i].start_time < segments[i - 1].end_time:
                desired_end = max(
                    segments[i - 1].start_time + min_dur,
                    segments[i].start_time,
                )
                # Cap at the next segment's start: if the gap is < min_dur,
                # non-overlap wins over the 0.5 s floor (else two dubs overlap).
                segments[i - 1].end_time = min(desired_end, segments[i].start_time)

        # Second pass: ensure every segment has at least min_dur without re-overlapping
        for i in range(len(segments)):
            if segments[i].end_time < segments[i].start_time + min_dur:
                desired_end = segments[i].start_time + min_dur
                if i < len(segments) - 1:
                    desired_end = min(desired_end, segments[i + 1].start_time)
                if video_duration > 0.0:
                    desired_end = min(desired_end, video_duration)
                segments[i].end_time = desired_end

        # Final safety clamp on last segment
        if video_duration > 0.0 and segments:
            last = segments[-1]
            last.end_time = min(last.end_time, video_duration)
            last.end_time = max(last.end_time, last.start_time + 0.1)

    def _assign_genders(self, segments: list[Segment]) -> None:
        logger.info("Running speaker gender detection...")
        for seg in segments:
            seg.gender = estimate_gender(
                self.config.tmp_audio, seg.start_time, seg.end_time
            )

    def _run_tts_for_all_genders(self, segments: list[Segment]) -> None:
        male_group   = [(i, s) for i, s in enumerate(segments) if s.gender == "male"]
        female_group = [(i, s) for i, s in enumerate(segments) if s.gender != "male"]
        self._run_piper_group(male_group, "male")
        self._run_piper_group(female_group, "female")

    def _run_piper_group(
        self, group: list[tuple[int, Segment]], gender: str
    ) -> None:
        if not group:
            return

        lang_models = _TTS_MODELS.get(self.config.target_lang, _DEFAULT_MODELS)
        model_name = lang_models.get(gender, _DEFAULT_MODELS[gender])
        model_path_candidate = os.path.join(self.config.tts_models_dir, model_name)
        model_path = model_path_candidate if os.path.exists(model_path_candidate) else model_name

        # Piper accepts newline-delimited JSON objects: {"text": ..., "output_file": ...}
        input_lines = "\n".join(
            json.dumps({"text": seg.translated_text, "output_file": f"/dev/shm/seg_{i}.wav"})
            for i, seg in group
        ) + "\n"

        logger.info(
            "Spawning Piper TTS (%s) for %d segments using model: %s",
            gender, len(group), model_path
        )
        try:
            subprocess.run(
                [self.config.piper_bin, "--model", model_path, "--json-input"],
                input=input_lines.encode("utf-8"),
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Piper TTS failed for {gender} voice: {exc}")

    @staticmethod
    def _trim_tts_silence(wav_path: str, sample_rate: int) -> None:
        """Strip leading/trailing silence from a Piper WAV in-place."""
        trimmed = wav_path.replace(".wav", "_trimmed.wav")
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path,
                "-af", "silenceremove=start_periods=1:start_duration=0.03:"
                       "start_threshold=-45dB:stop_periods=1:stop_duration=0.03:"
                       "stop_threshold=-45dB",
                "-ar", str(sample_rate), "-ac", "2",
                trimmed,
            ], check=True, capture_output=True)
            os.replace(trimmed, wav_path)
        except Exception as exc:
            logger.warning("Silence trim failed for %s: %s", wav_path, exc)
            try:
                os.remove(trimmed)
            except OSError:
                pass

    def _mix_segments_to_track(self, segments: list[Segment], sample_rate: int) -> str:
        """Build one ffmpeg filter_complex that delays + tempo-adjusts all segment WAVs."""
        logger.info("Compiling final audio track...")

        # ── Silent base track anchors the output to exact video duration ──
        video_duration = self._get_video_duration()
        silent_base = "/dev/shm/silent_base.wav"
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=r={sample_rate}:cl=stereo",
            "-t", str(video_duration),
            "-acodec", "pcm_s16le", "-ar", str(sample_rate),
            silent_base,
        ], check=True, capture_output=True)

        dub_inputs: list[str] = ["-i", silent_base]   # [0:a] is the silent anchor
        filter_parts: list[str] = []
        kept: list[Segment] = []
        kept_idx: list[int] = []
        n_total = len(segments)

        for i, seg in enumerate(segments):
            out_file = f"/dev/shm/seg_{i}.wav"
            target_duration = max(0.1, seg.end_time - seg.start_time)

            # Strip leading/trailing TTS silence before measuring duration
            self._trim_tts_silence(out_file, sample_rate)

            try:
                actual_duration = get_wav_duration(out_file)
            except Exception as exc:
                logger.warning("Could not read WAV duration for %s: %s", out_file, exc)
                actual_duration = target_duration

            # Piper emits a ~0.2 s all-zero WAV for text it can't vocalize (empty,
            # whitespace, "♪", "...").  After silence-trim it measures ~0 s: mixing
            # that silence while still ducking the original would blank the window.
            # Drop the segment — no dub, and no ducking (original audio stays).
            if actual_duration < 0.05:
                logger.warning(
                    "Dropping seg %d: TTS is silent (%.3f s) — keeping original audio",
                    i, actual_duration,
                )
                continue

            input_idx = len(dub_inputs) // 2   # silent_base is input 0
            dub_inputs.extend(["-i", out_file])
            kept.append(seg)
            kept_idx.append(i)

            # Normalize sample rate BEFORE any time-sensitive filters
            rate_filter = f"aresample={sample_rate}:async=1:first_pts=0,"

            tempo_filter = self._build_tempo_filter(actual_duration, target_duration)
            tempo_str = f"{tempo_filter}," if tempo_filter else ""

            # Hard-limit segment length so it can never spill into the next slot
            exact_dur = f"atrim=duration={target_duration:.4f}"

            # Round to nearest ms instead of truncating toward zero
            delay_ms = round(seg.start_time * 1000)

            filter_parts.append(
                f"[{input_idx}:a]{rate_filter}{tempo_str}{exact_dur},"
                f"adelay=delays={delay_ms}|{delay_ms}:all=1[a{i}];"
            )

        # Drop silent-TTS segments from the mux ducking too (they produce no dub),
        # so the original audio stays in those windows.
        segments[:] = kept
        n = len(kept_idx)
        amix_inputs = "[0:a]" + "".join(f"[a{i}]" for i in kept_idx)
        # duration=first forces output length == silent_base length == video_duration
        filter_complex = (
            "".join(filter_parts)
            + f"{amix_inputs}amix=inputs={n+1}:dropout_transition=0:"
            f"normalize=0:duration=first[out]"
        )

        dub_track = "/dev/shm/dub_track_main.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y"] + dub_inputs + [
                    "-filter_complex", filter_complex,
                    "-map", "[out]",
                    "-ar", str(sample_rate),
                    "-ac", "2",
                    "-threads", "2",
                    dub_track,
                ],
                check=True,
            )
        finally:
            for i in range(n_total):
                for f in (f"/dev/shm/seg_{i}.wav", f"/dev/shm/seg_{i}_trimmed.wav"):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            try:
                os.remove(silent_base)
            except OSError:
                pass

        # Diagnostic log check
        for i, seg in enumerate(segments[:5]):
            logger.info(
                "Sync check seg %d: start=%.3f  target_dur=%.3f  file=/dev/shm/seg_%d.wav",
                i, seg.start_time, seg.end_time - seg.start_time, i
            )

        return dub_track

    def _mux_final_video(self, segments: list[Segment], dub_track: str) -> None:
        """Mute source speech windows and mix in the dub track, then write output."""
        logger.info("Building dynamic muting filter...")
        padding = 0.1  # 100 ms boundary padding
        between_exprs = " + ".join(
            f"between(t,{max(0.0, s.start_time - padding):.3f},{s.end_time + padding:.3f})"
            for s in segments
        )
        volume_filter = (
            f"[0:a]volume='if({between_exprs},{self.config.bg_volume:.2f},1.0)':eval=frame[bg]"
            if between_exprs else "[0:a]volume=1.0[bg]"
        )

        logger.info("Final muxing...")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", self.config.tmp_video,
                "-i", dub_track,
                "-filter_complex",
                (
                    f"{volume_filter};"
                    "[1:a]volume=1.0[fg];"
                    "[bg][fg]amix=inputs=2:dropout_transition=0:normalize=0[out]"
                ),
                "-map", "0:v", "-map", "[out]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-threads", "4",
                self.config.output_video,
            ],
            check=True,
        )

    @staticmethod
    def _build_tempo_filter(actual: float, target: float) -> str:
        """Return a chained atempo filter string clamped to ffmpeg's 0.5–2.0 range.

        Only speeds up (ratio > 1.0) when TTS is longer than the time slot, so a
        long TTS never spills into the next window.  Never slows down: Whisper
        segment windows include natural trailing pause, so stretching the trimmed
        TTS to fill them distorts the speech rate and desyncs the dub from the
        video.  Short TTS simply ends early at its natural pace.
        """
        ratio = actual / target
        if ratio <= 1.0:
            return ""
        filters: list[str] = []
        # Speed up — chain atempo=2.0 for ratios above 2.0
        while ratio > 2.0:
            filters.append("atempo=2.0")
            ratio /= 2.0
        if abs(ratio - 1.0) > 1e-4:
            filters.append(f"atempo={ratio:.3f}")
        return ",".join(filters)

    @staticmethod
    def _cleanup_dub_track(dub_track: str) -> None:
        try:
            os.remove(dub_track)
        except OSError:
            pass
