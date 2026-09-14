from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import TYPE_CHECKING

from ..utils.audio_utils import estimate_gender, speech_bounds
from ..utils.memory_monitor import log_oc

if TYPE_CHECKING:
    from ..config import Config
    from ..models.segment import Segment

logger = logging.getLogger("video_translator.dubber")

# TTS model filenames per target language and speaker gender.
#
# Only Russian ships distinct male/female voices in this repo
# (ru_RU-irina-medium.onnx / ru_RU-dmitri-medium.onnx).  English, Spanish,
# French, and German reuse a single voice for both genders — the female/male
# keys are aliases — until additional Piper voices are added to models/.
# A single filename means that language uses one voice for both genders.
_TTS_MODELS: dict[str, str | dict[str, str]] = {
    "English": "en_US-lessac-medium.onnx",
    "Russian": {"female": "ru_RU-irina-medium.onnx", "male": "ru_RU-dmitri-medium.onnx"},
    "Spanish": "es_ES-sharvard-medium.onnx",
    "French":  "fr_FR-upmc-medium.onnx",
    "German":  "de_DE-thorsten-medium.onnx",
}
_DEFAULT_MODEL = "en_US-lessac-medium.onnx"


def _voice_for(lang: str, gender: str) -> str:
    """Piper voice filename for a language/gender, falling back to English."""
    entry = _TTS_MODELS.get(lang, _DEFAULT_MODEL)
    if isinstance(entry, str):
        return entry
    return entry.get(gender, _DEFAULT_MODEL)


def _duck_end(seg: Segment, padding: float) -> float:
    """End of the ducking window for a segment.

    Covers the effective dub extent (dub_end + padding) if set,
    otherwise falls back to the Whisper segment end (end_time + padding).
    """
    if seg.dub_end is not None:
        return seg.dub_end + padding
    return seg.end_time + padding


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
        self.audio_offset = 0.0

    def _get_model_sample_rate(self) -> int:
        """Resolve sample rate from target model config JSON; defaults to 22050 Hz."""
        female_model = _voice_for(self.config.target_lang, "female")
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

    def _probe(self) -> dict:
        """Everything the dubber needs from the container, in one ffprobe call.

        `-show_entries` takes several sections separated by `:`, so duration,
        both stream start PTS values and the audio sample rate come back
        together -- four subprocess spawns collapsed into one.
        """
        try:
            raw = subprocess.check_output([
                "ffprobe", "-v", "error",
                "-show_entries",
                "format=duration:stream=codec_type,start_time,sample_rate",
                "-of", "json",
                self.config.tmp_video,
            ])
            return self._parse_probe(json.loads(raw))
        except Exception as exc:
            logger.warning("ffprobe failed (%s) — using fallbacks", exc)
            return self._parse_probe({})

    @staticmethod
    def _parse_probe(data: dict) -> dict:
        """Pull the four fields out of ffprobe JSON; fall back on anything absent.

        Streams are matched on codec_type rather than position -- container
        stream order is not guaranteed.
        """
        def _stream(kind: str) -> dict:
            for st in data.get("streams", []):
                if st.get("codec_type") == kind:
                    return st
            return {}

        def _float(value, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        audio = _stream("audio")
        try:
            sample_rate = int(audio["sample_rate"])
        except (KeyError, TypeError, ValueError):
            sample_rate = None

        return {
            "duration": _float(data.get("format", {}).get("duration"), 0.0),
            "sample_rate": sample_rate,
            "a_start": _float(audio.get("start_time"), 0.0),
            "v_start": _float(_stream("video").get("start_time"), 0.0),
        }

    def _resolve_audio_offset(self, probe: dict) -> float:
        """Seconds to shift the dub so it lands on the *video* timeline.

        Whisper timestamps are relative to the first sample of the extracted
        WAV, which ffmpeg writes at t=0 regardless of the audio stream's
        container start PTS.  The final mux uses `-c:v copy`, so the video
        keeps its own start PTS.  When the two differ (routine for yt-dlp
        bestvideo+bestaudio merges) every segment is off by that constant.
        """
        if self.config.audio_offset is not None:
            logger.info("Audio offset (user-supplied): %+.3f s", self.config.audio_offset)
            return self.config.audio_offset

        offset = probe["a_start"] - probe["v_start"]
        if abs(offset) > 5.0:
            logger.warning(
                "Probed audio offset %+.3f s is implausible — using 0.0; "
                "pass --audio-offset to set it by ear", offset,
            )
            return 0.0
        logger.info("Audio offset (probed a:0 - v:0): %+.3f s", offset)
        return offset

    def dub(self, segment_iter) -> bool:
        """Generate the dubbed video; return False when there is nothing to dub."""
        segments: list[Segment] = list(segment_iter)
        if not segments:
            logger.info("No segments to dub.")
            return False

        logger.info("Generating dubbed audio & synchronizing...")
        probe = self._probe()
        self.audio_offset = self._resolve_audio_offset(probe)
        video_duration = probe["duration"]
        self._resolve_overlaps(segments, video_duration)
        self._assign_genders(segments)
        log_oc("dub:post-gender", self.config)
        self._run_tts_for_all_genders(segments)
        log_oc("dub:post-tts", self.config)

        sample_rate = probe["sample_rate"] or self._get_model_sample_rate()

        dub_track = self._mix_segments_to_track(segments, sample_rate, video_duration)
        log_oc("dub:post-mix", self.config)
        if dub_track is None:
            return False
        try:
            self._mux_final_video(segments, dub_track)
            log_oc("dub:post-mux", self.config)
        finally:
            try:
                os.remove(dub_track)
            except OSError:
                pass
        return True

    # ── Step helpers ─────────────────────────────────────────────────────────

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

        model_name = _voice_for(self.config.target_lang, gender)
        model_path_candidate = os.path.join(self.config.tts_models_dir, model_name)
        if not os.path.exists(model_path_candidate):
            logger.warning(
                "Piper voice %s not found in %s — passing bare filename to Piper; "
                "install the voice under models/ for %s dubbing.",
                model_name, self.config.tts_models_dir, self.config.target_lang,
            )
        model_path = model_path_candidate if os.path.exists(model_path_candidate) else model_name

        payloads = []
        for i, seg in group:
            target_dur = max(0.1, seg.end_time - seg.start_time)
            est_dur = len(seg.translated_text) * 0.065
            speedup = max(1.0, min(est_dur / target_dur, self.config.max_tempo))
            length_scale = round(1.0 / speedup, 3)

            payloads.append({
                "text": seg.translated_text,
                "output_file": f"/dev/shm/seg_{i}.wav",
                "length_scale": length_scale,
            })

        input_lines = "\n".join(json.dumps(p) for p in payloads) + "\n"

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

    def _mix_segments_to_track(
        self, segments: list[Segment], sample_rate: int, video_duration: float
    ) -> str | None:
        """Build one ffmpeg filter_complex that delays + tempo-adjusts all segment WAVs.

        Returns the dub track path, or None when every segment was dropped as
        silent TTS (nothing to dub).
        """
        logger.info("Compiling final audio track...")

        # A zero duration means the earlier ffprobe failed.  Do not build a
        # `-t 0` silent base and emit an empty/`-t 0` track: fail loudly instead.
        if video_duration <= 0.0:
            raise RuntimeError("Could not determine video duration — aborting dubbing")

        # ── Silent base anchors the output to exact video duration ──
        # lavfi is an input device, so anullsrc feeds the graph directly as
        # [0:a]; `-t` before `-i` bounds it.  No temp file, no second ffmpeg.
        dub_inputs: list[str] = [
            "-f", "lavfi", "-t", f"{video_duration:.4f}",
            "-i", f"anullsrc=r={sample_rate}:cl=stereo",
        ]
        filter_parts: list[str] = []
        kept: list[Segment] = []
        kept_idx: list[int] = []
        n_total = len(segments)

        for i, seg in enumerate(segments):
            out_file = f"/dev/shm/seg_{i}.wav"
            target_duration = max(0.1, seg.end_time - seg.start_time)

            # Locate the speech span instead of destructively trimming the WAV:
            # leading/trailing silence is skipped by atrim inside the main graph,
            # and internal pauses are left intact.
            try:
                bounds = speech_bounds(out_file)
            except Exception as exc:
                logger.warning("Could not read WAV %s: %s", out_file, exc)
                bounds = None

            # Piper emits a ~0.2 s all-zero WAV for text it can't vocalize (empty,
            # whitespace, "♪", "...").  Its speech span is empty: mixing that
            # silence while still ducking the original would blank the window.
            # Drop the segment — no dub, and no ducking (original audio stays).
            if bounds is None or bounds[1] - bounds[0] < 0.05:
                logger.warning(
                    "Dropping seg %d: TTS is silent — keeping original audio", i,
                )
                continue

            speech_start, speech_end = bounds
            actual_duration = speech_end - speech_start
            # atrim keeps the source timestamps, so asetpts must rebase to zero
            # before adelay — otherwise the leading silence is added back as delay.
            trim_str = (
                f"atrim=start={speech_start:.4f}:end={speech_end:.4f},asetpts=N/SR/TB,"
            )

            # -i count, not token count: the lavfi anchor spends 6 tokens.
            input_idx = dub_inputs.count("-i")
            dub_inputs.extend(["-i", out_file])
            kept.append(seg)
            kept_idx.append(i)

            # Speed up (capped) only when TTS is longer than the slot; if the
            # capped tempo still cannot fit it, let the excess spill into the
            # trailing pause (bounded by the next segment start) before the
            # hard atrim cut — never speed-garble beyond max_tempo.
            tempo_filter, fitted = self._build_tempo_filter(
                actual_duration, target_duration, self.config.max_tempo
            )
            tempo_str = f"{tempo_filter}," if tempo_filter else ""
            trim_duration = target_duration
            if fitted > target_duration:
                slack = 0.0
                if i + 1 < len(segments):
                    slack = max(0.0, segments[i + 1].start_time - seg.end_time)
                trim_duration = min(fitted, target_duration + slack)

            # Hard-limit segment length so it can never spill into the next slot
            exact_dur = f"atrim=duration={trim_duration:.4f}"

            # Remember the effective dub extent so the mux can duck the
            # original audio across the whole spill tail, not just the window.
            seg.dub_end = seg.start_time + trim_duration

            # Round to nearest ms instead of truncating toward zero
            delay_ms = max(0, round((seg.start_time + self.audio_offset) * 1000))

            filter_parts.append(
                f"[{input_idx}:a]{trim_str}{tempo_str}{exact_dur},"
                f"adelay=delays={delay_ms}|{delay_ms}:all=1[a{i}];"
            )

        # Drop silent-TTS segments from the mux ducking too (they produce no dub),
        # so the original audio stays in those windows.
        segments[:] = kept
        n = len(kept_idx)
        dropped = n_total - n
        if dropped and n > 0:
            logger.warning(
                "Dropped %d of %d segments (silent TTS) — original audio kept in those windows",
                dropped, n_total,
            )

        dub_track = "/dev/shm/dub_track_main.wav"
        try:
            if not kept:
                logger.warning(
                    "All %d segments produced silent TTS — no dub track; "
                    "keeping the original audio, no output video will be written",
                    n_total,
                )
                return None

            amix_inputs = "[0:a]" + "".join(f"[a{i}]" for i in kept_idx)
            # duration=first forces output length == silent_base length == video_duration
            filter_complex = (
                "".join(filter_parts)
                + f"{amix_inputs}amix=inputs={n+1}:dropout_transition=0:"
                f"normalize=0:duration=first[out]"
            )
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
                try:
                    os.remove(f"/dev/shm/seg_{i}.wav")
                except OSError:
                    pass

        return dub_track

    def _mux_final_video(self, segments: list[Segment], dub_track: str) -> None:
        """Mute source speech windows and mix in the dub track, or direct map if ducking disabled."""
        if not self.config.enable_ducking or self.config.bg_volume <= 0.0:
            logger.info("Fast path muxing (direct stream replacement, ducking disabled)...")
            t0 = time.perf_counter()
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", self.config.tmp_video,
                    "-i", dub_track,
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-threads", "4",
                    self.config.output_video,
                ],
                check=True,
            )
            elapsed = time.perf_counter() - t0
            logger.info("Mux stage completed in %.3f seconds (Fast Path)", elapsed)
            return

        logger.info("Dynamic ducking muxing (bg_volume=%.2f)...", self.config.bg_volume)
        t0 = time.perf_counter()
        padding = 0.1  # 100 ms boundary padding
        off = self.audio_offset
        between_exprs = " + ".join(
            f"between(t,{max(0.0, s.start_time + off - padding):.3f},"
            f"{max(0.0, _duck_end(s, padding) + off):.3f})"
            for s in segments
        )
        volume_filter = (
            f"[0:a]volume='if({between_exprs},{self.config.bg_volume:.2f},1.0)':eval=frame[bg]"
            if between_exprs else "[0:a]volume=1.0[bg]"
        )

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
        elapsed = time.perf_counter() - t0
        logger.info("Mux stage completed in %.3f seconds (Ducking Path)", elapsed)

    @staticmethod
    def _build_tempo_filter(
        actual: float, target: float, max_tempo: float
    ) -> tuple[str, float]:
        """Return a chained atempo filter clamped to `max_tempo`, and the
        duration the segment will actually occupy after it.

        Only speeds up (ratio > 1.0) when TTS is longer than the time slot, and
        never beyond `max_tempo` (default 1.35): beyond that the excess is cut
        or spills into the trailing pause instead of being speed-garbled.
        Never slows down: Whisper segment windows include natural trailing
        pause, so stretching the trimmed TTS to fill them distorts the speech
        rate and desyncs the dub from the video.  Short TTS simply ends early
        at its natural pace.

        Note: chaining atempo=2.0 stages is required for speedups above 2.0
        (some builds cap a single instance there); the `max_tempo` clamp means
        the chain is only exercised for unusually high user-configured caps.
        """
        ratio = actual / target
        if ratio <= 1.0 or max_tempo <= 1.0:
            return "", actual
        capped = min(ratio, max_tempo)
        filters: list[str] = []
        # Speed up — chain atempo=2.0 for ratios above 2.0
        r = capped
        while r > 2.0:
            filters.append("atempo=2.0")
            r /= 2.0
        if abs(r - 1.0) > 1e-4:
            filters.append(f"atempo={r:.3f}")
        return ",".join(filters), actual / capped



def _self_check() -> None:
    """_parse_probe must key off codec_type, not stream order, and survive gaps."""
    parse = AudioDubber._parse_probe

    # Audio listed first, video second — order must not matter.
    got = parse({
        "streams": [
            {"codec_type": "audio", "start_time": "1.250", "sample_rate": "48000"},
            {"codec_type": "video", "start_time": "0.500"},
        ],
        "format": {"duration": "64.32"},
    })
    assert got == {
        "duration": 64.32, "sample_rate": 48000, "a_start": 1.25, "v_start": 0.5
    }, got

    # A container with no audio stream and no duration must not raise.
    got = parse({"streams": [{"codec_type": "video"}]})
    assert got == {
        "duration": 0.0, "sample_rate": None, "a_start": 0.0, "v_start": 0.0
    }, got

    # Empty payload (the ffprobe-failed path) falls back across the board.
    assert parse({})["duration"] == 0.0

    # _voice_for: str entry = one voice for both genders, dict = per-gender.
    assert _voice_for("Russian", "male") == "ru_RU-dmitri-medium.onnx"
    assert _voice_for("Russian", "female") == "ru_RU-irina-medium.onnx"
    assert _voice_for("French", "male") == _voice_for("French", "female")
    assert _voice_for("Klingon", "male") == _DEFAULT_MODEL

    print("dubber self-check OK")


if __name__ == "__main__":
    _self_check()
