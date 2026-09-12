from __future__ import annotations

import logging
import wave

import numpy as np

logger = logging.getLogger("video_translator.audio_utils")


def speech_bounds(file_path: str, threshold_db: float = -45.0) -> tuple[float, float] | None:
    """Return (start, end) seconds of the non-silent span of a WAV, or None.

    Peak-based: the first and last sample whose magnitude exceeds
    `threshold_db` relative to full scale.  Only the leading and trailing
    silence is located -- pauses *inside* the utterance are deliberately left
    alone, since removing them reflows the speech and desyncs the dub.

    Returns None when the whole file is below the threshold.
    """
    with wave.open(file_path, "rb") as wf:
        framerate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        data = wf.readframes(wf.getnframes())

    if sampwidth == 2:
        signal = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        signal = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0

    if signal.size == 0:
        return None

    loud = np.flatnonzero(np.abs(signal) > 10.0 ** (threshold_db / 20.0))
    if loud.size == 0:
        return None

    return float(loud[0]) / framerate, float(loud[-1] + 1) / framerate


def estimate_gender(audio_path: str, start_time: float, end_time: float) -> str:
    """Estimate speaker gender from a WAV segment via pitch autocorrelation.

    Returns "male" if the median fundamental frequency is below 165 Hz,
    "female" otherwise.  Defaults to "female" on any processing failure.

    The 165 Hz threshold is a commonly used mid-point between the typical
    male range (85–180 Hz) and female range (165–255 Hz).
    """
    try:
        return _pitch_based_gender(audio_path, start_time, end_time)
    except Exception as exc:
        logger.warning("Gender estimation failed for [%.2fs–%.2fs]: %s", start_time, end_time, exc)
        return "female"


def _pitch_based_gender(audio_path: str, start_time: float, end_time: float) -> str:
    """Internal: load segment frames, run autocorrelation, classify by median pitch."""
    signal, framerate = _read_segment(audio_path, start_time, end_time)
    if signal is None:
        return "female"

    pitches = _estimate_pitches(signal, framerate)
    if not pitches:
        return "female"

    median_pitch = float(np.median(pitches))
    gender = "male" if median_pitch < 165 else "female"
    logger.info(
        "Segment [%.2fs–%.2fs]: pitch=%.1f Hz → %s",
        start_time, end_time, median_pitch, gender
    )
    return gender


def _read_segment(
    audio_path: str, start_time: float, end_time: float
) -> tuple[np.ndarray | None, int]:
    """Read raw PCM samples for the given time range from a WAV file."""
    with wave.open(audio_path, "rb") as wf:
        framerate = wf.getframerate()
        sampwidth = wf.getsampwidth()
        n_frames = int((end_time - start_time) * framerate)

        if n_frames <= 0:
            return None, framerate

        wf.setpos(int(start_time * framerate))
        data = wf.readframes(n_frames)

    if sampwidth == 2:
        signal = np.frombuffer(data, dtype=np.int16)
    else:
        # Convert 8-bit unsigned to signed int16
        signal = np.frombuffer(data, dtype=np.uint8).astype(np.int16) - 128

    return signal, framerate


def _estimate_pitches(signal: np.ndarray, framerate: int) -> list[float]:
    """Run per-frame autocorrelation and collect fundamental frequency estimates."""
    # Lag bounds corresponding to 50–300 Hz pitch range
    min_lag = int(framerate / 300)
    max_lag = int(framerate / 50)
    frame_size = int(0.05 * framerate)  # 50 ms analysis window

    pitches: list[float] = []
    for i in range(0, len(signal) - frame_size, frame_size):
        frame = signal[i : i + frame_size]
        # Skip near-silence frames to avoid noise contaminating the pitch estimate
        if np.std(frame) < 100:
            continue

        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(corr) // 2 :]

        if len(corr) <= max_lag:
            continue

        peak_lag = int(np.argmax(corr[min_lag:max_lag])) + min_lag
        pitch = framerate / peak_lag
        if 50 <= pitch <= 300:
            pitches.append(pitch)

    return pitches


def _self_check() -> None:
    """Smoke test for speech_bounds: internal pauses must survive the trim."""
    import tempfile

    sr = 22050

    def tone(seconds: float) -> np.ndarray:
        t = np.arange(int(seconds * sr)) / sr
        return (0.5 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)

    def hush(seconds: float) -> np.ndarray:
        return np.zeros(int(seconds * sr), dtype=np.int16)

    pcm = np.concatenate([hush(0.3), tone(0.4), hush(0.2), tone(0.3), hush(0.5)])
    with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
        with wave.open(fh.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

        bounds = speech_bounds(fh.name)
        assert bounds is not None, "expected speech, got silence"
        start, end = bounds
        assert abs(start - 0.3) < 0.01, start
        # 1.2 s, not 1.0: the 0.2 s pause between the two tones is kept.
        assert abs(end - 1.2) < 0.01, end

        with wave.open(fh.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(hush(0.2).tobytes())
        assert speech_bounds(fh.name) is None, "all-silent WAV must return None"

    print("audio_utils self-check OK")


if __name__ == "__main__":
    _self_check()
