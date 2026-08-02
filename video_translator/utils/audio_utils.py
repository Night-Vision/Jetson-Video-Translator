from __future__ import annotations

import logging
import wave

import numpy as np

logger = logging.getLogger("video_translator.audio_utils")


def get_wav_duration(file_path: str) -> float:
    """Return the duration of a WAV file in seconds using stdlib wave (no subprocess)."""
    with wave.open(file_path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


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
