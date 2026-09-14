from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    # ── Pipeline behaviour ──────────────────────────────────────────────────
    target_lang: str = "Russian"
    debug: bool = False
    # Output mode: "dubbed" | "srt" | "vtt" | "json"
    output_format: str = "dubbed"
    # Background music/audio ducking toggle and volume multiplier (0.0 to 1.0)
    enable_ducking: bool = False
    bg_volume: float = 0.0
    # Max TTS speech-rate speedup when the translation is longer than its
    # window (1.0 = never speed up; excess is cut / spills into the pause).
    max_tempo: float = 1.35
    # Seconds to shift the dub by, compensating for a container where the
    # audio and video streams do not share a start PTS.  None = auto-probe.
    audio_offset: float | None = None

    # ── Intermediate paths — /dev/shm/ is RAM-backed tmpfs on Linux ─────────
    tmp_video: str = "/dev/shm/input_vid.mp4"
    tmp_audio: str = "/dev/shm/raw_audio.wav"
    tmp_segments: str = "/dev/shm/segments.json"
    # Both are resolved to outputs/<name>_NNN by __post_init__; never used as-is.
    output_video: str = ""
    output_subtitles: str = ""

    # ── Runtime-resolved paths (populated by __post_init__ if left empty) ───
    piper_bin: str = ""
    tts_models_dir: str = ""

    # ── Whisper ─────────────────────────────────────────────────────────────
    whisper_model: str = "small"
    word_timestamps: bool = False
    vad_filter: bool = True
    # Minimum free RAM before attempting to load Whisper
    min_free_ram_gb: float = 2.0

    # ── NLLB Translation ────────────────────────────────────────────────────
    model_id: str = "facebook/nllb-200-distilled-600M"
    nllb_cache_dir: str = ""
    nllb_batch_size: int = 4
    nllb_device: str = "cuda"
    nllb_compute_type: str = "int8"
    
    # ISO to FLORES-200 mapping for translation
    lang_mappings: dict = None

    def __post_init__(self) -> None:
        # This file lives at video_translator/config.py; the project root is one level up.
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(pkg_dir)

        if not self.piper_bin:
            local_piper = os.path.join(project_root, "piper", "piper")
            self.piper_bin = local_piper if os.path.exists(local_piper) else "piper"

        if not self.tts_models_dir:
            self.tts_models_dir = os.path.join(project_root, "models")

        if not self.nllb_cache_dir:
            self.nllb_cache_dir = os.path.join(project_root, "models", "nllb")

        if self.lang_mappings is None:
            self.lang_mappings = {
                "Russian": "rus_Cyrl",
                "English": "eng_Latn",
                "Spanish": "spa_Latn",
                "French": "fra_Latn",
                "German": "deu_Latn",
                "Chinese": "zho_Hans",
                "Italian": "ita_Latn",
                "Japanese": "jpn_Jpan",
                "Korean": "kor_Hang",
                "Portuguese": "por_Latn"
            }

        # Ensure outputs directory exists and resolve paths
        outputs_dir = os.path.join(project_root, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        
        # One listing, not one per candidate slot.  Advance past both video and
        # subtitle slots so repeated subtitle-only runs don't overwrite the
        # previous subtitles_NNN file.
        existing = set(os.listdir(outputs_dir))
        i = 1
        while True:
            video_path = os.path.join(outputs_dir, f"dubbed_{i:03d}.mp4")
            sub_base = f"subtitles_{i:03d}."
            slot_free = f"dubbed_{i:03d}.mp4" not in existing and not any(
                name.startswith(sub_base) for name in existing
            )
            if slot_free:
                self.output_video = video_path
                self.output_subtitles = os.path.join(outputs_dir, f"subtitles_{i:03d}")
                break
            i += 1
