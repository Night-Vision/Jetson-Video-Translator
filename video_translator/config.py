from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    # ── Pipeline behaviour ──────────────────────────────────────────────────
    target_lang: str = "Russian"
    platform: str = "short-form video"
    debug: bool = False
    # Output mode: "dubbed" | "srt" | "vtt" | "json"
    output_format: str = "dubbed"
    # Background music/audio ducking volume multiplier (0.0 to 1.0)
    bg_volume: float = 0.15

    # ── Intermediate paths — /dev/shm/ is RAM-backed tmpfs on Linux ─────────
    tmp_video: str = "/dev/shm/input_vid.mp4"
    tmp_audio: str = "/dev/shm/raw_audio.wav"
    tmp_segments: str = "/dev/shm/segments.json"
    output_video: str = "final_dubbed.mp4"
    # Base name for subtitle files; extension appended from output_format
    output_subtitles: str = "subtitles"

    # ── Runtime-resolved paths (populated by __post_init__ if left empty) ───
    piper_bin: str = ""
    tts_models_dir: str = ""
    # Optional Firefox profile path for yt-dlp cookie auth; leave empty to skip
    browser_cookies_path: str = ""

    # ── Whisper ─────────────────────────────────────────────────────────────
    whisper_model: str = "small"
    word_timestamps: bool = False
    vad_filter: bool = True
    # Minimum free RAM before attempting to load Whisper
    min_free_ram_gb: float = 2.0
    # Backend: "faster_whisper" (CTranslate2, default) | "whisper_trt" (experimental)
    whisper_backend: str = "faster_whisper"
    # Path to pre-built TensorRT engine directory (only used when whisper_backend == "whisper_trt")
    whisper_trt_engine_dir: str = ""

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
        
        i = 1
        while True:
            video_path = os.path.join(outputs_dir, f"dubbed_{i:03d}.mp4")
            # Advance past both video and subtitle slots, so repeated
            # subtitle-only runs don't overwrite the previous subtitles_NNN file.
            sub_base = f"subtitles_{i:03d}."
            slot_free = not os.path.exists(video_path) and not any(
                name.startswith(sub_base) for name in os.listdir(outputs_dir)
            )
            if slot_free:
                self.output_video = video_path
                self.output_subtitles = os.path.join(outputs_dir, f"subtitles_{i:03d}")
                break
            i += 1
