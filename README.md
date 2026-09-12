# VideoTranslator

![Platform](https://img.shields.io/badge/platform-Jetson%20Orin%20Nano-76B900)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Models](https://img.shields.io/badge/NLLB--200-CC--BY--NC--4.0-orange)

Local, offline video dubbing pipeline built for the **Nvidia Jetson Orin Nano (8GB)**. Downloads a video, transcribes it, translates the speech, and generates a dubbed track (or subtitles) — no cloud APIs, no inference sent off-device. Only the download stage needs the network.

## What it does

```
URL → download → extract audio → transcribe → translate → dub/subtitle → output
```

1. **Download** — pulls the video via `yt-dlp`, preferring H.264 so decode stays on NVDEC (`downloader.py`)
2. **Extract** — pulls a mono 22.05kHz WAV track via `ffmpeg` (`extractor.py`)
3. **Transcribe** — `faster-whisper` (INT8), run in an isolated subprocess so the CUDA context is fully released afterward (`transcriber.py`)
4. **Translate** — NLLB-200 via CTranslate2, batched (`translator.py`, `models/nllb_model.py`)
5. **Output** — either:
   - **Dubbed video**: Piper TTS generates speech per segment, tempo-adjusted to fit the original timing, muxed back over the source video — replacing the original audio by default, or ducking it under the dub with `--enable-ducking` (`dubber.py`)
   - **Subtitles**: SRT / VTT / JSON (`writer.py`)

## Why it's built this way

Jetson has **unified memory** — CPU and GPU share the same 8GB pool. That drives most of the architecture:

- **Subprocess isolation for Whisper.** CUDA contexts don't fully release inside the same process. Whisper runs in a spawned subprocess so its VRAM is guaranteed back before the translator loads.
- **Explicit load/unload lifecycle.** Every heavy model (`WhisperModel`, `NLLBModel`) inherits `BaseModel`, which enforces a `lifecycle()` context manager — load, use, unload, `gc.collect()` — even on exceptions.
- **Strict stage boundaries.** Each pipeline stage takes typed `Segment` dataclasses in and yields them out. No stage silently swallows failure — they raise, and `main.py` handles the abort.

## Requirements

- Jetson Orin Nano (or similar Jetson w/ CUDA + cuDNN)
- Python 3 virtual environment
- `ffmpeg` / `ffprobe`, `yt-dlp` + `yt-dlp-ejs` on PATH — keep the pair version-matched, or YouTube returns `403 Forbidden` on every H.264 format
- CTranslate2 compiled from source for aarch64+CUDA (see `compile_ctranslate2.sh` — **do not** `pip install` it on Jetson)
- Piper TTS binary + voice models under `models/`

```bash
pip install -r requirements.txt
```

## Usage

```bash
./.venv/bin/python main.py <video_url> --lang Russian --format dubbed
```

| Flag | Default | Description |
|---|---|---|
| `--lang` | `Russian` | Target language (must exist in `config.py`'s `lang_mappings`) |
| `--format` | `dubbed` | `dubbed` \| `srt` \| `vtt` \| `json` |
| `--enable-ducking` | off | Duck the original audio under the dub instead of replacing it |
| `--bg-volume` | `0.0` | Original audio volume under the dub; any value > 0 implies `--enable-ducking` |
| `--audio-offset` | auto | Shift the dub by N seconds; auto-probes the a/v stream start PTS |
| `--max-tempo` | `1.35` | Max speedup applied to TTS audio that overruns its time slot |
| `--debug` | off | Verbose RAM / overcurrent logging |

Output lands in `outputs/dubbed_NNN.mp4` or `outputs/subtitles_NNN.<ext>`.

## Project layout

```
main.py                        # orchestration only, no business logic
video_translator/
├── config.py                  # single Config dataclass, all tunables
├── models/
│   ├── base.py                 # BaseModel — enforces load/use/unload lifecycle
│   ├── whisper_model.py        # faster-whisper wrapper, RAM guard
│   ├── nllb_model.py           # NLLB + CTranslate2 wrapper, VRAM guard
│   └── segment.py               # Segment dataclass — passed between all stages
├── pipeline/
│   ├── downloader.py           # yt-dlp
│   ├── extractor.py            # ffmpeg audio extraction
│   ├── transcriber.py          # Whisper stage (spawns isolated subprocess)
│   ├── translator.py           # NLLB batch translation
│   ├── dubber.py                # Piper TTS + tempo-fit + ffmpeg mux
│   └── writer.py                # SRT/VTT/JSON subtitle export
└── utils/
    ├── memory_monitor.py        # RAM + SoC overcurrent counter logging
    └── audio_utils.py           # speech-span detection, pitch-based gender detection
```

## Known limitations

- Translation covers 10 languages; dubbing voices are mapped for only 5 (English, Russian, Spanish, French, German). Other targets fall back to the English voice.
- Only Russian ships distinct male/female Piper voices, and only the Russian voices are checked into `models/`; other languages reuse one voice for both genders until more Piper models are added.
- TTS can only speed up to fit a time slot (`--max-tempo`), never slow down — segments that finish early just end early.
- Silent/unvocalizable TTS segments are dropped; original audio is kept in that window instead of a dub.
- The SoC overcurrent counter (`/sys/class/hwmon/hwmon*/oc3_event_cnt`) occasionally increments mid-run on the uncapped `MAXN_SUPER` power profile. `--debug` reports which stage it fired in. Cause not yet identified; `sudo nvpmodel -m 1` caps power if it becomes a problem.

## License

MIT — Copyright (c) 2026 Ilya Ashirov. See [`LICENSE`](LICENSE).

Model weights are **not** covered by this license and are downloaded at runtime:

- **NLLB-200** (`facebook/nllb-200-distilled-600M`) — CC-BY-NC-4.0, **non-commercial only**. Upstream states it is "a research model and is not released for production deployment."
- **Whisper** — MIT (OpenAI).
- **Piper voices** — vary per voice; check the source of each voice before redistributing.

Commercial use requires swapping NLLB for a permissively licensed translation model.
