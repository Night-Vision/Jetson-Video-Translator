# Third-Party Notices

The VideoTranslator source code in this repository is licensed under the MIT
License — see [`LICENSE`](LICENSE). Nothing in this file changes that.

This file lists the third-party software and model weights the pipeline
depends on. **No model weights are distributed in this repository** — they are
downloaded at runtime or installed separately, and they carry their own
licenses, listed below.

---

## Models

### NLLB-200 — non-commercial

`facebook/nllb-200-distilled-600M`, used for translation
(`video_translator/pipeline/translator.py`, `models/nllb_model.py`).

**License: CC-BY-NC-4.0 — non-commercial use only.**

Upstream states the model is "a research model and is not released for
production deployment." Downloaded at runtime by `huggingface_hub` into
`models/nllb/`, then converted to CTranslate2 format and cached.

Using this project for any commercial purpose requires replacing NLLB-200 with
a permissively licensed translation model. Research and demo use is fine.

Model card: https://huggingface.co/facebook/nllb-200-distilled-600M

### Whisper

OpenAI Whisper weights, served through `faster-whisper` and downloaded at
runtime from the HuggingFace Hub (`video_translator/models/whisper_model.py`).

**License: MIT** (OpenAI).

### Piper voices

Piper `.onnx` voice models and their `.onnx.json` configs
(`video_translator/pipeline/dubber.py`).

**Installed manually — not downloaded at runtime.** The bundled Piper binary
accepts only a local model path (`--model FILE`) and has no download
capability. Voices must be placed in `models/` by hand; the dubber logs a
warning and fails for that voice if one is missing.

**Licenses vary per voice.** Each voice has its own `MODEL_CARD` file upstream,
next to the `.onnx` file, which names its source dataset and license status.
Check the `MODEL_CARD` for any voice you use or redistribute — some do not
state a license at all. The `ru_RU-irina-medium` and `ru_RU-dmitri-medium`
voices this project defaults to derive from the RHVoice project and their
model cards list their license as "Unknown".

Voice collection: https://huggingface.co/rhasspy/piper-voices

Only the two `.onnx.json` config files are tracked in this repository; the
`.onnx` weights are gitignored.

---

## Python dependencies

| Package | License |
|---|---|
| faster-whisper | MIT |
| CTranslate2 | MIT |
| huggingface_hub | Apache-2.0 |
| tokenizers | Apache-2.0 |
| requests | Apache-2.0 |
| psutil | BSD-3-Clause |
| nvidia-ml-py | BSD |
| numpy | BSD-3-Clause |

CTranslate2 is compiled from source on Jetson (`compile_ctranslate2.sh`) rather
than installed from PyPI; the license is the same either way.

---

## External tools

These are invoked as subprocesses from `PATH` and are not linked into this
project. They are not bundled here and must be installed separately.

| Tool | License | Used by |
|---|---|---|
| yt-dlp | Unlicense | `pipeline/downloader.py` |
| ffmpeg / ffprobe | LGPL or GPL, depending on build | `pipeline/extractor.py`, `pipeline/dubber.py` |
| Piper (binary) | MIT | `pipeline/dubber.py` |

Note on ffmpeg: distribution builds configured with `--enable-gpl` (including
the Ubuntu package commonly present on JetPack) are GPL-licensed. This project
only ever runs `ffmpeg` and `ffprobe` as separate processes and never links
against their libraries, so that licensing does not propagate to this
repository's code. Check your own build with `ffmpeg -version` if you intend to
redistribute a bundle that includes it.
