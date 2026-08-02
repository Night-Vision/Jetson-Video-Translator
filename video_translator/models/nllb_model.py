from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List

import ctranslate2
from transformers import AutoTokenizer

from .base import BaseModel

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger("video_translator.nllb_model")

class NLLBModel(BaseModel):
    """NLLB translation model via CTranslate2."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._translator = None
        self._tokenizer = None
        self._ct2_model_path = os.path.join(self.config.nllb_cache_dir, "ct2_model")

    def _convert_model(self) -> None:
        if os.path.exists(self._ct2_model_path) and os.listdir(self._ct2_model_path):
            logger.info("CTranslate2 model already converted at %s", self._ct2_model_path)
            return

        logger.info("Converting HuggingFace model %s to CTranslate2 format...", self.config.model_id)
        converter = ctranslate2.converters.TransformersConverter(
            self.config.model_id,
            load_as_float16=True
        )
        converter.convert(self._ct2_model_path, quantization=self.config.nllb_compute_type, force=True)
        logger.info("Conversion complete.")

    def _load(self) -> None:
        self._check_vram_headroom()
        self._convert_model()
        logger.info("Loading CTranslate2 model from %s", self._ct2_model_path)
        self._translator = ctranslate2.Translator(
            self._ct2_model_path,
            device=self.config.nllb_device,
            compute_type=self.config.nllb_compute_type
        )
        logger.info("Loading tokenizer for %s", self.config.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            cache_dir=self.config.nllb_cache_dir,
            src_lang="eng_Latn"
        )

    def _unload(self) -> None:
        if self._translator is not None:
            logger.info("Unloading CTranslate2 translator to free VRAM...")
            del self._translator
            self._translator = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

    def reset(self) -> None:
        self._unload()

    def translate_batch(self, texts: List[str], target_prefix: str) -> List[str]:
        if not self._translator or not self._tokenizer:
            raise RuntimeError("NLLB model is not loaded.")
        
        self._tokenizer.src_lang = "eng_Latn"
        
        source = [self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text)) for text in texts]
        target_prefix_tokens = [[target_prefix]] * len(texts)
        
        results = self._translator.translate_batch(
            source,
            target_prefix=target_prefix_tokens,
            max_decoding_length=256
        )
        
        translations = []
        for res in results:
            target = res.hypotheses[0][1:]
            translation = self._tokenizer.decode(
                self._tokenizer.convert_tokens_to_ids(target)
            )
            translations.append(translation)
            
        return translations

    def _check_vram_headroom(self) -> None:
        try:
            import pynvml  # noqa: PLC0415
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = info.free / 1024 ** 3
            if free_gb < 1.0:
                raise MemoryError(
                    f"Insufficient VRAM: {free_gb:.1f} GB free, require at least 1GB."
                )
        except MemoryError:
            raise
        except Exception:
            pass
