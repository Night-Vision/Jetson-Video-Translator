from __future__ import annotations

import itertools
import logging
from typing import Generator, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..models.segment import Segment

logger = logging.getLogger("video_translator.translator")


class Translator:
    """Pipeline stage: translate Segment text via NLLB CTranslate2 model."""

    def __init__(self, config: Config) -> None:
        self.config = config
        from ..models.nllb_model import NLLBModel
        self._llm = NLLBModel(config)

    def reset(self) -> None:
        """Stop or kill any stale LLM service (safe to call when not running)."""
        self._llm.reset()

    def translate_segments(self, segment_iter) -> Generator:
        # Peek at the first chunk before starting the expensive model.
        first_chunk = list(itertools.islice(segment_iter, self.config.nllb_batch_size))
        if not first_chunk:
            return

        with self._llm.lifecycle():
            yield from self._process_all_chunks(first_chunk, segment_iter)

    # ── Private: chunk processing ────────────────────────────────────────────

    def _process_all_chunks(self, first_chunk: list, segment_iter) -> Generator:
        chunk = first_chunk
        
        target_prefix = self.config.lang_mappings.get(self.config.target_lang)
        if not target_prefix:
            raise ValueError(f"No FLORES-200 mapping for target language '{self.config.target_lang}'. Check config.py.")
            
        logger.info(
            "Translating segments to %s using prefix %s...", 
            self.config.target_lang, target_prefix
        )

        while chunk:
            texts = [seg.text for seg in chunk]
            try:
                translations = self._llm.translate_batch(texts, target_prefix)
                for seg, tgt in zip(chunk, translations):
                    seg.translated_text = tgt
                    yield seg
            except Exception as e:
                logger.error("Translation failed for chunk: %s", e)
                # Fallback to returning original text if translation completely fails
                for seg in chunk:
                    seg.translated_text = seg.text
                    yield seg
                    
            chunk = list(itertools.islice(segment_iter, self.config.nllb_batch_size))
