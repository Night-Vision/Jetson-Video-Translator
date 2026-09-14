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

        with self._llm:
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


def _self_check() -> None:
    """Each segment must be translated exactly once.

    translate_segments() re-islices its argument on every batch, so it needs a
    true iterator: islice on a list restarts at index 0 and loops forever.
    """
    from types import SimpleNamespace

    class _StubLLM:
        """`with` resolves __enter__/__exit__ on the type, so SimpleNamespace
        cannot stand in for a context-managed model."""

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

        @staticmethod
        def translate_batch(texts, prefix):
            return [f"ru:{t}" for t in texts]

    n, batch = 7, 3
    segments = [SimpleNamespace(text=f"s{i}", translated_text=None) for i in range(n)]

    translator = Translator.__new__(Translator)   # skip __init__: no model load
    translator.config = SimpleNamespace(
        nllb_batch_size=batch,
        target_lang="Russian",
        lang_mappings={"Russian": "rus_Cyrl"},
    )
    translator._llm = _StubLLM()

    out = list(translator.translate_segments(iter(segments)))
    assert len(out) == n, f"expected {n} segments, got {len(out)}"
    assert [s.translated_text for s in out] == [f"ru:s{i}" for i in range(n)], out

    # A list (not an iterator) would restart every islice -- guard the guard.
    looped = list(itertools.islice(itertools.islice(segments, batch), batch))
    assert [s.text for s in looped] == ["s0", "s1", "s2"], "islice-on-list assumption changed"

    print("translator self-check OK")


if __name__ == "__main__":
    _self_check()
