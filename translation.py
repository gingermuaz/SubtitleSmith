"""Translates subtitle text into other languages with Meta's NLLB-200 model, run locally by CTranslate2.

There are three sizes (MODELS). Each downloads once, the first time it's needed. NLLB-200's
licence (CC-BY-NC 4.0) allows non-commercial use only.
"""
import gc
import os

import ctranslate2
import tokenizers

import languages
from errors import ProcessCancelled, ProcessingError

# NLLB-200 converted to CTranslate2 (int8), smallest first. Bigger models translate better but need
# more memory and time. They share the same tokenizer and language codes, so any of them works here.
MODELS = {
    "fast": "skywood/nllb-200-distilled-600M-ct2-int8",  # 600 MB
    "better": "OpenNMT/nllb-200-distilled-1.3B-ct2-int8",  # 1.4 GB
    "best": "OpenNMT/nllb-200-3.3B-ct2-int8",  # 3.4 GB
}
DEFAULT = "better"
FILES = ["config.json", "model.bin", "shared_vocabulary.*", "tokenizer.json"]
BATCH_SIZE = 32


class Translator:
    """An NLLB-200 model loaded into memory, ready to translate."""

    def __init__(self, folder, device="auto", device_index=0):
        compute_type = "int8" if device == "cpu" else "int8_float16"
        self.model = ctranslate2.Translator(folder, device=device, device_index=device_index,
                                            compute_type=compute_type)
        self.tokenizer = tokenizers.Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))

    def translate(self, texts, source, target, cancel_check=None, on_progress=None):
        """Translates a list of texts from one Whisper language code to another ('ja' -> 'en').

        Empty texts stay empty. on_progress(done, total) reports how many texts are finished.
        Raises ProcessCancelled if cancel_check() becomes true between batches.
        """
        source_code, target_code = languages.nllb_code(source), languages.nllb_code(target)
        for code, language in ((source_code, source), (target_code, target)):
            if code is None:
                raise ProcessingError(f"The translation model can't translate {languages.name(language)}.")
        results = [""] * len(texts)
        todo = [(i, text.strip()) for i, text in enumerate(texts) if text.strip()]
        for start in range(0, len(todo), BATCH_SIZE):
            if cancel_check and cancel_check():
                raise ProcessCancelled()
            batch = todo[start:start + BATCH_SIZE]
            # NLLB's input is: source language code, the text's tokens, end-of-sentence
            sources = [[source_code] + self.tokenizer.encode(text, add_special_tokens=False).tokens + ["</s>"]
                       for _, text in batch]
            outputs = self._translate_batch(sources, target_code)
            for (i, _), output in zip(batch, outputs):
                tokens = [t for t in output.hypotheses[0] if t != target_code]  # it starts with the language code
                ids = [self.tokenizer.token_to_id(t) for t in tokens]
                results[i] = self.tokenizer.decode([i_ for i_ in ids if i_ is not None], skip_special_tokens=True).strip()
            if on_progress:
                on_progress(min(start + BATCH_SIZE, len(todo)), len(todo))
        return results

    def _translate_batch(self, sources, target_code):
        """Translates a batch, splitting it in half (and again if need be) if the GPU runs out of memory."""
        try:
            return self.model.translate_batch(sources, target_prefix=[[target_code]] * len(sources), beam_size=4)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if len(sources) == 1:
                raise ProcessingError("Not enough GPU memory for this translation model. A smaller one "
                                      "(Fast or Better) would fit.") from e
        gc.collect()
        half = len(sources) // 2
        return self._translate_batch(sources[:half], target_code) + self._translate_batch(sources[half:], target_code)
