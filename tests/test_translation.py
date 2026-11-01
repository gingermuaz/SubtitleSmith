import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import tokenizers
from tokenizers import models, pre_tokenizers, decoders

import translation
from errors import ProcessCancelled, ProcessingError


def make_tokenizer(folder):
    """A small real tokenizer built like NLLB's (SentencePiece-style pieces with ▁ for spaces)."""
    vocab = {"<s>": 0, "</s>": 1, "<unk>": 2, "eng_Latn": 3, "fra_Latn": 4, "ben_Beng": 5}
    for word in ["Hello", "there", "friend", "Bonjour", "mon", "ami", ".", ",", "!"]:
        vocab.setdefault("▁" + word, len(vocab))
        vocab.setdefault(word, len(vocab))
    tokenizer = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Metaspace(), pre_tokenizers.Punctuation()])
    tokenizer.decoder = decoders.Metaspace()
    tokenizer.add_special_tokens(["<s>", "</s>", "<unk>", "eng_Latn", "fra_Latn", "ben_Beng"])
    tokenizer.save(os.path.join(folder, "tokenizer.json"))


class FakeModel:
    """Stands in for ctranslate2.Translator: 'translates' by swapping known words."""
    WORDS = {"▁Hello": "▁Bonjour", "▁there": "▁mon", "▁friend": "▁ami"}

    def __init__(self, *args, **kwargs):
        self.calls = []

    def translate_batch(self, sources, target_prefix, beam_size):
        self.calls.append((sources, target_prefix))
        return [NS(hypotheses=[prefix + [self.WORDS.get(t, t) for t in source[1:-1]]])
                for source, prefix in zip(sources, target_prefix)]


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        make_tokenizer(self._temp.name)
        with mock.patch.object(translation.ctranslate2, "Translator", FakeModel):
            self.translator = translation.Translator(self._temp.name, "cpu")

    def test_input_format_and_output(self):
        result = self.translator.translate(["Hello there, friend!", "", "  "], "en", "fr")
        self.assertEqual(result, ["Bonjour mon, ami!", "", ""])
        sources, prefixes = self.translator.model.calls[0]
        self.assertEqual(sources, [["eng_Latn", "▁Hello", "▁there", ",", "▁friend", "!", "</s>"]])
        self.assertEqual(prefixes, [["fra_Latn"]])

    def test_batches_and_progress(self):
        progress = []
        with mock.patch.object(translation, "BATCH_SIZE", 2):
            self.translator.translate(["Hello"] * 5, "en", "fr", on_progress=lambda d, t: progress.append((d, t)))
        self.assertEqual(progress, [(2, 5), (4, 5), (5, 5)])
        self.assertEqual(len(self.translator.model.calls), 3)

    def test_cancel_between_batches(self):
        with mock.patch.object(translation, "BATCH_SIZE", 1), self.assertRaises(ProcessCancelled):
            self.translator.translate(["Hello", "Hello"], "en", "fr", cancel_check=lambda: True)

    def test_out_of_gpu_memory_splits_the_batch(self):
        model = self.translator.model
        real = model.translate_batch

        def limited(sources, target_prefix, beam_size):
            if len(sources) > 2:
                raise RuntimeError("CUDA failed with error out of memory")
            return real(sources, target_prefix, beam_size)
        model.translate_batch = limited
        with mock.patch.object(translation, "BATCH_SIZE", 8):
            result = self.translator.translate(["Hello friend"] * 7, "en", "fr")
        self.assertEqual(result, ["Bonjour ami"] * 7)
        self.assertEqual(sorted(len(sources) for sources, _ in model.calls), [1, 2, 2, 2])

    def test_out_of_gpu_memory_even_one_at_a_time(self):
        def broke(sources, target_prefix, beam_size):
            raise RuntimeError("CUDA failed with error out of memory")
        self.translator.model.translate_batch = broke
        with self.assertRaisesRegex(ProcessingError, "Not enough GPU memory"):
            self.translator.translate(["Hello", "there"], "en", "fr")

    def test_other_errors_are_not_retried(self):
        self.translator.model.translate_batch = mock.Mock(side_effect=RuntimeError("bad model file"))
        with self.assertRaisesRegex(RuntimeError, "bad model file"):
            self.translator.translate(["Hello", "there"], "en", "fr")
        self.assertEqual(self.translator.model.translate_batch.call_count, 1)

    def test_models(self):
        self.assertEqual(list(translation.MODELS), ["fast", "better", "best"])
        self.assertIn(translation.DEFAULT, translation.MODELS)

    def test_languages_it_cannot_translate(self):
        with self.assertRaisesRegex(ProcessingError, "can't translate Latin"):
            self.translator.translate(["Salve"], "la", "en")


if __name__ == "__main__":
    unittest.main()
