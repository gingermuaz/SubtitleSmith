import unittest

from faster_whisper.tokenizer import _LANGUAGE_CODES

import languages


class LanguageTests(unittest.TestCase):
    def test_every_whisper_language_has_a_name(self):
        self.assertEqual(sorted(c for c in _LANGUAGE_CODES if c not in languages.NAMES), [])

    def test_every_whisper_language_has_a_track_label(self):
        for code in _LANGUAGE_CODES:
            label = languages.iso639_2(code)
            self.assertTrue(len(label) == 3 and label.isalpha() and label != "und", code)
        self.assertEqual((languages.iso639_2("ja"), languages.iso639_2("xx")), ("jpn", "und"))

    def test_translation_codes(self):
        missing = sorted(c for c in _LANGUAGE_CODES if languages.nllb_code(c) is None)
        self.assertEqual(missing, ["br", "haw", "la"])  # NLLB-200 has no Breton, Hawaiian or Latin
        self.assertEqual((languages.nllb_code("bn"), languages.nllb_code("zh")), ("ben_Beng", "zho_Hans"))

    def test_describe(self):
        self.assertEqual(languages.describe("ja", 0.966), "Japanese (97%)")
        self.assertEqual(languages.describe("xx", 0.5), "xx (50%)")

    def test_line_lengths(self):
        self.assertEqual(languages.line_length("en"), 42)
        self.assertEqual(languages.line_length("ja"), 13)
        self.assertEqual(languages.line_length("zh"), 16)


if __name__ == "__main__":
    unittest.main()
