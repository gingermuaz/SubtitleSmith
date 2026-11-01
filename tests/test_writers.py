import os
import tempfile
import unittest
from types import SimpleNamespace as NS

import writers

SEGMENTS = [
    NS(start=0.5, end=2.5, text=" Hello there, friend.", words=[NS(start=0.5, end=1.0, word=" Hello"),
                                                               NS(start=1.0, end=2.5, word=" there, friend.")]),
    NS(start=3661.25, end=3663.0, text=" Second line ", words=None),
]


class FormatTimestampTests(unittest.TestCase):
    def test_srt_uses_comma(self):
        self.assertEqual(writers.format_timestamp(3661.5), "01:01:01,500")

    def test_vtt_uses_dot(self):
        self.assertEqual(writers.format_timestamp(0.25, is_vtt=True), "00:00:00.250")

    def test_zero(self):
        self.assertEqual(writers.format_timestamp(0), "00:00:00,000")


class RenderTests(unittest.TestCase):
    def test_srt(self):
        self.assertEqual(writers.render(SEGMENTS, ".srt"),
                         "1\n00:00:00,500 --> 00:00:02,500\nHello there, friend.\n\n"
                         "2\n01:01:01,250 --> 01:01:03,000\nSecond line\n\n")

    def test_vtt(self):
        text = writers.render(SEGMENTS, ".vtt")
        self.assertTrue(text.startswith("WEBVTT\n\n1\n00:00:00.500 --> 00:00:02.500\n"))

    def test_txt_has_no_timings(self):
        self.assertEqual(writers.render(SEGMENTS, ".txt"), "Hello there, friend.\nSecond line\n")

    def test_word_timestamps_give_one_block_per_word(self):
        text = writers.render(SEGMENTS, ".srt", word_timestamps=True)
        self.assertIn("1\n00:00:00,500 --> 00:00:01,000\nHello\n\n", text)
        self.assertIn("2\n00:00:01,000 --> 00:00:02,500\nthere, friend.\n\n", text)
        # A segment without word timings falls back to one block
        self.assertIn("3\n01:01:01,250 --> 01:01:03,000\nSecond line\n\n", text)

    def test_unknown_format(self):
        with self.assertRaises(ValueError):
            writers.render(SEGMENTS, ".docx")

    def test_write_saves_utf8(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "out.srt")
            writers.write(path, [NS(start=0, end=1, text="Café – 日本語", words=None)], ".srt")
            with open(path, encoding="utf-8") as f:
                self.assertIn("Café – 日本語", f.read())


if __name__ == "__main__":
    unittest.main()
