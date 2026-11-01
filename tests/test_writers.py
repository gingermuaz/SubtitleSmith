import os
import tempfile
import unittest
from types import SimpleNamespace as NS

import writers


def segment(text, start, seconds_per_word=0.4, pauses=None, timed=True):
    """A faster-whisper-like segment with evenly spaced words; pauses = {word index: extra seconds before it}."""
    words, t = [], start
    for i, w in enumerate(text.split(" ")):
        t += (pauses or {}).get(i, 0)
        words.append(NS(start=t, end=t + seconds_per_word * 0.8, word=" " + w))
        t += seconds_per_word
    return NS(start=start, end=words[-1].end, text=" " + text, words=words if timed else None)


LONG = ("I told you already, we can't stay here any longer. The storm is coming in from the north, "
        "and if we don't leave before sunset, we'll be trapped on this mountain for days. "
        "Pack only what you can carry, and meet me at the bridge in ten minutes.")


class FormatTimestampTests(unittest.TestCase):
    def test_srt_uses_comma(self):
        self.assertEqual(writers.format_timestamp(3661.5), "01:01:01,500")

    def test_vtt_uses_dot(self):
        self.assertEqual(writers.format_timestamp(0.25, is_vtt=True), "00:00:00.250")

    def test_rounds_instead_of_cutting_off(self):
        self.assertEqual(writers.format_timestamp(2.3), "00:00:02,300")  # used to be ,299
        self.assertEqual(writers.format_timestamp(1.001), "00:00:01,001")  # used to be ,000
        self.assertEqual(writers.format_timestamp(59.9996), "00:01:00,000")

    def test_never_negative(self):
        self.assertEqual(writers.format_timestamp(-0.01), "00:00:00,000")


class ReadableLinesTests(unittest.TestCase):
    def check_rules(self, cues, max_chars=42):
        for cue in cues:
            self.assertLessEqual(len(cue.lines), 2)
            for line in cue.lines:
                self.assertLessEqual(len(line), max_chars, line)
            self.assertLessEqual(cue.end - cue.start, writers.MAX_DURATION + 1e-9)
        for cue, following in zip(cues, cues[1:]):
            self.assertLessEqual(cue.end, following.start + 1e-9, "subtitles must not overlap")

    def test_long_segment_is_split_into_readable_subtitles(self):
        cues = writers.make_cues([segment(LONG, 10.0, pauses={10: 0.6, 26: 0.6})])
        self.assertGreater(len(cues), 2)
        self.check_rules(cues)
        # every word kept, in order
        self.assertEqual(" ".join(" ".join(c.lines) for c in cues), LONG)

    def test_breaks_at_sentence_ends(self):
        cues = writers.make_cues([segment(LONG, 0.0)])
        sentence_breaks = sum(c.lines[-1].endswith(".") for c in cues)
        self.assertGreaterEqual(sentence_breaks, 3)

    def test_breaks_at_long_pauses(self):
        text = "one two three four five six seven eight nine ten eleven twelve"
        cues = writers.make_cues([segment(text, 0.0, seconds_per_word=0.7, pauses={6: 1.5})])
        self.assertEqual(cues[0].lines[-1].split()[-1], "six")

    def test_short_text_stays_on_one_line(self):
        cues = writers.make_cues([segment("Hello there.", 1.0)])
        self.assertEqual([c.lines for c in cues], [["Hello there."]])

    def test_two_lines_are_balanced(self):
        text = "We should probably head back before it gets too dark outside"  # 61 characters
        lines = writers.make_cues([segment(text, 0.0, seconds_per_word=0.3)])[0].lines
        self.assertEqual(len(lines), 2)
        self.assertLess(abs(len(lines[0]) - len(lines[1])), 15)

    def test_short_subtitles_get_at_least_a_second_when_there_is_room(self):
        cues = writers.make_cues([segment("Yes.", 5.0, seconds_per_word=0.3), segment("No.", 10.0)])
        self.assertAlmostEqual(cues[0].end, 6.0)
        cues = writers.make_cues([segment("Yes.", 5.0, seconds_per_word=0.3), segment("No.", 5.5)])
        self.assertAlmostEqual(cues[0].end, 5.5)  # stretched only up to the next subtitle

    def test_shorter_lines_for_japanese(self):
        text = "今日はとても寒いですね。でも明日は晴れるそうですよ、楽しみにしています。"
        tokens = [text[i:i + 2] for i in range(0, len(text), 2)]  # Japanese words have no spaces
        seg = NS(start=0.0, end=9.0, text=text,
                 words=[NS(start=i * 0.4, end=i * 0.4 + 0.35, word=t) for i, t in enumerate(tokens)])
        cues = writers.make_cues([seg], max_chars=13)
        self.check_rules(cues, max_chars=13)
        self.assertEqual("".join("".join(c.lines) for c in cues), text)

    def test_segments_without_word_timings_are_still_split(self):
        cues = writers.make_cues([segment(LONG, 0.0, timed=False)])
        self.assertGreater(len(cues), 2)
        self.check_rules(cues)


class RenderTests(unittest.TestCase):
    SEGMENTS = [segment("Hello there, friend.", 0.5), segment("Second line", 3661.25)]

    def test_srt(self):
        self.assertEqual(writers.render(self.SEGMENTS, ".srt"),
                         "1\n00:00:00,500 --> 00:00:01,620\nHello there, friend.\n\n"
                         "2\n01:01:01,250 --> 01:01:02,250\nSecond line\n\n")

    def test_vtt(self):
        self.assertTrue(writers.render(self.SEGMENTS, ".vtt").startswith("WEBVTT\n\n1\n00:00:00.500 --> 00:00:01.620\n"))

    def test_two_line_subtitles_are_written_on_two_lines(self):
        text = writers.render([segment("We should probably head back before it gets too dark outside", 0.0, 0.3)])
        self.assertEqual(len(text.split("\n\n")[0].split("\n")), 4)  # number, timing, 2 lines

    def test_one_subtitle_per_segment_when_readable_lines_is_off(self):
        text = writers.render([segment(LONG, 0.0)], ".srt", readable_lines=False)
        self.assertEqual(text.count(" --> "), 1)
        self.assertIn(LONG, text)

    def test_txt_has_no_timings(self):
        self.assertEqual(writers.render(self.SEGMENTS, ".txt"), "Hello there, friend.\nSecond line\n")

    def test_unknown_format(self):
        with self.assertRaises(ValueError):
            writers.render(self.SEGMENTS, ".docx")

    def test_write_saves_utf8(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "out.srt")
            writers.write(path, [segment("Café – 日本語", 0.0)], ".srt")
            with open(path, encoding="utf-8") as f:
                self.assertIn("Café – 日本語", f.read())


if __name__ == "__main__":
    unittest.main()
