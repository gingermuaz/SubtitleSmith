import os
import sys
import time
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import speakers
from errors import ProcessCancelled, ProcessingError


def W(start, end, word):
    return NS(start=start, end=end, word=word)


class AssignTests(unittest.TestCase):
    def test_words_split_where_the_speaker_changes(self):
        segment = NS(start=0.0, end=4.0, text=" Are you ready? Yes, let's go.",
                     words=[W(0.0, 0.4, " Are"), W(0.4, 0.7, " you"), W(0.7, 1.4, " ready?"),
                            W(2.0, 2.6, " Yes,"), W(2.7, 3.2, " let's"), W(3.2, 4.0, " go.")])
        parts = speakers.assign([segment], [(0.0, 1.5, "SPEAKER_04"), (1.9, 4.2, "SPEAKER_01")])
        self.assertEqual([(p.speaker, p.text.strip()) for p in parts], [(1, "Are you ready?"), (2, "Yes, let's go.")])
        self.assertEqual((parts[1].start, parts[1].end), (2.0, 4.0))

    def test_numbered_in_order_of_appearance(self):
        segments = [NS(start=s, end=s + 1, text=f" line {s}", words=None) for s in (0, 2, 4)]
        parts = speakers.assign(segments, [(0, 1, "B"), (2, 3, "A"), (4, 5, "B")])
        self.assertEqual([p.speaker for p in parts], [1, 2, 1])

    def test_one_word_wobble_is_ignored(self):
        segment = NS(start=0.0, end=2.0, text=" one two three",
                     words=[W(0.0, 0.5, " one"), W(0.6, 0.9, " two"), W(1.0, 2.0, " three")])
        parts = speakers.assign([segment], [(0.0, 0.55, "A"), (0.55, 0.95, "B"), (0.95, 2.0, "A")])
        self.assertEqual([(p.speaker, p.text.strip()) for p in parts], [(1, "one two three")])

    def test_word_in_a_gap_goes_to_the_nearest_speaker(self):
        segment = NS(start=0.0, end=3.0, text=" hi there",
                     words=[W(0.0, 0.5, " hi"), W(2.4, 3.0, " there")])
        parts = speakers.assign([segment], [(0.0, 0.6, "A"), (2.5, 3.0, "B")])
        self.assertEqual([p.speaker for p in parts], [1, 2])

    def test_no_speakers_found(self):
        segments = [NS(start=0, end=1, text=" hi", words=None)]
        self.assertEqual(speakers.assign(segments, []), segments)


class LabelTests(unittest.TestCase):
    def test_label_only_when_the_speaker_changes(self):
        segments = [NS(start=i, end=i + 1, text=f" line {i}", words=None, speaker=s) for i, s in enumerate([1, 1, 2, 1])]
        self.assertEqual([s.text for s in speakers.with_labels(segments)],
                         [" Speaker 1: line 0", " line 1", " Speaker 2: line 2", " Speaker 1: line 3"])

    def test_label_is_part_of_the_first_word(self):
        segment = NS(start=0, end=1, text=" Hi there", words=[W(0, 0.5, " Hi"), W(0.5, 1, " there")], speaker=3)
        labelled = speakers.with_labels([segment])[0]
        self.assertEqual([w.word for w in labelled.words], [" Speaker 3: Hi", " there"])


class SetupTests(unittest.TestCase):
    def setUp(self):
        speakers._loaded.update(gpu=None, pipeline=None)

    def test_usage_reporting_is_off(self):
        self.assertEqual(os.environ.get("PYANNOTE_METRICS_ENABLED"), "false")

    def test_missing_packages(self):
        with mock.patch.object(speakers, "available", return_value=False):
            with self.assertRaisesRegex(ProcessingError, "requirements-speakers.txt"):
                speakers.get_pipeline()

    def test_missing_token(self):
        with mock.patch.object(speakers, "available", return_value=True), \
                mock.patch.object(speakers.huggingface_hub, "get_token", return_value=None):
            with self.assertRaisesRegex(ProcessingError, "access token"):
                speakers.get_pipeline()

    def fake_pyannote(self, pipeline):
        torch = types.ModuleType("torch")
        torch.cuda = NS(is_available=lambda: False)
        torch.from_numpy = lambda array: NS(unsqueeze=lambda dim: array)
        audio = types.ModuleType("pyannote.audio")
        audio.Pipeline = NS(from_pretrained=lambda model, token: pipeline)
        return mock.patch.dict(sys.modules, {"torch": torch, "pyannote.audio": audio})

    def test_terms_not_accepted(self):
        with self.fake_pyannote(None), mock.patch.object(speakers, "available", return_value=True), \
                mock.patch.object(speakers.huggingface_hub, "get_token", return_value="hf_x"):
            with self.assertRaisesRegex(ProcessingError, "accepted the model's terms"):
                speakers.get_pipeline()

    def test_find_speakers_and_cancel(self):
        turn = lambda s, e: NS(start=s, end=e)
        annotation = NS(itertracks=lambda yield_label: [(turn(0, 1), "_", "A"), (turn(1, 2), "_", "B")])

        def pipeline(audio, hook, num_speakers):
            for i in range(3):
                hook("embeddings", None, total=3, completed=i)
                time.sleep(0.05)
            return NS(exclusive_speaker_diarization=annotation)
        with self.fake_pyannote(pipeline), mock.patch.object(speakers, "available", return_value=True), \
                mock.patch.object(speakers.huggingface_hub, "get_token", return_value="hf_x"), \
                mock.patch("faster_whisper.decode_audio", return_value="AUDIO"):
            progress = []
            turns = speakers.find_speakers("film.mp4", progress_callback=lambda *a: progress.append(a[2]))
            self.assertEqual(turns, [(0, 1, "A"), (1, 2, "B")])
            self.assertIn("Finding speakers... 67%", progress)
            with self.assertRaises(ProcessCancelled):
                speakers.find_speakers("film.mp4", cancel_check=lambda: True)


if __name__ == "__main__":
    unittest.main()
