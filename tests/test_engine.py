import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import engine

SEGMENTS = [NS(start=0.5, end=2.5, text=" Hello there.", words=None),
            NS(start=3.0, end=6.0, text=" Second line.", words=None)]


class FakeModel:
    """Stands in for faster_whisper.WhisperModel, so tests need no model download or GPU."""
    loads = []

    def __init__(self, name, device, device_index, compute_type):
        FakeModel.loads.append((name, device, device_index, compute_type))
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return iter(SEGMENTS), NS(duration=6.0)


class EngineTests(unittest.TestCase):
    def setUp(self):
        engine._loaded.update(key=None, model=None)
        FakeModel.loads.clear()
        patcher = mock.patch.object(engine, "WhisperModel", FakeModel)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.folder = self._temp.name
        self.audio = os.path.join(self.folder, "clip.mp3")
        open(self.audio, "w").close()

    def run_file(self, name="out.srt", **kwargs):
        output = os.path.join(self.folder, name)
        return engine.process_file(self.audio, output, output_format=os.path.splitext(name)[1], **kwargs), output

    # --- model loading ---
    def test_model_loaded_once_and_reused(self):
        self.run_file("a.srt")
        self.run_file("b.srt")
        self.assertEqual(len(FakeModel.loads), 1)

    def test_reloads_when_model_or_device_changes(self):
        engine.get_model("small")
        engine.get_model("medium")
        engine.get_model("medium", "cpu")
        self.assertEqual([load[:2] for load in FakeModel.loads], [("small", "auto"), ("medium", "auto"), ("medium", "cpu")])

    def test_compute_types_and_repo_names(self):
        engine.get_model("large-v3-turbo", "auto", device_index=1)
        engine.get_model("tiny", "cpu")
        self.assertEqual(FakeModel.loads[0], ("deepdml/faster-whisper-large-v3-turbo-ct2", "auto", 1, "int8_float16"))
        self.assertEqual(FakeModel.loads[1], ("tiny", "cpu", 0, "int8"))

    # --- transcription ---
    def test_writes_subtitles_and_returns_result(self):
        result, output = self.run_file("out.srt")
        self.assertEqual(result, {"subtitle_file": output, "video_file": None, "notes": []})
        with open(output, encoding="utf-8") as f:
            self.assertTrue(f.read().startswith("1\n00:00:00,500 --> 00:00:02,500\nHello there.\n\n2\n"))

    def test_options_reach_faster_whisper(self):
        self.run_file(source_language="auto", task="transcribe", vad_filter=False)
        path, kwargs = engine._loaded["model"].calls[0]
        self.assertEqual(path, self.audio)
        self.assertIsNone(kwargs["language"])
        self.assertEqual(kwargs["task"], "transcribe")
        self.assertIsNone(kwargs["vad_parameters"])
        self.run_file(source_language="ja", vad_filter=True)
        kwargs = engine._loaded["model"].calls[1][1]
        self.assertEqual(kwargs["language"], "ja")
        self.assertEqual(kwargs["vad_parameters"], {"min_silence_duration_ms": 500, "speech_pad_ms": 400})

    def test_progress_messages(self):
        messages = []
        self.run_file(task="translate", progress_callback=lambda cur, total, text: messages.append(text))
        self.assertEqual(messages, ["Loading 'large-v3' model onto AUTO...", "Extracting audio and analyzing...",
                                    "Translating... 41.7%", "Translating... 100.0%", "Saving text file...",
                                    "Success! File saved."])

    def test_cancel_raises_and_writes_nothing(self):
        with self.assertRaises(engine.ProcessCancelled):
            self.run_file("cancel.srt", cancel_check=lambda: True)
        self.assertFalse(os.path.exists(os.path.join(self.folder, "cancel.srt")))

    # --- burn-in ---
    def test_burn_in_skipped_with_a_note(self):
        result, _ = self.run_file("out.txt", hardcode_subtitles=True)
        self.assertIn(".txt files have no timings", result["notes"][0])
        result, _ = self.run_file("out.srt", hardcode_subtitles=True)  # clip.mp3 is audio
        self.assertIn("only MP4, MKV, AVI and MOV", result["notes"][0])

    def test_burn_in_runs_for_videos(self):
        video = os.path.join(self.folder, "film.mkv")
        open(video, "w").close()
        output = os.path.join(self.folder, "film.srt")
        with mock.patch.object(engine.burn, "burn_subtitles", return_value="film_hardcoded.mp4") as burn_mock:
            result = engine.process_file(video, output, hardcode_subtitles=True)
        burn_mock.assert_called_once_with(video, output)
        self.assertEqual(result["video_file"], "film_hardcoded.mp4")


if __name__ == "__main__":
    unittest.main()
