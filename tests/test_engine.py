import inspect
import os
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import faster_whisper

import engine

SEGMENTS = [NS(start=0.5, end=2.5, text=" Hello there.", words=None),
            NS(start=3.0, end=6.0, text=" Second line.", words=None)]
OUT_OF_MEMORY = RuntimeError("CUDA failed with error out of memory")


def info(kwargs, language, probability):
    if kwargs["language"]:
        language, probability = kwargs["language"], 1.0
    return NS(duration=6.0, language=language, language_probability=probability)


class FakeModel:
    """Stands in for faster_whisper.WhisperModel, so tests need no model download or GPU."""
    loads = []
    language, probability, error = "fr", 0.97, None

    def __init__(self, name, device, device_index, compute_type):
        FakeModel.loads.append((name, device, device_index, compute_type))
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if FakeModel.error:
            raise FakeModel.error
        return iter(SEGMENTS), info(kwargs, FakeModel.language, FakeModel.probability)


class FakeBatched:
    """Stands in for faster_whisper.BatchedInferencePipeline (Fast Mode)."""
    calls = []
    errors = {}  # batch size -> error raised while the batches run

    def __init__(self, model):
        self.model = model

    def transcribe(self, path, **kwargs):
        FakeBatched.calls.append(kwargs)

        def segments():
            if kwargs["batch_size"] in FakeBatched.errors:
                raise FakeBatched.errors[kwargs["batch_size"]]
            yield from SEGMENTS
        return segments(), info(kwargs, FakeModel.language, FakeModel.probability)


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        engine._loaded.update(key=None, model=None)
        FakeModel.loads.clear()
        FakeModel.language, FakeModel.probability, FakeModel.error = "fr", 0.97, None
        FakeBatched.calls.clear()
        FakeBatched.errors = {}
        for name, fake in (("WhisperModel", FakeModel), ("BatchedInferencePipeline", FakeBatched)):
            patcher = mock.patch.object(engine, name, fake)
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

    def make_film(self):
        film = os.path.join(self.folder, "film.mkv")
        open(film, "w").close()
        return film, os.path.join(self.folder, "film.srt")

    def normal_calls(self):
        return [kwargs for _, kwargs in engine._loaded["model"].calls]


class ModelLoadingTests(EngineTestCase):
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


class TranscriptionSettingsTests(EngineTestCase):
    def test_options_reach_faster_whisper(self):
        self.run_file(source_language="auto", task="transcribe", vad_filter=False)
        kwargs = self.normal_calls()[-1]
        self.assertIsNone(kwargs["language"])
        self.assertEqual(kwargs["task"], "transcribe")
        self.assertIsNone(kwargs["vad_parameters"])
        self.run_file(source_language="ja", vad_filter=True)
        kwargs = self.normal_calls()[-1]
        self.assertEqual(kwargs["language"], "ja")
        self.assertEqual(kwargs["vad_parameters"], {"min_silence_duration_ms": 500, "speech_pad_ms": 400})

    def test_no_hard_coded_prompt(self):
        self.run_file()
        self.assertNotIn("initial_prompt", self.normal_calls()[-1])
        self.assertIsNone(self.normal_calls()[-1]["hotwords"])

    def test_names_and_terms_become_hotwords(self):
        self.run_file(names_and_terms="  Frodo, Gandalf, Mordor ")
        self.assertEqual(self.normal_calls()[-1]["hotwords"], "Frodo, Gandalf, Mordor")

    def test_settings_against_repeated_lines(self):
        self.run_file(readable_lines=False)
        kwargs = self.normal_calls()[-1]
        self.assertIs(kwargs["condition_on_previous_text"], False)
        self.assertEqual(kwargs["hallucination_silence_threshold"], 2.0)
        self.assertIs(kwargs["word_timestamps"], True)  # the silence check needs word timings

    def test_settings_exist_in_the_installed_faster_whisper(self):
        """Catches a faster-whisper upgrade that renames or drops a setting engine.py uses."""
        self.run_file(names_and_terms="Frodo")
        self.run_file(batch_size=8)
        inspect.signature(faster_whisper.WhisperModel.transcribe).bind(None, "clip.mp3", **self.normal_calls()[-1])
        inspect.signature(faster_whisper.BatchedInferencePipeline.transcribe).bind(None, "clip.mp3", **FakeBatched.calls[-1])


class FastModeTests(EngineTestCase):
    def test_fast_mode_transcribes_in_batches(self):
        self.run_file(batch_size=8, names_and_terms="Frodo")
        kwargs = FakeBatched.calls[0]
        self.assertEqual((kwargs["batch_size"], kwargs["beam_size"], kwargs["without_timestamps"]), (8, 5, False))
        self.assertEqual((kwargs["vad_filter"], kwargs["word_timestamps"], kwargs["hotwords"]), (True, True, "Frodo"))
        self.assertEqual(self.normal_calls(), [])

    def test_fast_mode_needs_vad(self):
        self.run_file(batch_size=8, vad_filter=False)
        self.assertEqual(FakeBatched.calls, [])
        self.assertEqual(self.normal_calls()[0]["beam_size"], 10)

    def test_batch_size_0_or_1_is_normal_mode(self):
        self.run_file(batch_size=0)
        self.run_file(batch_size=1)
        self.assertEqual((len(FakeBatched.calls), len(self.normal_calls())), (0, 2))

    def test_out_of_memory_halves_the_batch(self):
        FakeBatched.errors = {16: OUT_OF_MEMORY, 8: OUT_OF_MEMORY}
        messages = []
        result, output = self.run_file(batch_size=16, progress_callback=lambda c, t, m: messages.append(m))
        self.assertEqual([kwargs["batch_size"] for kwargs in FakeBatched.calls], [16, 8, 4])
        self.assertIn("Out of GPU memory, starting again with smaller batches (8)...", messages)
        self.assertTrue(os.path.exists(output))

    def test_out_of_memory_at_the_smallest_batch_falls_back_to_normal_mode(self):
        FakeBatched.errors = {2: OUT_OF_MEMORY}
        self.run_file(batch_size=2)
        self.assertEqual(len(self.normal_calls()), 1)

    def test_out_of_memory_in_normal_mode_is_reported(self):
        FakeModel.error = OUT_OF_MEMORY
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.run_file()

    def test_other_errors_are_not_retried(self):
        FakeBatched.errors = {8: RuntimeError("Invalid data found when processing input")}
        with self.assertRaisesRegex(RuntimeError, "Invalid data"):
            self.run_file(batch_size=8)
        self.assertEqual(len(FakeBatched.calls), 1)


class ResultTests(EngineTestCase):
    def test_writes_subtitles_and_returns_result(self):
        result, output = self.run_file("out.srt")
        self.assertEqual(result, {"subtitle_file": output, "video_file": None, "notes": [],
                                  "language": "fr", "language_probability": 0.97, "language_detected": True})
        with open(output, encoding="utf-8") as f:
            self.assertTrue(f.read().startswith("1\n00:00:00,500 --> 00:00:02,500\nHello there.\n\n2\n"))

    def test_language_chosen_by_user_is_not_reported_as_detected(self):
        result, _ = self.run_file(source_language="ja")
        self.assertEqual((result["language"], result["language_detected"]), ("ja", False))

    def test_unsure_detection_adds_a_note(self):
        FakeModel.probability = 0.31
        result, _ = self.run_file()
        self.assertIn("Not sure which language this is (best guess French (31%))", result["notes"][0])

    def test_line_length_follows_the_subtitle_language(self):
        FakeModel.language = "ja"
        with mock.patch.object(engine.writers, "write") as write:
            self.run_file(task="transcribe")  # Japanese subtitles
            self.run_file(task="translate")  # English subtitles
            self.run_file(task="transcribe", readable_lines=False)
        self.assertEqual([c.args[3:] for c in write.call_args_list], [(True, 13), (True, 42), (False, 13)])

    def test_progress_messages(self):
        messages = []
        self.run_file(task="translate", progress_callback=lambda cur, total, text: messages.append(text))
        self.assertEqual(messages, ["Loading 'large-v3' model onto AUTO...", "Extracting audio and analyzing...",
                                    "Translating from French... 41.7%", "Translating from French... 100.0%",
                                    "Saving text file...", "Success! File saved."])

    def test_cancel_raises_and_writes_nothing(self):
        with self.assertRaises(engine.ProcessCancelled):
            self.run_file("cancel.srt", cancel_check=lambda: True)
        self.assertFalse(os.path.exists(os.path.join(self.folder, "cancel.srt")))


class VideoOutputTests(EngineTestCase):
    def test_video_skipped_with_a_note(self):
        result, _ = self.run_file("out.txt", video_output="burn")
        self.assertIn(".txt files have no timings", result["notes"][0])
        result, _ = self.run_file("out.srt", video_output="track")  # clip.mp3 is audio
        self.assertIn("only MP4, MKV, AVI and MOV", result["notes"][0])

    def test_burn_in(self):
        film, output = self.make_film()
        with mock.patch.object(engine.video, "burn_subtitles", return_value=("film_hardcoded.mp4", True)) as burn:
            result = engine.process_file(film, output, video_output="burn")
        burn.assert_called_once_with(film, output)
        self.assertEqual((result["video_file"], result["notes"]), ("film_hardcoded.mp4", []))

    def test_burn_in_on_the_cpu_adds_a_note(self):
        film, output = self.make_film()
        with mock.patch.object(engine.video, "burn_subtitles", return_value=("film_hardcoded.mp4", False)):
            result = engine.process_file(film, output, video_output="burn")
        self.assertIn("CPU encoder", result["notes"][0])

    def test_subtitle_track_is_labelled_with_the_subtitle_language(self):
        FakeModel.language = "ja"
        film, output = self.make_film()
        with mock.patch.object(engine.video, "add_subtitle_track", return_value="film_subtitled.mkv") as track:
            result = engine.process_file(film, output, task="transcribe", video_output="track")
            engine.process_file(film, output, task="translate", video_output="track")
        self.assertEqual(result["video_file"], "film_subtitled.mkv")
        self.assertEqual([c.args for c in track.call_args_list], [(film, output, "ja"), (film, output, "en")])

    def test_unknown_video_output(self):
        with self.assertRaises(ValueError):
            self.run_file(video_output="sideways")


if __name__ == "__main__":
    unittest.main()
