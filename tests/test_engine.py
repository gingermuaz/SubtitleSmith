import inspect
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import faster_whisper
from huggingface_hub.errors import LocalEntryNotFoundError

import engine
from errors import ProcessingError

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
    load_seconds = transcribe_seconds = 0

    def __init__(self, name, device, device_index, compute_type):
        time.sleep(FakeModel.load_seconds)
        FakeModel.loads.append((name, device, device_index, compute_type))
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append((path, kwargs))
        time.sleep(FakeModel.transcribe_seconds)
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
        FakeModel.load_seconds = FakeModel.transcribe_seconds = 0
        FakeBatched.calls.clear()
        FakeBatched.errors = {}
        for name, fake in (("WhisperModel", FakeModel), ("BatchedInferencePipeline", FakeBatched),
                           ("model_folder", lambda size, *args: f"folder:{engine.MODEL_REPOS.get(size, size)}")):
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
        self.assertEqual([load[:2] for load in FakeModel.loads],
                         [("folder:Systran/faster-whisper-small", "auto"), ("folder:Systran/faster-whisper-medium", "auto"),
                          ("folder:Systran/faster-whisper-medium", "cpu")])

    def test_compute_types_and_repo_names(self):
        engine.get_model("large-v3-turbo", "auto", device_index=1)
        engine.get_model("tiny", "cpu")
        self.assertEqual(FakeModel.loads[0], ("folder:deepdml/faster-whisper-large-v3-turbo-ct2", "auto", 1, "int8_float16"))
        self.assertEqual(FakeModel.loads[1], ("folder:Systran/faster-whisper-tiny", "cpu", 0, "int8"))


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
        result, output = self.run_file(batch_size=16, progress_callback=lambda *args: messages.append(args[2]))
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
        self.assertEqual(result, {"subtitle_file": output, "extra_files": [], "extra_languages": [], "video_file": None,
                                  "notes": [],
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
        with mock.patch.object(engine.writers, "render", return_value="") as render:
            self.run_file(task="transcribe")  # Japanese subtitles
            self.run_file(task="translate")  # English subtitles
            self.run_file(task="transcribe", readable_lines=False)
        self.assertEqual([c.args[2:] for c in render.call_args_list], [(True, 13), (True, 42), (False, 13)])

    def test_progress_messages_and_stages(self):
        messages = []
        self.run_file(task="translate", progress_callback=lambda *args: messages.append((args[2], args[3])))
        self.assertEqual(messages, [("Loading 'large-v3' model onto AUTO...", "load"),
                                    ("Extracting audio and analyzing...", "prepare"),
                                    ("Translating from French... 41.7%", "transcribe"),
                                    ("Translating from French... 100.0%", "transcribe"),
                                    ("Saving text file...", "save"), ("Success! File saved.", "done")])

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
        self.assertEqual(burn.call_args.args, (film, output))
        self.assertEqual((result["video_file"], result["notes"]), ("film_hardcoded.mp4", []))

    def test_burn_in_reports_progress_and_can_be_cancelled(self):
        film, output = self.make_film()
        messages, cancel = [], lambda: False

        def fake_burn(media, subs, cancel_check, on_progress, style):
            on_progress(3.0)  # halfway through the 6-second film
            self.assertIs(cancel_check, cancel)
            return "film_hardcoded.mp4", True
        with mock.patch.object(engine.video, "burn_subtitles", side_effect=fake_burn):
            engine.process_file(film, output, video_output="burn", cancel_check=cancel,
                                progress_callback=lambda *args: messages.append((args[2], args[3])))
        self.assertIn(("Burning subtitles to video... 50%", "video"), messages)

    def test_burn_in_on_the_cpu_adds_a_note(self):
        film, output = self.make_film()
        with mock.patch.object(engine.video, "burn_subtitles", return_value=("film_hardcoded.mp4", False)):
            result = engine.process_file(film, output, video_output="burn")
        self.assertIn("CPU encoder", result["notes"][0])

    def test_subtitle_track_is_labelled_with_the_subtitle_language(self):
        FakeModel.language = "ja"
        film, output = self.make_film()
        with mock.patch.object(engine.video, "add_subtitle_tracks", return_value="film_subtitled.mkv") as track:
            result = engine.process_file(film, output, task="transcribe", video_output="track")
            engine.process_file(film, output, task="translate", video_output="track")
        self.assertEqual(result["video_file"], "film_subtitled.mkv")
        self.assertEqual([c.args for c in track.call_args_list], [(film, [(output, "ja")]), (film, [(output, "en")])])

    def test_unknown_video_output(self):
        with self.assertRaises(ValueError):
            self.run_file(video_output="sideways")


class CancelTests(EngineTestCase):
    def test_cancel_while_the_model_loads_does_not_wait_for_it(self):
        FakeModel.load_seconds = 1.0
        started = time.monotonic()
        with self.assertRaises(engine.ProcessCancelled):
            engine.get_model("small", cancel_check=lambda: time.monotonic() - started > 0.1)
        self.assertLess(time.monotonic() - started, 0.6)
        # the load carries on in the background and is kept, so next time is instant
        time.sleep(1.2)
        self.assertEqual(len(FakeModel.loads), 1)
        engine.get_model("small")
        self.assertEqual(len(FakeModel.loads), 1)

    def test_cancel_while_the_audio_is_read(self):
        FakeModel.transcribe_seconds = 1.0
        engine.get_model("large-v3")
        started = time.monotonic()
        with self.assertRaises(engine.ProcessCancelled):
            self.run_file(cancel_check=lambda: time.monotonic() - started > 0.1)
        self.assertLess(time.monotonic() - started, 0.6)


class FakeHub:
    """Stands in for huggingface_hub: one model repository whose files 'download' through the given progress bar."""

    def __init__(self, folder, cached=False, offline=False, fail_on=None):
        self.folder, self.cached, self.offline, self.fail_on = folder, cached, offline, fail_on
        self.files = {"config.json": 10, "model.bin": 3000, "tokenizer.json": 20, "vocabulary.txt": 5, "README.md": 7}
        self.downloads, self.api_calls = [], 0

    def snapshot_download(self, repo_id, allow_patterns, local_files_only):
        if not self.cached:
            raise LocalEntryNotFoundError("not downloaded")
        return self.folder

    def HfApi(self):
        def model_info(repo_id, files_metadata):
            self.api_calls += 1
            if self.offline:
                raise ConnectionError("no route to host")
            return NS(siblings=[NS(rfilename=name, size=size) for name, size in self.files.items()])
        return NS(model_info=model_info)

    def hf_hub_download(self, repo_id, filename, tqdm_class):
        if filename == self.fail_on:
            raise OSError("connection reset")
        size = self.files[filename]
        with tqdm_class(total=size, initial=0, desc=filename, unit="B") as bar:
            for start in range(0, size, 1000):
                bar.update(min(1000, size - start))
        self.downloads.append(filename)
        return os.path.join(self.folder, filename)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.folder = self._temp.name

    def download(self, hub, cancel_check=None):
        events = []
        with mock.patch.object(engine, "huggingface_hub", hub):
            result = engine.model_folder("large-v3", lambda *args: events.append(args), cancel_check)
        return result, events

    def test_already_downloaded_model_needs_no_network(self):
        open(os.path.join(self.folder, "model.bin"), "w").close()
        hub = FakeHub(self.folder, cached=True)
        folder, events = self.download(hub)
        self.assertEqual((folder, hub.downloads, hub.api_calls, events), (self.folder, [], 0, []))

    def test_downloads_the_needed_files_with_progress(self):
        hub = FakeHub(self.folder)
        folder, events = self.download(hub)
        self.assertEqual(folder, self.folder)
        self.assertEqual(hub.downloads, ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"])
        self.assertEqual({event[3] for event in events}, {"download"})
        self.assertEqual(events[-1][:2], (3035, 3035))
        self.assertIn("Downloading the 'large-v3' AI model (first use only)", events[-1][2])
        self.assertTrue(all(a[0] <= b[0] for a, b in zip(events, events[1:])), "progress only goes up")

    def test_partly_downloaded_model_is_completed(self):
        hub = FakeHub(self.folder, cached=True)  # cache entry exists but model.bin is missing
        self.download(hub)
        self.assertIn("model.bin", hub.downloads)

    def test_cancel_stops_the_download(self):
        hub = FakeHub(self.folder)
        with self.assertRaises(engine.ProcessCancelled):
            self.download(hub, cancel_check=lambda: len(hub.downloads) >= 1)  # cancel during model.bin
        self.assertEqual(hub.downloads, ["config.json"])

    def test_no_internet(self):
        with self.assertRaisesRegex(ProcessingError, "needs an internet connection"):
            self.download(FakeHub(self.folder, offline=True))

    def test_download_failing_part_way(self):
        with self.assertRaisesRegex(ProcessingError, "carries on from where it stopped"):
            self.download(FakeHub(self.folder, fail_on="model.bin"))


class TranslatorLoadingTests(unittest.TestCase):
    def setUp(self):
        self.downloads, self.loaded = [], []
        test = self

        class FakeNllb:
            def __init__(self, folder, device, device_index):
                test.loaded.append((folder, device, device_index))

        def download(repo_id, files, what, progress_callback=None, cancel_check=None):
            self.downloads.append((repo_id, what))
            return "folder:" + repo_id
        for patcher in (mock.patch.object(engine, "download_model", download),
                        mock.patch.object(engine.translation, "Translator", FakeNllb),
                        mock.patch.dict(engine._translator, {"key": None, "model": None})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_each_size_has_its_own_model(self):
        first = engine.get_translator("better", "cuda", 1)
        self.assertIs(engine.get_translator("better", "cuda", 1), first)  # loaded once, then reused
        self.assertEqual(self.downloads, [("OpenNMT/nllb-200-distilled-1.3B-ct2-int8",
                                           "the 'Better' translation model")])
        self.assertEqual(self.loaded, [("folder:OpenNMT/nllb-200-distilled-1.3B-ct2-int8", "cuda", 1)])
        best = engine.get_translator("best", "cuda", 1)
        self.assertIsNot(best, first)
        self.assertEqual(self.downloads[-1][0], "OpenNMT/nllb-200-3.3B-ct2-int8")
        self.assertIs(engine._translator["model"], best)  # only one translation model kept in memory
        engine.get_translator("fast", "cpu")
        self.assertEqual(self.downloads[-1][0], "skywood/nllb-200-distilled-600M-ct2-int8")

    def test_unknown_size(self):
        with self.assertRaises(ValueError):
            engine.get_translator("huge")


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg isn't installed")
class MediaDurationTests(unittest.TestCase):
    def test_reads_the_length(self):
        with tempfile.TemporaryDirectory() as folder:
            clip = os.path.join(folder, "clip.mp3")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=d=3", clip], check=True)
            self.assertAlmostEqual(engine.media_duration(clip), 3.0, delta=0.1)
            self.assertIsNone(engine.media_duration(os.path.join(folder, "missing.mp4")))


class FakeTranslator:
    calls = []

    def translate(self, texts, source, target, cancel_check=None, on_progress=None):
        FakeTranslator.calls.append((source, target))
        if on_progress:
            on_progress(len(texts), len(texts))
        return [f"[{target}] {text.strip()}" for text in texts]


class TranslationAndExtrasTests(EngineTestCase):
    def setUp(self):
        super().setUp()
        FakeTranslator.calls.clear()
        patcher = mock.patch.object(engine, "get_translator", return_value=FakeTranslator())
        patcher.start()
        self.addCleanup(patcher.stop)

    def read(self, path):
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_original_plus_english(self):
        result, output = self.run_file(task="transcribe", also_english=True)
        english = os.path.join(self.folder, "out.en.srt")
        self.assertEqual(result["extra_files"], [english])
        self.assertEqual(FakeTranslator.calls, [("fr", "en")])
        self.assertIn("Hello there.", self.read(output))
        self.assertIn("[en] Hello there.", self.read(english))

    def test_translate_to_another_language_as_well(self):
        result, output = self.run_file(task="transcribe", also_english=True, translate_to="bn")
        self.assertEqual([os.path.basename(p) for p in result["extra_files"]], ["out.en.srt", "out.bn.srt"])
        self.assertEqual(FakeTranslator.calls, [("fr", "en"), ("fr", "bn")])  # both from the original

    def test_english_main_subtitles_translated_onwards(self):
        self.run_file(task="translate", translate_to="es")
        self.assertEqual(FakeTranslator.calls, [("en", "es")])

    def test_no_duplicate_when_already_in_that_language(self):
        FakeModel.language = "en"
        result, _ = self.run_file(task="transcribe", also_english=True)
        self.assertEqual((result["extra_files"], FakeTranslator.calls), ([], []))
        self.assertIn("already in English", result["notes"][0])

    def test_translation_problem_is_a_note_not_a_failure(self):
        with mock.patch.object(FakeTranslator, "translate", side_effect=ProcessingError("can't translate Latin")):
            result, output = self.run_file(task="transcribe", translate_to="la")
        self.assertTrue(os.path.exists(output))
        self.assertIn("No Latin subtitles: can't translate Latin", result["notes"][0])

    def test_translated_subtitles_use_that_language_line_length(self):
        with mock.patch.object(engine.writers, "render", return_value="") as render:
            self.run_file(task="transcribe", translate_to="ja")
        self.assertEqual([c.args[3] for c in render.call_args_list], [42, 13])

    def test_every_language_becomes_a_video_track(self):
        film, output = self.make_film()
        with mock.patch.object(engine.video, "add_subtitle_tracks", return_value="x") as tracks:
            engine.process_file(film, output, task="transcribe", also_english=True, video_output="track")
        self.assertEqual(tracks.call_args.args[1], [(output, "fr"), (os.path.join(self.folder, "film.en.srt"), "en")])

    def test_review_can_edit_before_saving(self):
        seen = {}

        def review(texts):
            seen.update(texts)
            return {path: text.replace("Hello", "Howdy") for path, text in texts.items()}
        result, output = self.run_file(task="transcribe", also_english=True, review=review)
        self.assertEqual(sorted(map(os.path.basename, seen)), ["out.en.srt", "out.srt"])
        self.assertIn("Howdy there.", self.read(output))
        self.assertIn("[en] Howdy there.", self.read(result["extra_files"][0]))

    def test_review_cancelled_saves_nothing(self):
        with self.assertRaises(engine.ProcessCancelled):
            self.run_file("never.srt", review=lambda texts: None)
        self.assertFalse(os.path.exists(os.path.join(self.folder, "never.srt")))

    def test_speaker_labels(self):
        turns = [(0.0, 2.6, "SPEAKER_07"), (2.8, 6.0, "SPEAKER_02")]
        with mock.patch.object(engine.speakers, "find_speakers", return_value=turns) as find:
            result, output = self.run_file(task="transcribe", speaker_labels=True, num_speakers=2, also_english=True)
        self.assertEqual(find.call_args.args[2], 2)
        self.assertIn("Speaker 1: Hello there.", self.read(output))
        self.assertIn("Speaker 2: Second line.", self.read(output))
        self.assertIn("Speaker 2: [en] Second line.", self.read(result["extra_files"][0]))

    def test_speaker_problem_is_a_note(self):
        with mock.patch.object(engine.speakers, "find_speakers", side_effect=ProcessingError("token missing")):
            result, output = self.run_file(speaker_labels=True)
        self.assertIn("No speaker labels: token missing", result["notes"][0])
        self.assertNotIn("Speaker", self.read(output))

    def test_burn_style_is_passed_on(self):
        film, output = self.make_film()
        style = {"size": "Large", "colour": "Yellow", "position": "Top"}
        with mock.patch.object(engine.video, "burn_subtitles", return_value=("x", True)) as burn:
            engine.process_file(film, output, video_output="burn", burn_style=style)
        self.assertEqual(burn.call_args.kwargs["style"], style)

    def test_translator_size_is_passed_on(self):
        self.run_file(task="transcribe", translate_to="bn", translator="best")
        self.assertEqual(engine.get_translator.call_args.args[0], "best")
        self.run_file(task="transcribe", also_english=True)
        self.assertEqual(engine.get_translator.call_args.args[0], "better")  # the default

    def test_prepare_loads_only_what_is_needed(self):
        with mock.patch.object(engine.speakers, "get_pipeline") as pipeline:
            engine.prepare("small")
            self.assertEqual(len(engine.get_translator.call_args_list), 0)
            engine.prepare("small", translate=True, speaker_labels=True, compute_device="cpu")
        self.assertEqual(len(engine.get_translator.call_args_list), 1)
        self.assertIs(pipeline.call_args.args[0], False)  # CPU chosen: no GPU for speakers either
        engine.prepare("small", translate=True, translator="fast")
        self.assertEqual(engine.get_translator.call_args.args[:2], ("fast", "auto"))

    def test_extra_output_path(self):
        self.assertEqual(engine.extra_output_path("C:/a/film_subtitle.srt", "en"), "C:/a/film_subtitle.en.srt")


if __name__ == "__main__":
    unittest.main()
