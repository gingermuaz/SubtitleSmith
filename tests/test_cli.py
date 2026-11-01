import io
import os
import tempfile
import unittest
from unittest import mock

import cli
from errors import ProcessingError


class FakeEngine:
    """Records what the command line asks the engine to do."""

    def __init__(self, fail_on=()):
        self.fail_on, self.calls, self.prepared = fail_on, [], None

    def prepare(self, *args, **kwargs):
        self.prepared = (args, kwargs)

    def process_file(self, media, output, progress_callback=None, **job):
        self.calls.append((media, output, job))
        if progress_callback:
            progress_callback(5, 10, "Translating from French... 50.0%", "transcribe")
        if os.path.basename(media) in self.fail_on:
            raise ProcessingError("Invalid data found when processing input")
        extras = [output.replace(".srt", ".en.srt")] if job["also_english"] else []
        return {"subtitle_file": output, "extra_files": extras, "extra_languages": ["en"] if extras else [],
                "video_file": None, "notes": [], "language": "fr", "language_probability": 0.97,
                "language_detected": True}


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.folder = self._temp.name
        for name in ("a.mp4", "b.mkv", "notes.txt", os.path.join("season 2", "c.mp3")):
            os.makedirs(os.path.dirname(os.path.join(self.folder, name)), exist_ok=True)
            open(os.path.join(self.folder, name), "w").close()

    def run_cli(self, *argv, gpus=(), fail_on=(), ffmpeg=True):
        fake = FakeEngine(fail_on)
        out = io.StringIO()
        with mock.patch.object(cli, "engine", mock.Mock(wraps=fake, SUPPORTED_MEDIA=cli.engine.SUPPORTED_MEDIA,
                                                        MODEL_REPOS=cli.engine.MODEL_REPOS)), \
                mock.patch.object(cli.hardware, "get_nvidia_gpus", return_value=list(gpus)), \
                mock.patch.object(cli.video, "ffmpeg_available", return_value=ffmpeg):
            code = cli.main(list(argv), stream=out)
        return code, out.getvalue(), fake


class FindMediaTests(CliTestCase):
    def test_folders_files_and_missing(self):
        files, missing = cli.find_media([self.folder, os.path.join(self.folder, "nope.mp4")])
        self.assertEqual([os.path.basename(f) for f in files], ["a.mp4", "b.mkv"])
        self.assertEqual(missing, [os.path.join(self.folder, "nope.mp4")])

    def test_recursive(self):
        files, _ = cli.find_media([self.folder], recursive=True)
        self.assertEqual(sorted(os.path.basename(f) for f in files), ["a.mp4", "b.mkv", "c.mp3"])


class MainTests(CliTestCase):
    def test_success(self):
        code, out, fake = self.run_cli(self.folder)
        self.assertEqual(code, 0)
        self.assertIn("OK      a.mp4", out)
        self.assertIn("Done: 2 succeeded, 0 failed.", out)
        self.assertEqual(fake.calls[0][1], os.path.join(self.folder, "a_subtitle.srt"))

    def test_a_failed_file_gives_exit_code_1_and_the_rest_carry_on(self):
        code, out, fake = self.run_cli(self.folder, fail_on=("a.mp4",))
        self.assertEqual(code, 1)
        self.assertIn("FAILED  a.mp4: Invalid data", out)
        self.assertEqual(len(fake.calls), 2)

    def test_defaults_on_a_gpu(self):
        _, out, fake = self.run_cli(self.folder, gpus=[(0, "GPU A", 4096), (1, "GPU B", 8192)])
        job = fake.calls[0][2]
        self.assertEqual((job["model_size"], job["device_index"], job["batch_size"], job["compute_device"]),
                         ("large-v3", 1, 8, "auto"))
        self.assertIn("model 'large-v3' on GPU", out)

    def test_options_reach_the_engine(self):
        out_dir = os.path.join(self.folder, "subs")
        _, _, fake = self.run_cli(os.path.join(self.folder, "a.mp4"), "--also-english", "--translate-to", "bn",
                                  "--format", "vtt", "--output-dir", out_dir, "--video", "burn", "--burn-colour",
                                  "Yellow", "--names", "Frodo", "--language", "fr", "--cpu", "--no-readable-lines")
        media, output, job = fake.calls[0]
        self.assertEqual(output, os.path.join(out_dir, "a_subtitle.vtt"))
        self.assertTrue(os.path.isdir(out_dir))
        self.assertEqual((job["task"], job["also_english"], job["translate_to"], job["source_language"]),
                         ("transcribe", True, "bn", "fr"))
        self.assertEqual((job["compute_device"], job["batch_size"], job["readable_lines"]), ("cpu", 0, False))
        self.assertEqual((job["video_output"], job["burn_style"]["colour"], job["names_and_terms"]), ("burn", "Yellow", "Frodo"))
        self.assertEqual(fake.prepared[1]["translate"], True)
        ram_gb = round(cli.psutil.virtual_memory().total / 1024 ** 3)
        expected = cli.hardware.recommend_translator(False, 0, ram_gb)  # --cpu: picked for the CPU
        self.assertEqual((job["translator"], fake.prepared[1]["translator"]), (expected, expected))

    def test_translator_choice(self):
        film = os.path.join(self.folder, "a.mp4")
        _, out, fake = self.run_cli(film, "--translate-to", "fr", gpus=[(0, "RTX", 12288)])
        self.assertEqual(fake.calls[0][2]["translator"], "best")  # a roomy GPU gets the biggest
        self.assertIn("and the 'best' translation model", out)
        _, out, fake = self.run_cli(film, "--translate-to", "fr", "--translator", "fast", gpus=[(0, "RTX", 12288)])
        self.assertEqual((fake.calls[0][2]["translator"], fake.prepared[1]["translator"]), ("fast", "fast"))
        _, out, _ = self.run_cli(film)
        self.assertNotIn("translation model", out)  # not translating

    def test_nothing_to_do(self):
        code, out, _ = self.run_cli(os.path.join(self.folder, "notes.txt.missing"))
        self.assertEqual(code, 2)
        self.assertIn("No media files", out)

    def test_video_needs_ffmpeg(self):
        code, out, _ = self.run_cli(self.folder, "--video", "track", ffmpeg=False)
        self.assertEqual(code, 2)
        self.assertIn("needs FFmpeg", out)

    def test_speakers_need_pyannote(self):
        with mock.patch.object(cli.speakers, "available", return_value=False):
            code, out, _ = self.run_cli(self.folder, "--speakers")
        self.assertEqual(code, 2)
        self.assertIn("requirements-speakers.txt", out)

    def test_ctrl_c(self):
        with mock.patch.object(FakeEngine, "process_file", side_effect=KeyboardInterrupt), \
                mock.patch.object(cli.video, "stop_all") as stop_all:
            code, out, _ = self.run_cli(self.folder)
        self.assertEqual(code, 130)
        self.assertTrue(stop_all.called)


if __name__ == "__main__":
    unittest.main()
