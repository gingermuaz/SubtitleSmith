import os
import subprocess
import tempfile
import unittest
from unittest import mock

import burn
from errors import ProcessingError

SRT = "1\n00:00:00,500 --> 00:00:05,000\nHELLO SUBTITLES\n\n"


def make_video(path):
    """A 6-second black video with a tone, made by FFmpeg."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=6",
                    "-f", "lavfi", "-i", "sine=d=6", "-c:v", "libx264", "-c:a", "aac", "-shortest", path],
                   check=True)


def brightest_pixel(video, at_seconds=1.5):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(at_seconds), "-i", video, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return max(raw)


class MissingFFmpegTests(unittest.TestCase):
    def test_missing_ffmpeg_raises(self):
        with mock.patch.object(burn.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ProcessingError, "FFmpeg isn't installed"):
                burn.burn_subtitles("movie.mp4", "movie.srt")

    def test_can_burn(self):
        self.assertTrue(burn.can_burn("C:/x/Film.MKV"))
        self.assertFalse(burn.can_burn("song.mp3"))


@unittest.skipUnless(burn.ffmpeg_available(), "FFmpeg isn't installed")
class BurnTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.folder = self._temp.name
        self.addCleanup(self._temp.cleanup)

    def _subtitles(self, folder, name):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(SRT if name.endswith(".srt") else "WEBVTT\n\n" + SRT.replace(",", "."))
        return path

    def test_different_folder_and_awkward_names(self):
        movies = os.path.join(self.folder, "Movies (2024)")
        os.makedirs(movies)
        video = os.path.join(movies, "Ocean's Eleven, Part [1].mp4")
        make_video(video)
        subs = self._subtitles(os.path.join(self.folder, "Subs, out [B]"), "Ocean's Eleven, Part [1]_subtitle.srt")
        cwd = os.getcwd()

        output = burn.burn_subtitles(video, subs)

        self.assertEqual(output, os.path.join(os.path.dirname(subs), "Ocean's Eleven, Part [1]_hardcoded.mp4"))
        self.assertTrue(os.path.exists(output))
        self.assertEqual(os.getcwd(), cwd)
        self.assertLess(brightest_pixel(video), 40)
        self.assertGreater(brightest_pixel(output), 150, "subtitle text should be drawn on the frame")
        self.assertFalse(any("partial" in name for name in os.listdir(os.path.dirname(subs))))

    def test_vtt(self):
        video = os.path.join(self.folder, "clip.mp4")
        make_video(video)
        output = burn.burn_subtitles(video, self._subtitles(self.folder, "clip.vtt"))
        self.assertGreater(brightest_pixel(output), 150)

    def test_failure_raises_and_keeps_earlier_video(self):
        bad_video = os.path.join(self.folder, "broken.mp4")
        with open(bad_video, "w") as f:
            f.write("not a video")
        earlier = os.path.join(self.folder, "broken_hardcoded.mp4")
        with open(earlier, "w") as f:
            f.write("earlier finished video")

        with self.assertRaisesRegex(ProcessingError, "burning them into the video failed"):
            burn.burn_subtitles(bad_video, self._subtitles(self.folder, "broken.srt"))
        with open(earlier) as f:
            self.assertEqual(f.read(), "earlier finished video")
        self.assertFalse(os.path.exists(os.path.join(self.folder, "broken_hardcoded.partial.mp4")))


if __name__ == "__main__":
    unittest.main()
