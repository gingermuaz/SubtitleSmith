import json
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import video
from errors import ProcessingError

SRT = "1\n00:00:00,500 --> 00:00:05,000\nHELLO SUBTITLES\n\n"
HAVE_FFMPEG = video.ffmpeg_available() and shutil.which("ffprobe") is not None


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def make_video(path, video_codec="libx264", audio_codec="aac"):
    """A 6-second black video with a tone."""
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=320x240:d=6", "-f", "lavfi", "-i", "sine=d=6",
           "-c:v", video_codec, "-c:a", audio_codec, "-shortest", path)


def brightest_pixel(path, at_seconds=1.5):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(at_seconds), "-i", path, "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    return max(raw)


def subtitle_tracks(path):
    """[(format, language, on by default), ...] for each subtitle track."""
    streams = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", path],
                                        capture_output=True, check=True).stdout)["streams"]
    return [(s["codec_name"], s.get("tags", {}).get("language"), s["disposition"]["default"])
            for s in streams if s["codec_type"] == "subtitle"]


def video_stream_md5(path):
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-map", "0:v", "-c", "copy", "-f", "md5", "-"],
                          capture_output=True, check=True).stdout


def write_subtitles(path, vtt=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n" + SRT.replace(",", ".") if vtt else SRT)
    return path


class ChoosingTheEncoderTests(unittest.TestCase):
    """Which encoder burn-in uses, with FFmpeg's answers simulated."""

    def burn(self, nvenc_listed, *exit_codes):
        results = [subprocess.CompletedProcess([], code, b"", b"No NVENC capable devices found") for code in exit_codes]
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(video, "ffmpeg_available", return_value=True), \
                mock.patch.object(video, "nvenc_listed", return_value=nvenc_listed), \
                mock.patch.object(video, "_run_ffmpeg", side_effect=results) as run, \
                mock.patch.object(video.os, "replace"):
            _, used_gpu = video.burn_subtitles(os.path.join(folder, "film.mp4"),
                                               write_subtitles(os.path.join(folder, "film.srt")))
        return used_gpu, [call.args[0] for call in run.call_args_list]

    def test_nvenc_is_used_when_it_works(self):
        used_gpu, commands = self.burn(True, 0)
        self.assertTrue(used_gpu)
        self.assertEqual(len(commands), 1)
        self.assertIn("h264_nvenc", commands[0])

    def test_cpu_encoder_when_nvenc_fails(self):
        used_gpu, commands = self.burn(True, 1, 0)
        self.assertFalse(used_gpu)
        self.assertEqual(len(commands), 2)
        self.assertNotIn("h264_nvenc", commands[1])

    def test_cpu_encoder_straight_away_when_ffmpeg_has_no_nvenc(self):
        used_gpu, commands = self.burn(False, 0)
        self.assertFalse(used_gpu)
        self.assertEqual(len(commands), 1)
        self.assertNotIn("h264_nvenc", commands[0])


class WithoutFFmpegTests(unittest.TestCase):
    def test_missing_ffmpeg_raises(self):
        with mock.patch.object(video.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ProcessingError, "FFmpeg isn't installed"):
                video.burn_subtitles("movie.mp4", "movie.srt")
            with self.assertRaisesRegex(ProcessingError, "FFmpeg isn't installed"):
                video.add_subtitle_track("movie.mp4", "movie.srt")

    def test_is_supported_video(self):
        self.assertTrue(video.is_supported_video("C:/x/Film.MKV"))
        self.assertFalse(video.is_supported_video("song.mp3"))


@unittest.skipUnless(HAVE_FFMPEG, "FFmpeg isn't installed")
class FFmpegTestCase(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.folder = self._temp.name
        self.addCleanup(self._temp.cleanup)
        # Video and subtitles in different folders, with characters that trip FFmpeg up
        self.movies = os.path.join(self.folder, "Movies (2024)")
        os.makedirs(self.movies)
        self.subs = write_subtitles(os.path.join(self.folder, "Subs, out [B]", "Ocean's Eleven, Part [1]_subtitle.srt"))


class SubtitleTrackTests(FFmpegTestCase):
    def test_mp4_gets_a_switchable_track_without_re_encoding(self):
        film = os.path.join(self.movies, "Ocean's Eleven, Part [1].mp4")
        make_video(film)
        output = video.add_subtitle_track(film, self.subs, "ja")
        self.assertEqual(output, os.path.join(os.path.dirname(self.subs), "Ocean's Eleven, Part [1]_subtitled.mp4"))
        self.assertEqual(subtitle_tracks(output), [("mov_text", "jpn", 1)])
        self.assertEqual(video_stream_md5(output), video_stream_md5(film))

    def test_mkv_keeps_its_own_subtitles(self):
        plain = os.path.join(self.folder, "plain.mp4")
        make_video(plain)
        ass = os.path.join(self.folder, "old.ass")
        ffmpeg("-i", self.subs, ass)
        film = os.path.join(self.movies, "film.mkv")
        ffmpeg("-i", plain, "-i", ass, "-map", "0", "-map", "1", "-c", "copy",
               "-metadata:s:s:0", "language=eng", "-disposition:s:0", "default", film)
        output = video.add_subtitle_track(film, self.subs, "fr")
        self.assertEqual(subtitle_tracks(output), [("subrip", "fre", 1), ("ass", "eng", 0)])

    def test_avi_becomes_mkv(self):
        film = os.path.join(self.movies, "home video.avi")
        make_video(film, "mpeg4", "mp2")
        output = video.add_subtitle_track(film, self.subs, "en")
        self.assertTrue(output.endswith("home video_subtitled.mkv"))
        self.assertEqual(subtitle_tracks(output), [("subrip", "eng", 1)])
        self.assertEqual(video_stream_md5(output), video_stream_md5(film))

    def test_vtt_subtitles(self):
        film = os.path.join(self.movies, "clip.mov")
        make_video(film)
        vtt = write_subtitles(os.path.join(self.folder, "clip.vtt"), vtt=True)
        self.assertEqual(subtitle_tracks(video.add_subtitle_track(film, vtt, "de")), [("mov_text", "ger", 1)])

    def test_failure_raises_and_keeps_earlier_video(self):
        bad = os.path.join(self.folder, "broken.mp4")
        with open(bad, "w") as f:
            f.write("not a video")
        earlier = os.path.join(self.folder, "broken_subtitled.mp4")
        with open(earlier, "w") as f:
            f.write("earlier finished video")
        with self.assertRaisesRegex(ProcessingError, "adding them to the video failed"):
            video.add_subtitle_track(bad, write_subtitles(os.path.join(self.folder, "broken.srt")))
        with open(earlier) as f:
            self.assertEqual(f.read(), "earlier finished video")
        self.assertFalse(os.path.exists(os.path.join(self.folder, "broken_subtitled.partial.mp4")))


class BurnTests(FFmpegTestCase):
    def test_different_folder_and_awkward_names(self):
        film = os.path.join(self.movies, "Ocean's Eleven, Part [1].mp4")
        make_video(film)
        cwd = os.getcwd()
        output, _ = video.burn_subtitles(film, self.subs)  # NVENC if this PC can, otherwise the CPU encoder
        self.assertEqual(output, os.path.join(os.path.dirname(self.subs), "Ocean's Eleven, Part [1]_hardcoded.mp4"))
        self.assertEqual(os.getcwd(), cwd)
        self.assertLess(brightest_pixel(film), 40)
        self.assertGreater(brightest_pixel(output), 150, "subtitle text should be drawn on the frame")
        self.assertFalse(any("partial" in name for name in os.listdir(os.path.dirname(self.subs))))

    def test_vtt(self):
        film = os.path.join(self.folder, "clip.mp4")
        make_video(film)
        output, _ = video.burn_subtitles(film, write_subtitles(os.path.join(self.folder, "clip.vtt"), vtt=True))
        self.assertGreater(brightest_pixel(output), 150)

    def test_failure_raises_and_keeps_earlier_video(self):
        bad = os.path.join(self.folder, "broken.mp4")
        with open(bad, "w") as f:
            f.write("not a video")
        earlier = os.path.join(self.folder, "broken_hardcoded.mp4")
        with open(earlier, "w") as f:
            f.write("earlier finished video")
        with self.assertRaisesRegex(ProcessingError, "burning them into the video failed"):
            video.burn_subtitles(bad, write_subtitles(os.path.join(self.folder, "broken.srt")))
        with open(earlier) as f:
            self.assertEqual(f.read(), "earlier finished video")
        self.assertFalse(os.path.exists(os.path.join(self.folder, "broken_hardcoded.partial.mp4")))


if __name__ == "__main__":
    unittest.main()
