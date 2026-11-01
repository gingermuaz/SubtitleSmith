"""Burns (hardcodes) subtitles into a copy of a video with FFmpeg."""
import os
import shutil
import subprocess
import tempfile

from errors import ProcessingError

VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov')


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def can_burn(media_path):
    """True for the video types burn-in supports."""
    return media_path.lower().endswith(VIDEO_EXTS)


def burn_subtitles(video_path, subtitle_file):
    """Re-encodes the video with the subtitles drawn on. Returns the new video's path.

    The new video is saved next to the subtitle file as <video name>_hardcoded.mp4.
    """
    if not ffmpeg_available():
        raise ProcessingError("Subtitles saved, but FFmpeg isn't installed, so they couldn't be burned in.")

    video_path = os.path.abspath(video_path)
    subtitle_file = os.path.abspath(subtitle_file)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    output_video = os.path.join(os.path.dirname(subtitle_file), f"{stem}_hardcoded.mp4")
    partial_video = os.path.join(os.path.dirname(subtitle_file), f"{stem}_hardcoded.partial.mp4")

    # FFmpeg's subtitles filter treats ' , : [ ] \ as special, and Windows paths always contain : and \.
    # Rather than escape them, copy the subtitles to a plain name in a temp folder and run FFmpeg there.
    # The video paths are normal arguments, so they need no escaping.
    with tempfile.TemporaryDirectory() as temp_dir:
        sub_name = "subtitles" + os.path.splitext(subtitle_file)[1]
        shutil.copyfile(subtitle_file, os.path.join(temp_dir, sub_name))
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"subtitles={sub_name}",
            "-c:a", "copy",
            partial_video
        ]
        creationflags = 0x08000000 if os.name == 'nt' else 0
        process = subprocess.run(ffmpeg_cmd, cwd=temp_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 creationflags=creationflags)

    if process.returncode != 0:
        # Remove our half-written file; any earlier finished video is left untouched
        if os.path.exists(partial_video):
            os.remove(partial_video)
        lines = [line.strip() for line in process.stderr.decode('utf-8', errors='replace').splitlines()
                 if line.strip() and line.strip() != "Conversion failed!"]
        detail = " ".join(lines[-2:]) if lines else f"FFmpeg exit code {process.returncode}"
        raise ProcessingError(f"Subtitles saved, but burning them into the video failed: {detail}")

    os.replace(partial_video, output_video)
    return output_video
