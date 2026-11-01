"""Puts subtitles into videos with FFmpeg: as a track viewers can switch on and off, or burned into the picture."""
import os
import shutil
import subprocess
import tempfile

import languages
from errors import ProcessingError

VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov')

# Where a subtitle track goes for each video type: (output extension, subtitle format).
# AVI can't hold subtitle tracks, so AVI videos are copied into MKV.
_TRACK_FORMATS = {".mp4": (".mp4", "mov_text"), ".mov": (".mov", "mov_text"),
                  ".mkv": (".mkv", "srt"), ".avi": (".mkv", "srt")}

# NVIDIA's hardware encoder at roughly the quality of FFmpeg's default CPU encoder (x264 at CRF 23)
NVENC = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0"]

_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
_nvenc_listed = None


def ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def is_supported_video(media_path):
    """True for the video types that subtitles can be added to."""
    return media_path.lower().endswith(VIDEO_EXTS)


def nvenc_listed():
    """True if this FFmpeg includes NVIDIA's encoder. Whether the GPU can use it is only known when it runs."""
    global _nvenc_listed
    if _nvenc_listed is None:
        process = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
        _nvenc_listed = b"h264_nvenc" in process.stdout
    return _nvenc_listed


def _run_ffmpeg(args, cwd=None):
    return subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args, cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)


def _failed(process, partial_video, what):
    """Removes our half-written file (any earlier finished video is left alone) and raises a readable error."""
    if os.path.exists(partial_video):
        os.remove(partial_video)
    lines = [line.strip() for line in process.stderr.decode('utf-8', errors='replace').splitlines()
             if line.strip() and line.strip() != "Conversion failed!"]
    detail = " ".join(lines[-2:]) if lines else f"FFmpeg exit code {process.returncode}"
    raise ProcessingError(f"Subtitles saved, but {what} failed: {detail}")


def add_subtitle_track(video_path, subtitle_file, language="und"):
    """Copies the video with the subtitles added as a track viewers can switch on and off. Returns its path.

    Nothing is re-encoded, so this takes seconds. The copy is saved next to the subtitle file as
    <video name>_subtitled.<ext> (AVI becomes .mkv). The new track is first and on by default,
    and is labelled with the subtitle language.
    """
    if not ffmpeg_available():
        raise ProcessingError("Subtitles saved, but FFmpeg isn't installed, so they couldn't be added to the video.")

    video_path = os.path.abspath(video_path)
    subtitle_file = os.path.abspath(subtitle_file)
    stem, ext = os.path.splitext(os.path.basename(video_path))
    out_ext, subtitle_codec = _TRACK_FORMATS[ext.lower()]
    folder = os.path.dirname(subtitle_file)
    output_video = os.path.join(folder, f"{stem}_subtitled{out_ext}")
    partial_video = os.path.join(folder, f"{stem}_subtitled.partial{out_ext}")

    # Keep the video's own streams; the new subtitles become the first subtitle track (s:0)
    args = ["-i", video_path, "-i", subtitle_file,
            "-map", "0:v", "-map", "0:a?", "-map", "1:0", "-map", "0:s?"]
    if out_ext == ".mkv":
        args += ["-map", "0:t?"]  # fonts attached to MKV files
    args += ["-c", "copy", "-c:s:0", subtitle_codec,
             "-metadata:s:s:0", f"language={languages.iso639_2(language)}",
             "-metadata:s:s:0", f"{'title' if out_ext == '.mkv' else 'handler_name'}={languages.name(language)}",
             "-disposition:s", "0", "-disposition:s:0", "default",  # ours is the one players switch on
             partial_video]
    process = _run_ffmpeg(args)
    if process.returncode != 0:
        _failed(process, partial_video, "adding them to the video")
    os.replace(partial_video, output_video)
    return output_video


def burn_subtitles(video_path, subtitle_file):
    """Re-encodes the video with the subtitles drawn on. Returns (new video's path, True if the GPU encoded it).

    The new video is saved next to the subtitle file as <video name>_hardcoded.mp4. NVIDIA's encoder
    (NVENC) is tried first because it's much faster; if FFmpeg can't use it, FFmpeg's standard CPU
    encoder is used instead.
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
        args = ["-i", video_path, "-vf", f"subtitles={sub_name}", "-pix_fmt", "yuv420p", "-c:a", "copy"]
        encoders = ([NVENC] if nvenc_listed() else []) + [[]]  # [] = FFmpeg's default CPU encoder
        for encoder in encoders:
            process = _run_ffmpeg(args + encoder + [partial_video], cwd=temp_dir)
            if process.returncode == 0:
                break

    if process.returncode != 0:
        _failed(process, partial_video, "burning them into the video")
    os.replace(partial_video, output_video)
    return output_video, encoder == NVENC
