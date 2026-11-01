"""Puts subtitles into videos with FFmpeg: as a track viewers can switch on and off, or burned into the picture."""
import os
import shutil
import subprocess
import tempfile
import threading
import time

import languages
from errors import ProcessCancelled, ProcessingError

VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov')

# Where a subtitle track goes for each video type: (output extension, subtitle format).
# AVI can't hold subtitle tracks, so AVI videos are copied into MKV.
_TRACK_FORMATS = {".mp4": (".mp4", "mov_text"), ".mov": (".mov", "mov_text"),
                  ".mkv": (".mkv", "srt"), ".avi": (".mkv", "srt")}

# Burn-in looks (the standard look is Medium, White, Bottom). Sizes are libass's: 16 is its default.
BURN_SIZES = {"Small": 12, "Medium": 16, "Large": 22, "Extra large": 28}
BURN_COLOURS = {"White": "&H00FFFFFF", "Yellow": "&H0000FFFF", "Cyan": "&H00FFFF00", "Green": "&H0000FF00"}
# libass reads a forced alignment for .srt files the old SSA way: 2 is bottom centre, 6 is top centre
BURN_POSITIONS = {"Bottom": 2, "Top": 6}

# NVIDIA's hardware encoder at roughly the quality of FFmpeg's default CPU encoder (x264 at CRF 23)
NVENC = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "0"]

_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
_nvenc_listed = None
_running = set()  # FFmpeg processes in progress, so closing the window can stop them


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


def _run_ffmpeg(args, cwd=None, cancel_check=None, on_progress=None):
    """Runs FFmpeg, stopping it if cancel_check() becomes true (raises ProcessCancelled).

    on_progress(seconds) is called with how far into the video FFmpeg has got.
    """
    if cancel_check and cancel_check():
        raise ProcessCancelled()
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-nostats",
               "-progress", "pipe:1"] + args
    with tempfile.TemporaryFile() as errors:  # a file, so a chatty FFmpeg can't fill a pipe and stall
        process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=errors, creationflags=_NO_WINDOW)
        _running.add(process)
        reader = threading.Thread(target=_read_progress, args=(process.stdout, on_progress), daemon=True)
        reader.start()
        try:
            while process.poll() is None:
                if cancel_check and cancel_check():
                    raise ProcessCancelled()
                time.sleep(0.2)
        except BaseException:  # Cancel, Ctrl+C or anything else: don't leave FFmpeg running
            process.kill()
            process.wait()
            raise
        finally:
            _running.discard(process)
            reader.join(timeout=5)
            process.stdout.close()
        errors.seek(0)
        return subprocess.CompletedProcess(command, process.returncode, b"", errors.read())


def _read_progress(stdout, on_progress):
    """Reads FFmpeg's -progress lines (key=value), passing the position in seconds to on_progress."""
    for raw in stdout:
        key, _, value = raw.decode("utf-8", errors="replace").strip().partition("=")
        if on_progress and key in ("out_time_us", "out_time_ms") and value.lstrip("-").isdigit():
            on_progress(max(0, int(value)) / 1_000_000)  # both are microseconds (out_time_ms is misnamed)


def stop_all():
    """Stops any FFmpeg still running, e.g. when the window is closed in the middle of a burn-in."""
    for process in list(_running):
        try:
            process.kill()
        except OSError:
            pass


def _remove(partial_video):
    """Removes our half-written file. Any earlier finished video is left alone."""
    if os.path.exists(partial_video):
        os.remove(partial_video)


def _failed(process, partial_video, what):
    """Removes our half-written file and raises a readable error."""
    _remove(partial_video)
    lines = [line.strip() for line in process.stderr.decode('utf-8', errors='replace').splitlines()
             if line.strip() and line.strip() != "Conversion failed!"]
    detail = " ".join(lines[-2:]) if lines else f"FFmpeg exit code {process.returncode}"
    raise ProcessingError(f"Subtitles saved, but {what} failed: {detail}")


def add_subtitle_track(video_path, subtitle_file, language="und", cancel_check=None, on_progress=None):
    """add_subtitle_tracks() with one subtitle file."""
    return add_subtitle_tracks(video_path, [(subtitle_file, language)], cancel_check, on_progress)


def add_subtitle_tracks(video_path, tracks, cancel_check=None, on_progress=None):
    """Copies the video with subtitle tracks viewers can switch on and off. Returns the new video's path.

    tracks is [(subtitle file, Whisper language code), ...]; each is labelled with its language and the
    first is on by default. Nothing is re-encoded, so this takes seconds. The copy is saved next to the
    first subtitle file as <video name>_subtitled.<ext> (AVI becomes .mkv). Raises ProcessCancelled
    if cancel_check() becomes true.
    """
    if not ffmpeg_available():
        raise ProcessingError("Subtitles saved, but FFmpeg isn't installed, so they couldn't be added to the video.")

    video_path = os.path.abspath(video_path)
    stem, ext = os.path.splitext(os.path.basename(video_path))
    out_ext, subtitle_codec = _TRACK_FORMATS[ext.lower()]
    folder = os.path.dirname(os.path.abspath(tracks[0][0]))
    output_video = os.path.join(folder, f"{stem}_subtitled{out_ext}")
    partial_video = os.path.join(folder, f"{stem}_subtitled.partial{out_ext}")

    # Keep the video's own streams; the new subtitles become the first subtitle tracks (s:0, s:1, ...)
    args = ["-i", video_path]
    for subtitle_file, _ in tracks:
        args += ["-i", os.path.abspath(subtitle_file)]
    args += ["-map", "0:v", "-map", "0:a?"]
    for n in range(len(tracks)):
        args += ["-map", f"{n + 1}:0"]
    args += ["-map", "0:s?"]
    if out_ext == ".mkv":
        args += ["-map", "0:t?"]  # fonts attached to MKV files
    args += ["-c", "copy", "-disposition:s", "0"]
    name_key = "title" if out_ext == ".mkv" else "handler_name"
    for n, (_, language) in enumerate(tracks):
        args += [f"-c:s:{n}", subtitle_codec,
                 f"-metadata:s:s:{n}", f"language={languages.iso639_2(language)}",
                 f"-metadata:s:s:{n}", f"{name_key}={languages.name(language)}"]
    args += ["-disposition:s:0", "default", partial_video]  # ours first, and the one players switch on
    try:
        process = _run_ffmpeg(args, cancel_check=cancel_check, on_progress=on_progress)
    except ProcessCancelled:
        _remove(partial_video)
        raise
    if process.returncode != 0:
        _failed(process, partial_video, "adding them to the video")
    os.replace(partial_video, output_video)
    return output_video


def burn_style_filter(style):
    """FFmpeg force_style text for a burn-in look like {"size": "Large", "colour": "Yellow", "position": "Top"}.

    Returns "" for the standard look, so the command is exactly as before.
    """
    style = style or {}
    parts = []
    size = BURN_SIZES.get(style.get("size"), 16)
    if size != 16:
        parts.append(f"FontSize={size}")
    if style.get("colour") in BURN_COLOURS and style.get("colour") != "White":
        parts.append(f"PrimaryColour={BURN_COLOURS[style['colour']]}")
    if BURN_POSITIONS.get(style.get("position"), 2) != 2:
        parts.append(f"Alignment={BURN_POSITIONS[style['position']]}")
    return ",".join(parts)


def burn_subtitles(video_path, subtitle_file, cancel_check=None, on_progress=None, style=None):
    """Re-encodes the video with the subtitles drawn on. Returns (new video's path, True if the GPU encoded it).

    The new video is saved next to the subtitle file as <video name>_hardcoded.mp4. NVIDIA's encoder
    (NVENC) is tried first because it's much faster; if FFmpeg can't use it, FFmpeg's standard CPU
    encoder is used instead. on_progress(seconds) reports how far the encode has got, and it raises
    ProcessCancelled if cancel_check() becomes true. style sets the look (see burn_style_filter).
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
        force_style = burn_style_filter(style)
        subtitles_filter = f"subtitles={sub_name}" + (f":force_style='{force_style}'" if force_style else "")
        args = ["-i", video_path, "-vf", subtitles_filter, "-pix_fmt", "yuv420p", "-c:a", "copy"]
        encoders = ([NVENC] if nvenc_listed() else []) + [[]]  # [] = FFmpeg's default CPU encoder
        for encoder in encoders:
            try:
                process = _run_ffmpeg(args + encoder + [partial_video], cwd=temp_dir,
                                      cancel_check=cancel_check, on_progress=on_progress)
            except ProcessCancelled:
                _remove(partial_video)
                raise
            if process.returncode == 0:
                break

    if process.returncode != 0:
        _failed(process, partial_video, "burning them into the video")
    os.replace(partial_video, output_video)
    return output_video, encoder == NVENC
