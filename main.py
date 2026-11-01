import os
import sys
import gc
import subprocess
import shutil
import tempfile

# Number GPUs the same way nvidia-smi does, so the GPU the hardware scan picks is the one
# the model loads onto. Must be set before CUDA starts up.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from faster_whisper import WhisperModel
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- AGGRESSIVE WINDOWS DLL INJECTION ---
venv_dir = os.path.dirname(os.path.dirname(sys.executable))
site_packages_dir = os.path.join(venv_dir, "Lib", "site-packages")
cublas_bin = os.path.join(site_packages_dir, "nvidia", "cublas", "bin")
cudnn_bin = os.path.join(site_packages_dir, "nvidia", "cudnn", "bin")

os.environ["PATH"] = f"{cublas_bin};{cudnn_bin};" + os.environ.get("PATH", "")
if hasattr(os, 'add_dll_directory'):
    if os.path.exists(cublas_bin): os.add_dll_directory(cublas_bin)
    if os.path.exists(cudnn_bin): os.add_dll_directory(cudnn_bin)


# ----------------------------------------

def format_timestamp(seconds: float, is_vtt=False):
    """Formats time to SRT (00:00:00,000) or VTT (00:00:00.000) standards."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    separator = "." if is_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


# Map UI names to HuggingFace repositories for the optimized models
MODEL_MAPPINGS = {
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}

VIDEO_EXTS = ('.mp4', '.mkv', '.avi', '.mov')


class ProcessingError(Exception):
    """A problem with one file, worded for the user."""


class ProcessCancelled(Exception):
    """The user pressed Cancel."""


_loaded = {"key": None, "model": None}


def get_model(model_size, compute_device="auto", device_index=0, progress_callback=None):
    """Loads a model once and reuses it until the model or device changes."""
    key = (model_size, compute_device, device_index)
    if _loaded["key"] == key:
        return _loaded["model"]

    # Drop the old model first so two large models never sit in VRAM at once
    _loaded["key"] = None
    _loaded["model"] = None
    gc.collect()

    if progress_callback: progress_callback(0, 100, f"Loading '{model_size}' model onto {compute_device.upper()}...")
    c_type = "int8" if compute_device == "cpu" else "int8_float16"
    model = WhisperModel(MODEL_MAPPINGS.get(model_size, model_size), device=compute_device,
                         device_index=device_index, compute_type=c_type)
    _loaded["key"] = key
    _loaded["model"] = model
    return model


def burn_subtitles(video_path, subtitle_file):
    """Re-encodes the video with the subtitles drawn on. Returns the new video's path."""
    if not shutil.which("ffmpeg"):
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


def translate_movie(video_path, source_language, output_file, model_size="large-v3", task="translate",
                    compute_device="auto", vad_filter=True, output_format=".srt", hardcode_subtitles=False,
                    word_timestamps=False, device_index=0, progress_callback=None, cancel_check=None):
    """Makes subtitles for one media file.

    Returns {"subtitle_file", "video_file", "notes"}. Raises ProcessCancelled if the user cancels,
    and ProcessingError (or faster-whisper's own errors) if something goes wrong.
    """
    model = get_model(model_size, compute_device, device_index, progress_callback)

    if progress_callback: progress_callback(0, 100, "Extracting audio and analyzing...")

    lang_param = None if source_language == "auto" else source_language

    # Advanced tuning for VHS static and noisy background audio
    vad_params = dict(min_silence_duration_ms=500, speech_pad_ms=400) if vad_filter else None

    segments, info = model.transcribe(
        video_path,
        task=task,
        language=lang_param,
        beam_size=10,
        best_of=5,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        initial_prompt="Clear, formal dialogue, high-definition audio transcription.",
        vad_filter=vad_filter,
        vad_parameters=vad_params,
        word_timestamps=word_timestamps
    )
    total_duration = round(info.duration, 2)

    content = "WEBVTT\n\n" if output_format == ".vtt" else ""
    is_vtt = (output_format == ".vtt")

    subtitle_index = 1

    with tqdm(total=total_duration, unit="sec",
              bar_format="{l_bar}{bar}| {n:.1f}/{total:.1f} [{elapsed}<{remaining}]") as pbar:
        for segment in segments:
            if cancel_check and cancel_check():
                if progress_callback: progress_callback(0, 100, "Process Canceled.")
                raise ProcessCancelled()

            if output_format in [".srt", ".vtt"]:
                if word_timestamps and segment.words:
                    # Output individual words as their own subtitle block
                    for word in segment.words:
                        start_time = format_timestamp(word.start, is_vtt)
                        end_time = format_timestamp(word.end, is_vtt)
                        content += f"{subtitle_index}\n"
                        content += f"{start_time} --> {end_time}\n"
                        content += f"{word.word.strip()}\n\n"
                        subtitle_index += 1
                else:
                    # Standard sentence-level subtitle block
                    start_time = format_timestamp(segment.start, is_vtt)
                    end_time = format_timestamp(segment.end, is_vtt)
                    content += f"{subtitle_index}\n"
                    content += f"{start_time} --> {end_time}\n"
                    content += f"{segment.text.strip()}\n\n"
                    subtitle_index += 1
            elif output_format == ".txt":
                # Standard text ignores timestamps completely
                content += f"{segment.text.strip()}\n"

            pbar.update(segment.end - pbar.n)

            if progress_callback:
                percentage = min(100.0, round((segment.end / total_duration) * 100, 1))
                action_text = "Translating" if task == "translate" else "Transcribing"
                progress_callback(segment.end, total_duration, f"{action_text}... {percentage}%")

    if progress_callback: progress_callback(100, 100, "Saving text file...")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)

    result = {"subtitle_file": output_file, "video_file": None, "notes": []}

    # --- PHASE 3: HARDCODE SUBTITLES ---
    if hardcode_subtitles:
        if output_format not in (".srt", ".vtt"):
            result["notes"].append("Burn-in skipped: .txt files have no timings to burn in.")
        elif not video_path.lower().endswith(VIDEO_EXTS):
            result["notes"].append("Burn-in skipped: only MP4, MKV, AVI and MOV videos are supported.")
        else:
            if progress_callback: progress_callback(100, 100, "Burning subtitles to video... (This takes a while)")
            result["video_file"] = burn_subtitles(video_path, output_file)

    if progress_callback: progress_callback(100, 100, "Success! File saved.")
    return result


if __name__ == "__main__":
    print("Please run GUI.py to start the application.")