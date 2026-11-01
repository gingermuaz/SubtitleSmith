"""The speech-to-text engine: loads faster-whisper models and turns one media file into subtitles.

process_file() is the entry point. It runs transcribe(), saves the result with writers.py
and, if asked, burns it into a copy of the video with burn.py.
"""
import os
import sys
import gc
import importlib.util
import warnings

# Number GPUs the same way nvidia-smi does, so the GPU the hardware scan picks is the one
# the model loads onto. Must be set before CUDA starts up.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


# --- WINDOWS CUDA DLLs ---
# CTranslate2 needs cuBLAS and cuDNN from the nvidia-cublas-cu12 / nvidia-cudnn-cu12 pip packages.
# Their DLLs live in nvidia/<lib>/bin: under site-packages when running from source, and inside the
# bundle folder (sys._MEIPASS) in the packaged .exe, where SubtitleSmith.spec puts them.
def _cuda_dll_folders():
    if getattr(sys, "frozen", False):
        nvidia_roots = [os.path.join(sys._MEIPASS, "nvidia")]
    else:
        spec = importlib.util.find_spec("nvidia")
        nvidia_roots = list(spec.submodule_search_locations or []) if spec else []
    folders = [os.path.join(root, lib, "bin") for root in nvidia_roots for lib in ("cublas", "cudnn")]
    return [folder for folder in folders if os.path.isdir(folder)]


_dll_folders = _cuda_dll_folders()
os.environ["PATH"] = os.pathsep.join(_dll_folders + [os.environ.get("PATH", "")])
if hasattr(os, 'add_dll_directory'):
    for _folder in _dll_folders:
        os.add_dll_directory(_folder)
# ----------------------------------------

from faster_whisper import WhisperModel

import burn
import writers
from errors import ProcessCancelled  # GUI.py catches this as engine.ProcessCancelled

warnings.filterwarnings("ignore")

# Map UI names to HuggingFace repositories for the optimized models
MODEL_MAPPINGS = {
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}

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


def transcribe(media_path, model, source_language="auto", task="translate", vad_filter=True,
               word_timestamps=False, progress_callback=None, cancel_check=None):
    """Runs the model over one file and returns its segments.

    Raises ProcessCancelled if cancel_check() becomes true part-way through.
    """
    if progress_callback: progress_callback(0, 100, "Extracting audio and analyzing...")

    lang_param = None if source_language == "auto" else source_language

    # Advanced tuning for VHS static and noisy background audio
    vad_params = dict(min_silence_duration_ms=500, speech_pad_ms=400) if vad_filter else None

    segments, info = model.transcribe(
        media_path,
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
    action_text = "Translating" if task == "translate" else "Transcribing"

    results = []
    for segment in segments:  # faster-whisper does the work as we loop
        if cancel_check and cancel_check():
            if progress_callback: progress_callback(0, 100, "Process Canceled.")
            raise ProcessCancelled()
        results.append(segment)

        if progress_callback:
            percentage = min(100.0, round((segment.end / total_duration) * 100, 1))
            progress_callback(segment.end, total_duration, f"{action_text}... {percentage}%")
    return results


def process_file(media_path, output_file, model_size="large-v3", source_language="auto", task="translate",
                 compute_device="auto", device_index=0, vad_filter=True, output_format=".srt",
                 word_timestamps=False, hardcode_subtitles=False, progress_callback=None, cancel_check=None):
    """Makes subtitles for one media file and, if asked, burns them into a copy of the video.

    Returns {"subtitle_file", "video_file", "notes"}. Raises ProcessCancelled if the user cancels,
    and ProcessingError (or faster-whisper's own errors) if something goes wrong.
    """
    model = get_model(model_size, compute_device, device_index, progress_callback)
    segments = transcribe(media_path, model, source_language, task, vad_filter, word_timestamps,
                          progress_callback, cancel_check)

    if progress_callback: progress_callback(100, 100, "Saving text file...")
    writers.write(output_file, segments, output_format, word_timestamps)
    result = {"subtitle_file": output_file, "video_file": None, "notes": []}

    if hardcode_subtitles:
        if output_format not in (".srt", ".vtt"):
            result["notes"].append("Burn-in skipped: .txt files have no timings to burn in.")
        elif not burn.can_burn(media_path):
            result["notes"].append("Burn-in skipped: only MP4, MKV, AVI and MOV videos are supported.")
        else:
            if progress_callback: progress_callback(100, 100, "Burning subtitles to video... (This takes a while)")
            result["video_file"] = burn.burn_subtitles(media_path, output_file)

    if progress_callback: progress_callback(100, 100, "Success! File saved.")
    return result
