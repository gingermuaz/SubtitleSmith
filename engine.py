"""The speech-to-text engine: loads faster-whisper models and turns one media file into subtitles.

process_file() is the entry point. It runs transcribe(), saves the result with writers.py and,
if asked, adds it to a copy of the video with video.py (as a subtitle track, or burned in).
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

from faster_whisper import BatchedInferencePipeline, WhisperModel

import languages
import video
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
               names_and_terms="", batch_size=0, progress_callback=None, cancel_check=None):
    """Runs the model over one file. Returns (segments, info); info.language is the spoken language.

    batch_size > 1 turns on Fast Mode (it needs vad_filter). If the GPU runs out of memory, the
    batch size is halved and the file starts again. Raises ProcessCancelled if cancel_check()
    becomes true part-way through.
    """
    while True:
        try:
            return _transcribe_once(media_path, model, source_language, task, vad_filter, names_and_terms,
                                    batch_size, progress_callback, cancel_check)
        except RuntimeError as e:
            if batch_size <= 1 or "out of memory" not in str(e).lower():
                raise
            batch_size //= 2
            gc.collect()
            if progress_callback:
                progress_callback(0, 100, f"Out of GPU memory, starting again with smaller batches ({batch_size})...")


def _transcribe_once(media_path, model, source_language, task, vad_filter, names_and_terms, batch_size,
                     progress_callback, cancel_check):
    if progress_callback: progress_callback(0, 100, "Extracting audio and analyzing...")

    lang_param = None if source_language == "auto" else source_language

    # Advanced tuning for VHS static and noisy background audio
    vad_params = dict(min_silence_duration_ms=500, speech_pad_ms=400) if vad_filter else None

    options = dict(
        task=task,
        language=lang_param,
        best_of=5,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        # Names & terms to spell correctly; faster-whisper adds them to every 30-second window
        hotwords=(names_and_terms or "").strip() or None,
        vad_parameters=vad_params,
        word_timestamps=True,  # for readable line splitting (and the silence check below)
    )
    if batch_size > 1 and vad_filter:
        # Fast Mode: faster-whisper cuts the audio at pauses and transcribes batch_size pieces at once.
        # In this mode it never carries text between pieces, doesn't retry at higher temperatures and
        # has no silence check. Beam size 5 (faster-whisper's default) keeps the batches within VRAM.
        segments, info = BatchedInferencePipeline(model=model).transcribe(
            media_path, beam_size=5, batch_size=batch_size, vad_filter=True,
            without_timestamps=False,  # timings inside each piece, so segments end at natural breaks
            **options)
    else:
        segments, info = model.transcribe(
            media_path, beam_size=10, vad_filter=vad_filter,
            # Stop repeated lines on long films: don't feed each window the previous window's text
            # (so one mistake can't loop), and skip long silences where Whisper tends to invent text
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
            **options)
    total_duration = round(info.duration, 2)
    spoken = languages.name(info.language)
    action_text = f"Translating from {spoken}" if task == "translate" else f"Transcribing {spoken}"

    results = []
    for segment in segments:  # faster-whisper does the work as we loop
        if cancel_check and cancel_check():
            if progress_callback: progress_callback(0, 100, "Process Canceled.")
            raise ProcessCancelled()
        results.append(segment)

        if progress_callback:
            percentage = min(100.0, round((segment.end / total_duration) * 100, 1))
            progress_callback(segment.end, total_duration, f"{action_text}... {percentage}%")
    return results, info


def process_file(media_path, output_file, model_size="large-v3", source_language="auto", task="translate",
                 compute_device="auto", device_index=0, vad_filter=True, output_format=".srt",
                 readable_lines=True, names_and_terms="", video_output="none", batch_size=0,
                 progress_callback=None, cancel_check=None):
    """Makes subtitles for one media file and, if asked, adds them to a copy of the video.

    video_output: "none", "track" (a subtitle track viewers can switch on and off) or "burn" (drawn
    into the picture). batch_size > 1 turns on Fast Mode (see transcribe).

    Returns {"subtitle_file", "video_file", "notes", "language", "language_probability",
    "language_detected"}. Raises ProcessCancelled if the user cancels, and ProcessingError
    (or faster-whisper's own errors) if something goes wrong.
    """
    if video_output not in ("none", "track", "burn"):
        raise ValueError(f"Unknown video output: {video_output}")
    model = get_model(model_size, compute_device, device_index, progress_callback)
    segments, info = transcribe(media_path, model, source_language, task, vad_filter, names_and_terms,
                                batch_size, progress_callback, cancel_check)

    if progress_callback: progress_callback(100, 100, "Saving text file...")
    # Translations come out in English; otherwise the subtitles are in the spoken language
    subtitle_language = "en" if task == "translate" else info.language
    writers.write(output_file, segments, output_format, readable_lines, languages.line_length(subtitle_language))
    result = {"subtitle_file": output_file, "video_file": None, "notes": [],
              "language": info.language, "language_probability": info.language_probability,
              "language_detected": source_language == "auto"}
    if result["language_detected"] and info.language_probability < 0.5:
        result["notes"].append(f"Not sure which language this is (best guess {languages.describe(info.language, info.language_probability)}). "
                               "If the subtitles look wrong, choose the Source Language yourself.")

    if video_output != "none":
        if output_format not in (".srt", ".vtt"):
            result["notes"].append("Video skipped: .txt files have no timings to add to a video.")
        elif not video.is_supported_video(media_path):
            result["notes"].append("Video skipped: only MP4, MKV, AVI and MOV videos are supported.")
        elif video_output == "track":
            if progress_callback: progress_callback(100, 100, "Adding the subtitle track to the video...")
            result["video_file"] = video.add_subtitle_track(media_path, output_file, subtitle_language)
        else:
            if progress_callback: progress_callback(100, 100, "Burning subtitles to video... (This takes a while)")
            result["video_file"], used_gpu = video.burn_subtitles(media_path, output_file)
            if not used_gpu:
                result["notes"].append("Burned in with the CPU encoder, because FFmpeg couldn't use the NVIDIA "
                                       "encoder (NVENC). It works, just more slowly.")

    if progress_callback: progress_callback(100, 100, "Success! File saved.")
    return result
