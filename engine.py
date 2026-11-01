"""The speech-to-text engine: loads faster-whisper models and turns one media file into subtitles.

process_file() is the entry point. It runs transcribe(), saves the result with writers.py and,
if asked, adds it to a copy of the video with video.py (as a subtitle track, or burned in).
"""
import os
import sys
import gc
import fnmatch
import importlib.util
import threading
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

# Download models over plain HTTPS rather than Hugging Face's Xet engine, which runs in its own native
# thread: plain downloads report progress to us chunk by chunk, stop cleanly on Cancel and resume later.
# Must be set before huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import av
import huggingface_hub
from faster_whisper import BatchedInferencePipeline, WhisperModel
from huggingface_hub.utils import tqdm as HfProgressBar

import languages
import video
import writers
from errors import ProcessCancelled, ProcessingError  # GUI.py catches ProcessCancelled as engine.ProcessCancelled

warnings.filterwarnings("ignore")

# Hugging Face repositories for the models in the AI Model menu
MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}
# The files faster-whisper needs from a model repository (the same list it downloads itself)
MODEL_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]

_loaded = {"key": None, "model": None}
_load_lock = threading.Lock()


def _run_cancellable(work, cancel_check):
    """Runs work() on a helper thread so Cancel doesn't have to wait for it to finish.

    On Cancel this raises ProcessCancelled straight away. work() carries on in the background and its
    result is dropped (except a model that finishes loading, which is kept for next time).
    """
    outcome = {}

    def run():
        try:
            outcome["result"] = work()
        except BaseException as e:  # handed back to the caller below
            outcome["error"] = e

    helper = threading.Thread(target=run, daemon=True)
    helper.start()
    while helper.is_alive():
        helper.join(0.2)
        if cancel_check and cancel_check() and helper.is_alive():
            raise ProcessCancelled()
    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def _size(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _download_progress_bar(on_bytes):
    """A progress-bar class for huggingface_hub that passes the size of each downloaded chunk to on_bytes."""
    class DownloadProgress(HfProgressBar):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True  # no console progress bar
            super().__init__(*args, **kwargs)
            on_bytes(kwargs.get("initial") or 0)  # bytes already there from an interrupted download

        def update(self, n=1):
            on_bytes(n or 0)  # raising ProcessCancelled here stops the download
            return super().update(n)
    return DownloadProgress


def model_folder(model_size, progress_callback=None, cancel_check=None):
    """The local folder holding a model's files, downloading them from Hugging Face first if needed.

    Downloads report progress (stage "download") and stop on Cancel; an interrupted download
    carries on from where it stopped next time.
    """
    repo_id = MODEL_REPOS.get(model_size, model_size)
    try:
        folder = huggingface_hub.snapshot_download(repo_id, allow_patterns=MODEL_FILES, local_files_only=True)
        if os.path.exists(os.path.join(folder, "model.bin")):
            return folder  # already downloaded
    except Exception:  # not downloaded yet, or only partly
        pass

    try:
        info = huggingface_hub.HfApi().model_info(repo_id, files_metadata=True)
    except Exception as e:
        raise ProcessingError(f"Couldn't download the '{model_size}' AI model. Each model downloads once, the "
                              f"first time you use it, so this needs an internet connection. ({e})") from e
    files = [(f.rfilename, f.size or 0) for f in info.siblings
             if any(fnmatch.fnmatch(f.rfilename, pattern) for pattern in MODEL_FILES)]
    total = sum(size for _, size in files) or 1
    done = 0

    def on_bytes(n):
        nonlocal done
        done += n
        if cancel_check and cancel_check():
            raise ProcessCancelled()
        if progress_callback:
            progress_callback(min(done, total), total, f"Downloading the '{model_size}' AI model (first use only)... "
                                                       f"{_size(min(done, total))} of {_size(total)}", "download")

    path = None
    for filename, size in files:
        before = done
        try:
            path = huggingface_hub.hf_hub_download(repo_id, filename, tqdm_class=_download_progress_bar(on_bytes))
        except ProcessCancelled:
            raise
        except Exception as e:
            raise ProcessingError(f"Downloading the '{model_size}' AI model failed ({e}). Try again: it carries "
                                  "on from where it stopped.") from e
        done = before + size
    return os.path.dirname(path)


def get_model(model_size, compute_device="auto", device_index=0, progress_callback=None, cancel_check=None):
    """Loads a model once and reuses it until the model or device changes, downloading it first if needed.

    Raises ProcessCancelled if cancel_check() becomes true while it downloads or loads.
    """
    key = (model_size, compute_device, device_index)
    if _loaded["key"] == key:
        return _loaded["model"]
    folder = model_folder(model_size, progress_callback, cancel_check)
    if progress_callback:
        progress_callback(0, 100, f"Loading '{model_size}' model onto {compute_device.upper()}...", "load")
    return _run_cancellable(lambda: _load(key, folder), cancel_check)


def _load(key, folder):
    """Loads the model into memory (RAM or VRAM), replacing any other. Runs on a helper thread."""
    model_size, compute_device, device_index = key
    with _load_lock:  # one load at a time, even while a cancelled one is still finishing
        if _loaded["key"] == key:
            return _loaded["model"]
        # Drop the old model first so two large models never sit in VRAM at once
        _loaded["key"] = None
        _loaded["model"] = None
        gc.collect()
        c_type = "int8" if compute_device == "cpu" else "int8_float16"
        model = WhisperModel(folder, device=compute_device, device_index=device_index, compute_type=c_type)
        _loaded["key"] = key
        _loaded["model"] = model
        return model


def media_duration(path):
    """Length of a media file in seconds, or None if unknown. Reads only the file's header, so it's quick."""
    try:
        with av.open(path) as container:
            if container.duration:
                return container.duration / av.time_base
    except Exception:
        pass
    return None


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
                progress_callback(0, 100, f"Out of GPU memory, starting again with smaller batches ({batch_size})...",
                                  "prepare")


def _transcribe_once(media_path, model, source_language, task, vad_filter, names_and_terms, batch_size,
                     progress_callback, cancel_check):
    if progress_callback: progress_callback(0, 100, "Extracting audio and analyzing...", "prepare")

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
        def start():
            return BatchedInferencePipeline(model=model).transcribe(
                media_path, beam_size=5, batch_size=batch_size, vad_filter=True,
                without_timestamps=False,  # timings inside each piece, so segments end at natural breaks
                **options)
    else:
        def start():
            return model.transcribe(
                media_path, beam_size=10, vad_filter=vad_filter,
                # Stop repeated lines on long films: don't feed each window the previous window's text
                # (so one mistake can't loop), and skip long silences where Whisper tends to invent text
                condition_on_previous_text=False,
                hallucination_silence_threshold=2.0,
                **options)
    # Reading the audio, finding speech and detecting the language all happen before the first
    # segment; on a long film that takes a while, so Cancel shouldn't have to wait for it
    segments, info = _run_cancellable(start, cancel_check)
    total_duration = round(info.duration, 2)
    spoken = languages.name(info.language)
    action_text = f"Translating from {spoken}" if task == "translate" else f"Transcribing {spoken}"

    results = []
    for segment in segments:  # faster-whisper does the work as we loop
        if cancel_check and cancel_check():
            if progress_callback: progress_callback(0, 100, "Process Canceled.", "canceled")
            raise ProcessCancelled()
        results.append(segment)

        if progress_callback:
            percentage = min(100.0, round((segment.end / total_duration) * 100, 1))
            progress_callback(segment.end, total_duration, f"{action_text}... {percentage}%", "transcribe")
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
    model = get_model(model_size, compute_device, device_index, progress_callback, cancel_check)
    segments, info = transcribe(media_path, model, source_language, task, vad_filter, names_and_terms,
                                batch_size, progress_callback, cancel_check)

    if progress_callback: progress_callback(100, 100, "Saving text file...", "save")
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
            if progress_callback: progress_callback(0, 100, "Adding the subtitle track to the video...", "video")
            result["video_file"] = video.add_subtitle_track(media_path, output_file, subtitle_language,
                                                            cancel_check=cancel_check)
        else:
            if progress_callback: progress_callback(0, 100, "Burning subtitles to video... 0%", "video")

            def burn_progress(seconds):
                if progress_callback and info.duration:
                    percentage = min(100, round(seconds / info.duration * 100))
                    progress_callback(min(seconds, info.duration), info.duration,
                                      f"Burning subtitles to video... {percentage}%", "video")
            result["video_file"], used_gpu = video.burn_subtitles(media_path, output_file, cancel_check=cancel_check,
                                                                  on_progress=burn_progress)
            if not used_gpu:
                result["notes"].append("Burned in with the CPU encoder, because FFmpeg couldn't use the NVIDIA "
                                       "encoder (NVENC). It works, just more slowly.")

    if progress_callback: progress_callback(100, 100, "Success! File saved.", "done")
    return result
