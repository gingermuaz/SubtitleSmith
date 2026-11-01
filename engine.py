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
from types import SimpleNamespace
from faster_whisper import BatchedInferencePipeline, WhisperModel
from huggingface_hub.utils import tqdm as HfProgressBar

import languages
import speakers
import translation
import video
import writers
from cancellable import run_cancellable as _run_cancellable
from errors import ProcessCancelled, ProcessingError  # GUI.py catches ProcessCancelled as engine.ProcessCancelled

warnings.filterwarnings("ignore")

SUPPORTED_MEDIA = (
    '.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm',
    '.mp3', '.wav', '.aac', '.m4a', '.flac', '.ogg', '.wma'
)

# Hugging Face repositories for the models in the AI Model menu
MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
    "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
}
# The files faster-whisper needs from a model repository (the same list it downloads itself)
MODEL_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]

_loaded = {"key": None, "model": None}
_translator = {"key": None, "model": None}
_load_lock = threading.Lock()


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


def download_model(repo_id, files, what, progress_callback=None, cancel_check=None):
    """The local folder holding a model's files, downloading them from Hugging Face first if needed.

    files are filename patterns; what names the model in messages ("the 'large-v3' AI model").
    Downloads report progress (stage "download") and stop on Cancel; an interrupted download
    carries on from where it stopped next time.
    """
    try:
        folder = huggingface_hub.snapshot_download(repo_id, allow_patterns=files, local_files_only=True)
        if os.path.exists(os.path.join(folder, "model.bin")):
            return folder  # already downloaded
    except Exception:  # not downloaded yet, or only partly
        pass

    try:
        info = huggingface_hub.HfApi().model_info(repo_id, files_metadata=True)
    except Exception as e:
        raise ProcessingError(f"Couldn't download {what}. Each model downloads once, the first time you use "
                              f"it, so this needs an internet connection. ({e})") from e
    wanted = [(f.rfilename, f.size or 0) for f in info.siblings
              if any(fnmatch.fnmatch(f.rfilename, pattern) for pattern in files)]
    total = sum(size for _, size in wanted) or 1
    done = 0

    def on_bytes(n):
        nonlocal done
        done += n
        if cancel_check and cancel_check():
            raise ProcessCancelled()
        if progress_callback:
            progress_callback(min(done, total), total, f"Downloading {what} (first use only)... "
                                                       f"{_size(min(done, total))} of {_size(total)}", "download")

    path = None
    for filename, size in wanted:
        before = done
        try:
            path = huggingface_hub.hf_hub_download(repo_id, filename, tqdm_class=_download_progress_bar(on_bytes))
        except ProcessCancelled:
            raise
        except Exception as e:
            raise ProcessingError(f"Downloading {what} failed ({e}). Try again: it carries on from where it "
                                  "stopped.") from e
        done = before + size
    return os.path.dirname(path)


def model_folder(model_size, progress_callback=None, cancel_check=None):
    """The local folder holding a Whisper model's files, downloading them first if needed."""
    return download_model(MODEL_REPOS.get(model_size, model_size), MODEL_FILES, f"the '{model_size}' AI model",
                          progress_callback, cancel_check)


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


def get_translator(translator=translation.DEFAULT, compute_device="auto", device_index=0, progress_callback=None,
                   cancel_check=None):
    """An NLLB-200 translation model ("fast", "better" or "best"), downloaded on first use and loaded once."""
    if translator not in translation.MODELS:
        raise ValueError(f"Unknown translation model: {translator}")
    key = (translator, compute_device, device_index)
    if _translator["key"] == key:
        return _translator["model"]
    name = f"the '{translator.capitalize()}' translation model"
    folder = download_model(translation.MODELS[translator], translation.FILES, name, progress_callback, cancel_check)
    if progress_callback:
        progress_callback(0, 100, f"Loading {name}...", "load")

    def load():
        with _load_lock:
            if _translator["key"] != key:
                # Drop the old one first so two translation models never sit in memory at once
                _translator["key"] = None
                _translator["model"] = None
                gc.collect()
                _translator["model"] = translation.Translator(folder, compute_device, device_index)
                _translator["key"] = key
            return _translator["model"]
    return _run_cancellable(load, cancel_check)


def prepare(model_size, compute_device="auto", device_index=0, translate=False, speaker_labels=False,
            translator=translation.DEFAULT, progress_callback=None, cancel_check=None):
    """Downloads and loads everything a queue needs before its first file, so problems show up once."""
    get_model(model_size, compute_device, device_index, progress_callback, cancel_check)
    if translate:
        get_translator(translator, compute_device, device_index, progress_callback, cancel_check)
    if speaker_labels:
        speakers.get_pipeline(compute_device != "cpu", progress_callback, cancel_check)


def extra_output_path(path, language):
    """'film_subtitle.srt', 'en' -> 'film_subtitle.en.srt'"""
    base, ext = os.path.splitext(path)
    return f"{base}.{language}{ext}"


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
                 also_english=False, translate_to=None, translator=translation.DEFAULT, speaker_labels=False,
                 num_speakers=None, burn_style=None, review=None, progress_callback=None, cancel_check=None):
    """Makes subtitles for one media file and, if asked, adds them to a copy of the video.

    task: "translate" (English subtitles) or "transcribe" (the spoken language).
    also_english: also save an English translation next to it (<name>.en.srt).
    translate_to: also save a translation into this language, e.g. "fr" (<name>.fr.srt).
    translator: which size of translation model those use: "fast", "better" or "best".
    speaker_labels: start a subtitle with "Speaker 1:" etc. whenever the speaker changes.
    video_output: "none", "track" (every language as a subtitle track viewers can switch on and off)
    or "burn" (the main subtitles drawn into the picture, styled by burn_style).
    batch_size > 1 turns on Fast Mode (see transcribe).
    review(texts) is called with {path: text} before anything is saved and returns the texts to
    save (edited or not), or None to cancel.

    Returns {"subtitle_file", "extra_files", "extra_languages", "video_file", "notes", "language", "language_probability",
    "language_detected"}. Raises ProcessCancelled if the user cancels, and ProcessingError (or
    faster-whisper's own errors) if something goes wrong.
    """
    if video_output not in ("none", "track", "burn"):
        raise ValueError(f"Unknown video output: {video_output}")
    model = get_model(model_size, compute_device, device_index, progress_callback, cancel_check)
    segments, info = transcribe(media_path, model, source_language, task, vad_filter, names_and_terms,
                                batch_size, progress_callback, cancel_check)
    # Translations come out in English; otherwise the subtitles are in the spoken language
    language = "en" if task == "translate" else info.language
    notes = []
    if source_language == "auto" and info.language_probability < 0.5:
        notes.append(f"Not sure which language this is (best guess "
                     f"{languages.describe(info.language, info.language_probability)}). "
                     "If the subtitles look wrong, choose the Source Language yourself.")

    if speaker_labels:
        try:
            turns = speakers.find_speakers(media_path, compute_device != "cpu", num_speakers, progress_callback,
                                           cancel_check)
            segments = speakers.assign(segments, turns)
        except ProcessingError as e:
            notes.append(f"No speaker labels: {e}")

    # The main subtitles, then any translations of them
    outputs = [(output_file, language, segments)]
    for target in (["en"] if also_english else []) + ([translate_to] if translate_to else []):
        if any(target == existing for _, existing, _ in outputs):
            if target == "en" and also_english and language == "en":
                notes.append("It's already in English, so there's no separate English file.")
            continue
        try:
            nllb = get_translator(translator, compute_device, device_index, progress_callback, cancel_check)
            name = languages.name(target)

            def translate_progress(done, total, name=name):
                if progress_callback:
                    progress_callback(done, total, f"Translating subtitles into {name}... "
                                                   f"{round(done / total * 100)}%", "translate")
            texts = nllb.translate([segment.text for segment in segments], language, target, cancel_check,
                                   translate_progress)
        except ProcessingError as e:
            notes.append(f"No {languages.name(target)} subtitles: {e}")
            continue
        translated = [SimpleNamespace(start=segment.start, end=segment.end, text=text, words=None,
                                      speaker=getattr(segment, "speaker", None))
                      for segment, text in zip(segments, texts)]
        outputs.append((extra_output_path(output_file, target), target, translated))

    texts = {}
    for path, lang, lines in outputs:
        if speaker_labels:
            lines = speakers.with_labels(lines)
        texts[path] = writers.render(lines, output_format, readable_lines, languages.line_length(lang))
    if review:
        texts = review(texts)
        if texts is None:
            raise ProcessCancelled()

    if progress_callback: progress_callback(100, 100, "Saving text file...", "save")
    for path, text in texts.items():
        writers.save_text(path, text)
    result = {"subtitle_file": output_file, "extra_files": [path for path, _, _ in outputs[1:]],
              "extra_languages": [lang for _, lang, _ in outputs[1:]], "video_file": None,
              "notes": notes, "language": info.language, "language_probability": info.language_probability,
              "language_detected": source_language == "auto"}

    if video_output != "none":
        if output_format not in (".srt", ".vtt"):
            notes.append("Video skipped: .txt files have no timings to add to a video.")
        elif not video.is_supported_video(media_path):
            notes.append("Video skipped: only MP4, MKV, AVI and MOV videos are supported.")
        elif video_output == "track":
            if progress_callback: progress_callback(0, 100, "Adding the subtitle track to the video...", "video")
            result["video_file"] = video.add_subtitle_tracks(media_path, [(path, lang) for path, lang, _ in outputs],
                                                             cancel_check=cancel_check)
        else:
            if progress_callback: progress_callback(0, 100, "Burning subtitles to video... 0%", "video")

            def burn_progress(seconds):
                if progress_callback and info.duration:
                    percentage = min(100, round(seconds / info.duration * 100))
                    progress_callback(min(seconds, info.duration), info.duration,
                                      f"Burning subtitles to video... {percentage}%", "video")
            result["video_file"], used_gpu = video.burn_subtitles(media_path, output_file, cancel_check=cancel_check,
                                                                  on_progress=burn_progress, style=burn_style)
            if not used_gpu:
                notes.append("Burned in with the CPU encoder, because FFmpeg couldn't use the NVIDIA "
                             "encoder (NVENC). It works, just more slowly.")

    if progress_callback: progress_callback(100, 100, "Success! File saved.", "done")
    return result
