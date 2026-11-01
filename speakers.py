"""Speaker labels ("Speaker 1:", "Speaker 2:") using pyannote.audio's speaker-diarization-community-1 model.

Optional: it needs the packages in requirements-speakers.txt (PyTorch and pyannote.audio) and a free
Hugging Face account that has accepted the model's terms. pyannote's usage reporting is switched off,
so apart from the one-time model download nothing leaves the computer.
"""
import bisect
import importlib
import importlib.util
import os
import threading
from types import SimpleNamespace

import huggingface_hub

from cancellable import run_cancellable
from errors import ProcessCancelled, ProcessingError

MODEL = "pyannote/speaker-diarization-community-1"
TERMS_URL = "https://huggingface.co/pyannote/speaker-diarization-community-1"
TOKEN_URL = "https://huggingface.co/settings/tokens"
INSTALL_COMMAND = "python -m pip install -r requirements-speakers.txt"

# pyannote reports usage to its own servers unless this is set before it's imported
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

_loaded = {"gpu": None, "pipeline": None}
_lock = threading.Lock()


def available():
    """True if pyannote.audio is installed."""
    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except (ImportError, ValueError):
        return False


def has_token():
    """True if a Hugging Face access token is saved (or set in HF_TOKEN)."""
    return bool(huggingface_hub.get_token())


def save_token(token):
    """Checks a Hugging Face access token and saves it where Hugging Face tools look for it."""
    huggingface_hub.login(token=token.strip(), add_to_git_credential=False, skip_if_logged_in=False)


def _access_problem(error=None):
    detail = f" ({error})" if error else ""
    return ProcessingError(f"Couldn't load pyannote's speaker model{detail}. Check that the Hugging Face account "
                           f"your token belongs to has accepted the model's terms at {TERMS_URL}, then try again.")


def get_pipeline(use_gpu=True, progress_callback=None, cancel_check=None):
    """pyannote's speaker model, downloaded on first use and loaded once."""
    if _loaded["pipeline"] is not None and _loaded["gpu"] == use_gpu:
        return _loaded["pipeline"]
    if not available():
        raise ProcessingError(f"Speaker labels need extra packages. Install them with: {INSTALL_COMMAND}")
    if not has_token():
        raise ProcessingError(f"Speaker labels need a Hugging Face access token. Create one at {TOKEN_URL} "
                              f"after accepting the model's terms at {TERMS_URL}.")
    if progress_callback:
        progress_callback(0, 100, "Loading the speaker model (it downloads on first use)...", "load")

    def load():
        with _lock:
            torch = importlib.import_module("torch")
            pyannote_audio = importlib.import_module("pyannote.audio")
            try:
                pipeline = pyannote_audio.Pipeline.from_pretrained(MODEL, token=huggingface_hub.get_token())
            except Exception as e:
                raise _access_problem(e) from e
            if pipeline is None:  # pyannote returns None when the download isn't allowed
                raise _access_problem()
            if use_gpu and torch.cuda.is_available():
                pipeline.to(torch.device("cuda"))
            _loaded.update(gpu=use_gpu, pipeline=pipeline)
            return pipeline
    return run_cancellable(load, cancel_check)


def find_speakers(media_path, use_gpu=True, num_speakers=None, progress_callback=None, cancel_check=None):
    """Who speaks when: [(start, end, speaker), ...], with overlapping speech already resolved.

    num_speakers can be given if known; otherwise pyannote works it out.
    """
    pipeline = get_pipeline(use_gpu, progress_callback, cancel_check)
    if progress_callback:
        progress_callback(0, 100, "Finding speakers...", "speakers")

    def hook(step_name, step_artifact, file=None, total=None, completed=None):
        if cancel_check and cancel_check():
            raise ProcessCancelled()  # stops pyannote at its next step
        if progress_callback and total:
            done = completed or 0
            progress_callback(done, total, f"Finding speakers... {round(done / total * 100)}%", "speakers")

    def run():
        torch = importlib.import_module("torch")
        from faster_whisper import decode_audio
        audio = decode_audio(media_path, sampling_rate=16000)  # pyannote gets the audio itself, so it needs no decoder
        output = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": 16000},
                          hook=hook, num_speakers=num_speakers)
        # pyannote 4 gives speech turns without overlaps, made for matching to subtitles
        annotation = getattr(output, "exclusive_speaker_diarization", output)
        return [(turn.start, turn.end, label) for turn, _, label in annotation.itertracks(yield_label=True)]

    try:
        return run_cancellable(run, cancel_check)
    except (ProcessCancelled, ProcessingError):
        raise
    except Exception as e:
        raise ProcessingError(f"finding speakers failed ({e})") from e


def assign(segments, turns):
    """Splits segments where the speaker changes; each part gets .speaker: 1, 2, ... in order of appearance.

    Each word goes to the speaker talking most during it (or the nearest one if nobody is). Single
    short words don't start a new speaker, as that's usually the speaker detection wobbling.
    """
    if not turns:
        return list(segments)
    turns = sorted(turns)
    starts = [start for start, _, _ in turns]
    numbers = {}

    def speaker_at(start, end):
        i = bisect.bisect_right(starts, end)
        nearby = turns[max(0, i - 4):i + 1]
        best = max(nearby, key=lambda turn: min(end, turn[1]) - max(start, turn[0]))
        if min(end, best[1]) - max(start, best[0]) <= 0:  # nobody speaking: the nearest turn
            middle = (start + end) / 2
            best = min(nearby, key=lambda turn: min(abs(turn[0] - middle), abs(turn[1] - middle)))
        return best[2]

    result = []
    for segment in segments:
        words = list(segment.words or [])
        if not words:
            result.append((speaker_at(segment.start, segment.end), segment.start, segment.end, segment.text, None))
            continue
        runs = []
        for word in words:
            label = speaker_at(word.start, word.end)
            if runs and runs[-1][0] == label:
                runs[-1][1].append(word)
            else:
                runs.append((label, [word]))
        # smooth out one-word wobbles: merge a lone short word into the run before it
        smoothed = []
        for label, run in runs:
            if smoothed and len(run) == 1 and run[0].end - run[0].start < 0.6:
                smoothed[-1][1].extend(run)
            elif smoothed and smoothed[-1][0] == label:
                smoothed[-1][1].extend(run)
            else:
                smoothed.append((label, list(run)))
        for label, run in smoothed:
            result.append((label, run[0].start, run[-1].end, "".join(w.word for w in run), run))

    return [SimpleNamespace(start=start, end=end, text=text, words=words,
                            speaker=numbers.setdefault(label, len(numbers) + 1))
            for label, start, end, text, words in result]


def with_labels(segments):
    """Copies of the segments with "Speaker N:" in front whenever the speaker changes."""
    labelled, previous = [], None
    for segment in segments:
        speaker = getattr(segment, "speaker", None)
        text, words = segment.text, segment.words
        if speaker is not None and speaker != previous:
            label = f"Speaker {speaker}:"
            text = f" {label} {segment.text.strip()}"
            if words:
                first = words[0]
                words = [SimpleNamespace(start=first.start, end=first.end, word=f" {label} {first.word.strip()}")]
                words += list(segment.words[1:])
            previous = speaker
        labelled.append(SimpleNamespace(start=segment.start, end=segment.end, text=text, words=words, speaker=speaker))
    return labelled
