"""Turns transcribed segments into .srt, .vtt or .txt text.

A segment is anything with .start and .end (seconds), .text and .words: a list of items with
.start, .end and .word, or None. faster-whisper's segments look like this.
"""
import re
from dataclasses import dataclass

FORMATS = (".srt", ".vtt", ".txt")

# Readable-subtitle rules
MIN_DURATION = 1.0  # seconds on screen, if the next subtitle leaves room
MAX_DURATION = 7.0

_SENTENCE_ENDS = (".", "?", "!", "…", "。", "？", "！")  # . ? ! ... and the CJK forms
_CLAUSE_ENDS = (",", ";", ":", "，", "；", "：", "、")
_CLOSING_MARKS = "\"')]”’」"


@dataclass
class Cue:
    """One subtitle: when it shows and its lines of text."""
    start: float
    end: float
    lines: list


@dataclass
class _Word:
    start: float
    end: float
    word: str


def format_timestamp(seconds, is_vtt=False):
    """Formats time to SRT (00:00:00,000) or VTT (00:00:00.000) standards."""
    total_ms = max(0, round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    separator = "." if is_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


# ---------------------------------------------------------------- readable lines

def _join(words):
    # Each faster-whisper word carries its own leading space (none for Chinese/Japanese)
    return "".join(w.word for w in words).strip()


def _break_bonus(words, i):
    """How good the gap before words[i] is for a break: sentence ends, then commas, then pauses."""
    before = words[i - 1].word.strip().rstrip(_CLOSING_MARKS)
    bonus = 0.0
    if before.endswith(_SENTENCE_ENDS):
        bonus += 0.5
    elif before.endswith(_CLAUSE_ENDS):
        bonus += 0.25
    pause = words[i].start - words[i - 1].end
    return bonus + min(max(pause, 0.0), 2.0) / 4  # up to 0.5 more for a 2-second pause


def _wrap(words, max_chars):
    """Lays a subtitle out as 1 or 2 lines of at most max_chars, or returns None if it won't fit."""
    text = _join(words)
    if len(text) <= max_chars:
        return [text]
    best = None
    for i in range(1, len(words)):
        top, bottom = _join(words[:i]), _join(words[i:])
        if len(top) > max_chars or len(bottom) > max_chars:
            continue
        score = abs(len(top) - len(bottom)) / max_chars - _break_bonus(words, i)
        if best is None or score < best[0]:
            best = (score, [top, bottom])
    return best[1] if best else None


def _split(words, max_chars):
    """Splits one segment's words into groups that each fit on screen, choosing balanced break points."""
    if len(words) <= 1 or (words[-1].end - words[0].start <= MAX_DURATION and _wrap(words, max_chars)):
        return [words]
    total = len(_join(words))
    best_i, best_score = 1, None
    for i in range(1, len(words)):
        balance = abs(len(_join(words[:i])) - len(_join(words[i:]))) / total
        score = balance - _break_bonus(words, i)
        if best_score is None or score < best_score:
            best_i, best_score = i, score
    return _split(words[:best_i], max_chars) + _split(words[best_i:], max_chars)


def _words_of(segment):
    """The segment's timed words. If faster-whisper gave none, spread the segment's time over its words."""
    words = [w for w in (segment.words or []) if w.word.strip()]
    if words:
        return words
    tokens = segment.text.split()
    if not tokens:
        return []
    weights = [len(token) + 1 for token in tokens]
    step = (segment.end - segment.start) / sum(weights)
    estimated, t = [], segment.start
    for token, weight in zip(tokens, weights):
        estimated.append(_Word(t, t + weight * step, " " + token))
        t += weight * step
    return estimated


def _fix_timing(cues):
    """Keeps each subtitle on screen 1-7 seconds without running into the next one."""
    for cue, next_cue in zip(cues, cues[1:] + [None]):
        cue.end = min(cue.end, cue.start + MAX_DURATION)  # only a single long-held word gets here
        if cue.end - cue.start < MIN_DURATION:
            cue.end = cue.start + MIN_DURATION
        if next_cue is not None and next_cue.start > cue.start:
            cue.end = min(cue.end, next_cue.start)


def make_cues(segments, max_chars=42):
    """Re-splits Whisper's segments into readable subtitles.

    Each subtitle has at most 2 lines of max_chars and stays on screen 1-7 seconds. Breaks go at
    sentence ends, commas and pauses where possible, and the two lines are kept roughly even.
    """
    cues = []
    for segment in segments:
        for group in _split(_words_of(segment), max_chars):
            if group:
                lines = _wrap(group, max_chars) or [_join(group)]  # a single word longer than a line
                cues.append(Cue(group[0].start, group[-1].end, lines))
    _fix_timing(cues)
    return cues


# ---------------------------------------------------------------- output

def render(segments, output_format=".srt", readable_lines=True, max_chars=42):
    """Returns the whole file's text in the given format.

    readable_lines=False gives one subtitle per Whisper segment, however long.
    """
    if output_format == ".txt":
        # Standard text ignores timestamps completely
        return "".join(f"{segment.text.strip()}\n" for segment in segments)
    if output_format not in (".srt", ".vtt"):
        raise ValueError(f"Unknown subtitle format: {output_format}")

    if readable_lines:
        cues = make_cues(segments, max_chars)
    else:
        cues = [Cue(segment.start, segment.end, [segment.text.strip()]) for segment in segments]

    cues = [cue for cue in cues if any(line.strip() for line in cue.lines)]  # nothing to show, so no subtitle
    is_vtt = output_format == ".vtt"
    blocks = ["WEBVTT\n\n"] if is_vtt else []
    for index, cue in enumerate(cues, start=1):
        blocks.append(f"{index}\n"
                      f"{format_timestamp(cue.start, is_vtt)} --> {format_timestamp(cue.end, is_vtt)}\n"
                      + "\n".join(cue.lines) + "\n\n")
    return "".join(blocks)


def write(path, segments, output_format=".srt", readable_lines=True, max_chars=42):
    """Saves the segments to path as UTF-8 text."""
    save_text(path, render(segments, output_format, readable_lines, max_chars))


def save_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


_SRT_TIMING = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})$")
_VTT_TIMING = re.compile(r"^(?:(\d{2,}):)?(\d{2}):(\d{2})\.(\d{3}) --> (?:(\d{2,}):)?(\d{2}):(\d{2})\.(\d{3})(?: .*)?$")


def _seconds(parts):
    hours, minutes, secs, millis = (int(p or 0) for p in parts)
    return hours * 3600 + minutes * 60 + secs + millis / 1000


def problems(text, output_format):
    """Mistakes in hand-edited subtitles that would stop players or FFmpeg reading them ([] if none)."""
    if output_format not in (".srt", ".vtt"):
        return []
    found = []
    blocks = [block for block in re.split(r"\n[ \t]*\n", text.replace("\r\n", "\n").strip()) if block.strip()]
    if output_format == ".vtt":
        if not blocks or not blocks[0].startswith("WEBVTT"):
            found.append("The first line must be WEBVTT.")
        blocks = blocks[1:]
    timing_form = "00:00:01,000 --> 00:00:02,500" if output_format == ".srt" else "00:00:01.000 --> 00:00:02.500"
    pattern = _SRT_TIMING if output_format == ".srt" else _VTT_TIMING
    for n, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.split("\n")]
        timing_index = next((i for i, line in enumerate(lines[:2]) if "-->" in line), None)
        if timing_index is None:
            found.append(f"Subtitle {n} has no timing line (like {timing_form}).")
            continue
        match = pattern.match(lines[timing_index])
        if not match:
            found.append(f"Subtitle {n}: the timing line \"{lines[timing_index]}\" should look like {timing_form}.")
        elif _seconds(match.groups()[4:8]) < _seconds(match.groups()[:4]):
            found.append(f"Subtitle {n} ends before it starts.")
        if output_format == ".srt" and (timing_index != 1 or not lines[0].isdigit()):
            found.append(f"Subtitle {n} should start with its number on its own line.")
        if timing_index == len(lines) - 1:
            found.append(f"Subtitle {n} has no text.")
    return found
