"""Turns transcribed segments into .srt, .vtt or .txt text.

A segment is anything with .start and .end (seconds) and .text, plus .words (items with
.start, .end and .word) when word timestamps are on, such as faster-whisper's segments.
"""

FORMATS = (".srt", ".vtt", ".txt")


def format_timestamp(seconds: float, is_vtt=False):
    """Formats time to SRT (00:00:00,000) or VTT (00:00:00.000) standards."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    separator = "." if is_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _cues(segments, word_timestamps):
    """Yields (start, end, text) for each subtitle block."""
    for segment in segments:
        if word_timestamps and segment.words:
            # Output individual words as their own subtitle block
            for word in segment.words:
                yield word.start, word.end, word.word.strip()
        else:
            # Standard sentence-level subtitle block
            yield segment.start, segment.end, segment.text.strip()


def render(segments, output_format=".srt", word_timestamps=False):
    """Returns the whole file's text in the given format."""
    if output_format == ".txt":
        # Standard text ignores timestamps completely
        return "".join(f"{segment.text.strip()}\n" for segment in segments)
    if output_format not in (".srt", ".vtt"):
        raise ValueError(f"Unknown subtitle format: {output_format}")

    is_vtt = output_format == ".vtt"
    blocks = ["WEBVTT\n\n"] if is_vtt else []
    for index, (start, end, text) in enumerate(_cues(segments, word_timestamps), start=1):
        blocks.append(f"{index}\n"
                      f"{format_timestamp(start, is_vtt)} --> {format_timestamp(end, is_vtt)}\n"
                      f"{text}\n\n")
    return "".join(blocks)


def write(path, segments, output_format=".srt", word_timestamps=False):
    """Saves the segments to path as UTF-8 text."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(segments, output_format, word_timestamps))
