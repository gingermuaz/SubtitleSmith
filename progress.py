"""The queue list and the time-left estimate. No window code here, so it's easy to test."""
import os
from dataclasses import dataclass

_LABELS = {"waiting": "Waiting", "working": "Working", "done": "Done", "failed": "Failed", "canceled": "Canceled"}
_ICONS = {"waiting": "•", "working": "▶", "done": "✓", "failed": "✗", "canceled": "–"}
NAME_WIDTH = 34


@dataclass
class QueueItem:
    """One file in the queue. status is waiting, working, done, failed or canceled."""
    path: str
    status: str = "waiting"
    detail: str = ""

    @property
    def name(self):
        return os.path.basename(self.path)


def queue_columns(item):
    """(status column, file name column, detail) for one line of the queue list, padded to line up."""
    name = item.name if len(item.name) <= NAME_WIDTH else item.name[:NAME_WIDTH - 1] + "…"
    return f"{_ICONS[item.status]} {_LABELS[item.status]:<9}", f" {name:<{NAME_WIDTH}}", f" {item.detail}"


def format_duration(seconds):
    """95 -> '1 min 35 s', 3725 -> '1 h 2 min'"""
    seconds = max(0, int(round(seconds)))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def format_time_left(seconds):
    if seconds < 60:
        return "less than a minute left"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {minutes} min left"
    return f"about {minutes // 60} h {minutes % 60} min left"


class Eta:
    """Estimates the time left for a queue from how fast it has gone so far.

    It works in seconds of media: durations[i] is file i's length, or None if unknown (an unknown
    length counts as the average of the known ones). Time spent burning subtitles in counts towards
    the speed, so later estimates include it.
    """

    def __init__(self, durations):
        known = [d for d in durations if d]
        fallback = sum(known) / len(known) if known else 60.0
        self.durations = [d if d else fallback for d in durations]
        self.total = sum(self.durations)
        self.started = None
        self.finished = 0.0  # media seconds in finished files
        self.current = 0.0  # media seconds done in the file being worked on

    def start(self, now):
        """Call when the first file starts (after the model has loaded)."""
        if self.started is None:
            self.started = now

    def progress(self, index, fraction):
        self.current = self.durations[index] * max(0.0, min(1.0, fraction))

    def file_finished(self, index):
        self.finished += self.durations[index]
        self.current = 0.0

    def seconds_left(self, now):
        """None until there's enough to go on: at least 5 seconds and 1% of the queue done."""
        if self.started is None:
            return None
        done = self.finished + self.current
        elapsed = now - self.started
        if elapsed < 5 or done <= 0 or done < 0.01 * self.total:
            return None
        return max(0.0, elapsed * (self.total - done) / done)
