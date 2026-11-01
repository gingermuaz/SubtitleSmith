"""Remembers the user's choices between launches, in a small JSON file in their settings folder."""
import json
import os


def default_path():
    """%APPDATA%\\SubtitleSmith\\settings.json on Windows, ~/.config/SubtitleSmith/settings.json elsewhere."""
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "SubtitleSmith", "settings.json")


def load(path=None):
    """The saved settings, or {} if there are none yet or the file can't be read."""
    try:
        with open(path or default_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(values, path=None):
    """Saves the settings. Failing to save is ignored: remembering settings is only a convenience."""
    path = path or default_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(values, f, indent=2, ensure_ascii=False)
        os.replace(temp, path)  # all or nothing, so a crash can't leave a half-written file
    except (OSError, TypeError, ValueError):
        pass
