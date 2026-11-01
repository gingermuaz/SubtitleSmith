import json
import os
import tempfile
import unittest

import settings


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.path = os.path.join(self._temp.name, "SubtitleSmith", "settings.json")

    def test_round_trip(self):
        values = {"model": "large-v3 (Slowest / Best Accuracy)", "vad_filter": False, "names_and_terms": "Frodo, Mordor"}
        settings.save(values, self.path)
        self.assertEqual(settings.load(self.path), values)

    def test_nothing_saved_yet(self):
        self.assertEqual(settings.load(self.path), {})

    def test_damaged_file_is_ignored(self):
        os.makedirs(os.path.dirname(self.path))
        for content in ("{not json", "[1, 2, 3]"):
            with open(self.path, "w") as f:
                f.write(content)
            self.assertEqual(settings.load(self.path), {})

    def test_failing_to_save_is_not_an_error(self):
        blocker = os.path.join(self._temp.name, "a file, not a folder")
        open(blocker, "w").close()
        settings.save({"x": 1}, os.path.join(blocker, "settings.json"))  # can't create the folder: no exception

    def test_saved_file_is_readable_json(self):
        settings.save({"names_and_terms": "Ōkami, 日本"}, self.path)
        with open(self.path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["names_and_terms"], "Ōkami, 日本")

    def test_default_location(self):
        self.assertTrue(settings.default_path().endswith(os.path.join("SubtitleSmith", "settings.json")))


if __name__ == "__main__":
    unittest.main()
