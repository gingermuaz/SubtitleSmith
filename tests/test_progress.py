import unittest

import progress
from progress import Eta, QueueItem


class QueueLineTests(unittest.TestCase):
    def test_columns_line_up(self):
        short = progress.queue_columns(QueueItem("C:/films/a.mp4", "done", "3 min 2 s"))
        long = progress.queue_columns(QueueItem("C:/films/" + "x" * 80 + ".mkv", "working", "42%"))
        self.assertEqual(short[0], "\u2713 Done     ")
        self.assertEqual(len(short[0]), len(long[0]))
        self.assertEqual(len(short[1]), len(long[1]))
        self.assertTrue(long[1].rstrip().endswith("\u2026"))  # long names are shortened
        self.assertEqual(long[2], " 42%")

    def test_every_status_has_a_label(self):
        for status in ("waiting", "working", "done", "failed", "canceled"):
            self.assertIn(status.capitalize(), progress.queue_columns(QueueItem("f.mp4", status))[0])


class FormattingTests(unittest.TestCase):
    def test_durations(self):
        self.assertEqual([progress.format_duration(s) for s in (12, 95, 3725)], ["12 s", "1 min 35 s", "1 h 2 min"])

    def test_time_left(self):
        self.assertEqual(progress.format_time_left(40), "less than a minute left")
        self.assertEqual(progress.format_time_left(9 * 60 + 20), "about 9 min left")
        self.assertEqual(progress.format_time_left(2 * 3600 + 5 * 60), "about 2 h 5 min left")


class EtaTests(unittest.TestCase):
    def test_no_estimate_until_there_is_something_to_go_on(self):
        eta = Eta([600, 600])
        self.assertIsNone(eta.seconds_left(0))  # not started
        eta.start(0)
        eta.progress(0, 0.5)
        self.assertIsNone(eta.seconds_left(2))  # under 5 seconds in

    def test_estimate_from_speed_so_far(self):
        # 3 files of 10 minutes; the first took 100 s, the second is half done after 50 s more
        eta = Eta([600, 600, 600])
        eta.start(0)
        eta.file_finished(0)
        eta.progress(1, 0.5)
        self.assertAlmostEqual(eta.seconds_left(150), 150)  # 900 media s left at 6 media s per second

    def test_unknown_lengths_count_as_average(self):
        eta = Eta([100, None, 300])
        self.assertEqual(eta.durations, [100, 200, 300])

    def test_failed_and_finished_files_both_count_as_done(self):
        eta = Eta([60, 60])
        eta.start(0)
        eta.file_finished(0)
        self.assertAlmostEqual(eta.seconds_left(10), 10)


if __name__ == "__main__":
    unittest.main()
