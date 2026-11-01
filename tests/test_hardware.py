import unittest
from unittest import mock

import hardware

ALL_MODELS = {"tiny", "base", "small", "medium", "distil-large-v3", "large-v3-turbo", "large-v3"}


def fake_nvidia_smi(output=None, error=None):
    """Replaces subprocess.check_output: answers nvidia-smi, fails everything else (e.g. PowerShell)."""
    def check_output(cmd, **kwargs):
        if cmd[0] != "nvidia-smi" or error:
            raise error or FileNotFoundError(cmd[0])
        return output
    return mock.patch.object(hardware.subprocess, "check_output", check_output)


class GpuDetectionTests(unittest.TestCase):
    def test_two_gpus_uses_the_biggest(self):
        with fake_nvidia_smi("0, NVIDIA GeForce RTX 3080, 10240\n1, NVIDIA GeForce RTX 3060, 12288\n"):
            self.assertEqual(len(hardware.get_nvidia_gpus()), 2)
            scan = hardware.scan_system()
        self.assertEqual(scan["gpu_index"], 1)
        self.assertIn("RTX 3060 (12 GB VRAM) - used for processing", scan["report"])
        self.assertIn("RTX 3080 (10 GB VRAM)", scan["report"])

    def test_no_nvidia_smi(self):
        with fake_nvidia_smi(error=FileNotFoundError("nvidia-smi")):
            scan = hardware.scan_system()
        self.assertEqual(scan["gpu_index"], 0)
        self.assertIn("No NVIDIA GPU detected", scan["report"])

    def test_unreadable_line_is_skipped(self):
        with fake_nvidia_smi("0, GPU A, [N/A]\n1, GPU B, 8192\n"):
            self.assertEqual(hardware.get_nvidia_gpus(), [(1, "GPU B", 8192)])


class RecommendationTests(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(hardware.recommend_models(True, 12, 32), {"transcribe": "large-v3-turbo", "translate": "large-v3"})
        self.assertEqual(hardware.recommend_models(True, 6, 16), {"transcribe": "large-v3-turbo", "translate": "medium"})
        self.assertEqual(hardware.recommend_models(False, 0, 8), {"transcribe": "tiny", "translate": "tiny"})

    def test_every_pick_can_do_its_job(self):
        for has_gpu, vram, ram in [(True, 2, 8), (True, 4, 8), (True, 6, 8), (True, 8, 8), (True, 24, 64),
                                   (False, 0, 8), (False, 0, 32)]:
            for task, code in hardware.recommend_models(has_gpu, vram, ram).items():
                self.assertIn(code, ALL_MODELS)
                self.assertIsNone(hardware.model_job_problem(code, task, "ja"), (vram, task, code))

    def test_model_job_problem(self):
        problem = hardware.model_job_problem
        self.assertIsNotNone(problem("large-v3-turbo", "translate", "ja"))
        self.assertIsNone(problem("large-v3-turbo", "translate", "en"))  # English audio is already English
        self.assertIsNone(problem("large-v3-turbo", "transcribe", "ja"))
        self.assertIn("English", problem("distil-large-v3", "translate", "auto"))
        self.assertIsNotNone(problem("distil-large-v3", "transcribe", "ja"))
        self.assertIsNone(problem("distil-large-v3", "transcribe", "auto"))
        self.assertIsNone(problem("large-v3", "translate", "ja"))


if __name__ == "__main__":
    unittest.main()
