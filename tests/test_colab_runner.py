"""
Automated unit tests for Google Colab CLI cloud orchestrator and repository packaging.
Tests:
1. scripts/create_colab_bundle.py: Archive generation, source inclusion, and exclusion of
   large photo directories (range_photos/, autolabeled_photos/), venvs, caches, and weights.
2. colab_runner.py: Binary resolution, usage/compute unit parsing, session URL discovery,
   session listing, stop session handlers, and progressive resolution stage parsing.
"""

import io
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

from scripts.create_colab_bundle import EXCLUDE_DIRS, EXCLUDE_EXTS, make_colab_bundle
from target_scorer.colab_runner import (
    check_colab_usage,
    get_colab_bin,
    get_session_url,
    list_colab_sessions,
    stop_colab_session,
)
from target_scorer.train import parse_progressive_stages


class TestColabPackaging(unittest.TestCase):
    """Verifies that create_colab_bundle packages source code while excluding heavy assets."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_repo = os.path.join(self.temp_dir, "mock_project")
        os.makedirs(self.mock_repo, exist_ok=True)

        # Create essential source tree
        os.makedirs(os.path.join(self.mock_repo, "src", "target_scorer"), exist_ok=True)
        with open(os.path.join(self.mock_repo, "src", "target_scorer", "train.py"), "w") as f:
            f.write("# training module\n")
        with open(os.path.join(self.mock_repo, "src", "target_scorer", "__init__.py"), "w") as f:
            f.write("# init\n")

        # Create tests
        os.makedirs(os.path.join(self.mock_repo, "tests"), exist_ok=True)
        with open(os.path.join(self.mock_repo, "tests", "test_sample.py"), "w") as f:
            f.write("# test\n")

        # Create directories that MUST be excluded
        os.makedirs(os.path.join(self.mock_repo, "range_photos"), exist_ok=True)
        with open(os.path.join(self.mock_repo, "range_photos", "photo1.jpg"), "w") as f:
            f.write("large photo dummy data")

        os.makedirs(os.path.join(self.mock_repo, "autolabeled_photos"), exist_ok=True)
        with open(os.path.join(self.mock_repo, "autolabeled_photos", "photo2.png"), "w") as f:
            f.write("large auto-labeled photo dummy")

        os.makedirs(os.path.join(self.mock_repo, "target_env", "bin"), exist_ok=True)
        with open(os.path.join(self.mock_repo, "target_env", "bin", "python"), "w") as f:
            f.write("python binary dummy")

        os.makedirs(os.path.join(self.mock_repo, ".git", "objects"), exist_ok=True)
        with open(os.path.join(self.mock_repo, ".git", "HEAD"), "w") as f:
            f.write("ref: refs/heads/main")

        os.makedirs(os.path.join(self.mock_repo, "__pycache__"), exist_ok=True)
        with open(os.path.join(self.mock_repo, "__pycache__", "module.cpython-311.pyc"), "w") as f:
            f.write("pyc bytecode")

        os.makedirs(os.path.join(self.mock_repo, "weights"), exist_ok=True)
        with open(os.path.join(self.mock_repo, "weights", "model.pt"), "w") as f:
            f.write("large model weights checkpoint")

        self.output_zip = os.path.join(self.temp_dir, "test_bundle.zip")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_bundle_creation_and_exclusion_rules(self):
        """Verifies bundle creation accurately captures src while strictly filtering heavy assets."""
        make_colab_bundle(output_path=self.output_zip, project_root=self.mock_repo)
        self.assertTrue(os.path.exists(self.output_zip))

        with zipfile.ZipFile(self.output_zip, "r") as zf:
            namelist = zf.namelist()

            # Verify essential sources are bundled with target_scorer prefix
            self.assertIn("target_scorer/src/target_scorer/train.py", namelist)
            self.assertIn("target_scorer/src/target_scorer/__init__.py", namelist)
            self.assertIn("target_scorer/tests/test_sample.py", namelist)

            # Verify strictly excluded directories
            self.assertFalse(any("range_photos" in name for name in namelist))
            self.assertFalse(any("autolabeled_photos" in name for name in namelist))
            self.assertFalse(any("target_env" in name for name in namelist))
            self.assertFalse(any(".git" in name for name in namelist))
            self.assertFalse(any("__pycache__" in name for name in namelist))
            self.assertFalse(any("weights" in name for name in namelist))

            # Verify strictly excluded extensions
            self.assertFalse(any(name.endswith(".pyc") for name in namelist))
            self.assertFalse(any(name.endswith(".pt") for name in namelist))

    def test_exclude_sets_completeness(self):
        """Verifies that EXCLUDE_DIRS and EXCLUDE_EXTS cover critical space-consuming folders."""
        for required_dir in ["range_photos", "autolabeled_photos", ".git", "target_env", "weights", "__pycache__"]:
            self.assertIn(required_dir, EXCLUDE_DIRS)

        for required_ext in [".pyc", ".pt", ".pth", ".tmp"]:
            self.assertIn(required_ext, EXCLUDE_EXTS)


class TestColabRunnerCLI(unittest.TestCase):
    """Verifies colab_runner command-line parsing, output decoding, and session state management."""

    @patch("shutil.which")
    def test_get_colab_bin_from_path(self, mock_which):
        """Verifies get_colab_bin returns path when found in PATH."""
        mock_which.return_value = "/usr/bin/colab"
        self.assertEqual(get_colab_bin(), "/usr/bin/colab")

    @patch("shutil.which", return_value=None)
    @patch("os.path.exists")
    def test_get_colab_bin_from_home_bin(self, mock_exists, mock_which):
        """Verifies fallback to ~/.local/bin/colab when not in PATH."""
        mock_exists.side_effect = lambda p: p.endswith(".local/bin/colab")
        result = get_colab_bin()
        self.assertTrue(result.endswith(".local/bin/colab"))

    @patch("shutil.which", return_value=None)
    @patch("os.path.exists", return_value=False)
    def test_get_colab_bin_raises_when_missing(self, mock_exists, mock_which):
        """Verifies FileNotFoundError raised when colab binary is nowhere to be found."""
        with self.assertRaises(FileNotFoundError):
            get_colab_bin()

    @patch("subprocess.check_output")
    def test_check_colab_usage_parsing(self, mock_check_output):
        """Verifies usage query parses compute unit balances without errors."""
        mock_check_output.return_value = (
            "Colab Pro+ Account\n"
            "Compute Units: 84.5 remaining\n"
            "Consumption Rate: ~2.1 CU/hour (A100)\n"
        )
        # Should execute cleanly without raising
        check_colab_usage(colab_bin="/fake/colab")
        mock_check_output.assert_called_once_with(["/fake/colab", "usage"], text=True, stderr=subprocess.STDOUT)

    @patch("subprocess.check_output")
    def test_check_colab_usage_handles_failure(self, mock_check_output):
        """Verifies usage query handles subprocess failures gracefully."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, ["colab", "usage"])
        # Should catch and not raise
        check_colab_usage(colab_bin="/fake/colab")

    @patch("subprocess.check_output")
    def test_get_session_url_success(self, mock_check_output):
        """Verifies extraction of Colab browser URL from CLI output."""
        mock_check_output.return_value = (
            "Session target-train-1024 active\n"
            "https://colab.research.google.com/drive/1A2B3C4D5E6F7G8H9\n"
            "Status: Running\n"
        )
        url = get_session_url(colab_bin="/fake/colab", session_name="target-train-1024")
        self.assertEqual(url, "https://colab.research.google.com/drive/1A2B3C4D5E6F7G8H9")

    @patch("subprocess.check_output")
    def test_get_session_url_missing(self, mock_check_output):
        """Verifies None is returned when no URL is present in the output."""
        mock_check_output.return_value = "No active URL assigned for session\n"
        url = get_session_url(colab_bin="/fake/colab", session_name="nonexistent")
        self.assertIsNone(url)

    @patch("subprocess.check_output")
    def test_get_session_url_handles_exception(self, mock_check_output):
        """Verifies None returned when subprocess fails."""
        mock_check_output.side_effect = Exception("Connection timed out")
        url = get_session_url(colab_bin="/fake/colab", session_name="failed-session")
        self.assertIsNone(url)

    @patch("subprocess.check_output")
    def test_list_colab_sessions(self, mock_check_output):
        """Verifies listing active sessions invokes CLI sessions command."""
        mock_check_output.return_value = "NAME                  GPU     STATUS   UPTIME\ntarget-train-1024     A100    Active   12m\n"
        list_colab_sessions(colab_bin="/fake/colab")
        mock_check_output.assert_called_once_with(["/fake/colab", "sessions"], text=True, stderr=subprocess.STDOUT)

    @patch("subprocess.run")
    def test_stop_colab_session(self, mock_run):
        """Verifies stop_colab_session invokes colab stop with session name."""
        mock_run.return_value = MagicMock(returncode=0)
        stop_colab_session(colab_bin="/fake/colab", session_name="session-to-kill")
        mock_run.assert_called_once_with(["/fake/colab", "stop", "-s", "session-to-kill"])

    def test_progressive_stages_schedule_parsing(self):
        """Verifies that progressive stage strings used by Colab runner parse accurately."""
        # Standard default schedule
        default_stages = parse_progressive_stages(None)
        self.assertEqual(len(default_stages), 4)
        self.assertEqual([s[0] for s in default_stages], [256, 512, 768, 1024])

        # Custom multi-resolution Colab run (size:epochs:lr format)
        custom_str_3 = "256:2:1e-3, 512:3:5e-4, 1024:5:2e-4"
        custom_stages_3 = parse_progressive_stages(custom_str_3)
        self.assertEqual(len(custom_stages_3), 3)
        self.assertEqual(custom_stages_3[0][0], 256)
        self.assertEqual(custom_stages_3[0][4], 2)
        self.assertEqual(custom_stages_3[0][5], 0.001)

        # Full 6-element specification (size:scale:sigma:batch:epochs:lr)
        custom_str_6 = "256:1.0:1.6:16:2:1e-3, 512:2.0:2.2:12:3:5e-4"
        custom_stages_6 = parse_progressive_stages(custom_str_6)
        self.assertEqual(len(custom_stages_6), 2)
        self.assertEqual(custom_stages_6[0], (256, 1.0, 1.6, 16, 2, 0.001, 400, 50))
        self.assertEqual(custom_stages_6[1], (512, 2.0, 2.2, 12, 3, 0.0005, 400, 50))


if __name__ == "__main__":
    unittest.main()
