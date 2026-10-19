"""
Unit tests for the Target Scorer unified CLI dispatcher.
"""

import sys
import unittest
from unittest.mock import patch
from io import StringIO

from target_scorer.cli import main as cli_main


class TestCLIDispatcher(unittest.TestCase):
    """Verifies CLI dispatch, argument parsing, and help messages."""

    def test_cli_help(self):
        """Verifies target-scorer --help prints available subcommands and exits 0."""
        with patch.object(sys, "argv", ["target-scorer", "--help"]):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                with self.assertRaises(SystemExit) as cm:
                    cli_main()
                self.assertEqual(cm.exception.code, 0)
                output = fake_out.getvalue()
                self.assertIn("target-scorer", output)
                self.assertIn("infer", output)
                self.assertIn("train", output)
                self.assertIn("rectify", output)
                self.assertIn("autolabel", output)
                self.assertIn("annotate", output)

    def test_cli_no_args_exits_1(self):
        """Verifies running target-scorer with no arguments exits 1 and prints help."""
        with patch.object(sys, "argv", ["target-scorer"]):
            with patch("sys.stdout", new=StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    cli_main()
                self.assertEqual(cm.exception.code, 1)

    def test_cli_subcommand_forwarding(self):
        """Verifies rectify --help dispatches properly."""
        with patch.object(sys, "argv", ["target-scorer", "rectify", "--help"]):
            with patch("sys.stdout", new=StringIO()) as fake_out:
                with self.assertRaises(SystemExit) as cm:
                    cli_main()
                self.assertEqual(cm.exception.code, 0)
                output = fake_out.getvalue()
                self.assertIn("--image", output)
                self.assertIn("--target", output)


if __name__ == "__main__":
    unittest.main()
