"""
Target Rectifier Module (NRA Target Metrology & Scoring Engine).
Backward-compatible shim importing from the target_scorer package.
"""

import sys
import os

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from target_scorer.geometry.rectifier import RectificationResult, TargetRectifier
from target_scorer.cli import rectify_cli

__all__ = ["RectificationResult", "TargetRectifier"]

if __name__ == "__main__":
    rectify_cli()
