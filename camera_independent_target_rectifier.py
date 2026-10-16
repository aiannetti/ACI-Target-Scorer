"""
Legacy compatibility module for camera_independent_target_rectifier.py.
Delegates to target_scorer.geometry.rectifier.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from target_scorer.geometry.rectifier import RectificationResult, TargetRectifier
from target_scorer.cli import rectify_cli as main

__all__ = ["RectificationResult", "TargetRectifier", "main"]

if __name__ == "__main__":
    main()