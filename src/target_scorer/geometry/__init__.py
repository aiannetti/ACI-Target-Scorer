"""
Geometric target normalization and camera rectification package.
"""

from target_scorer.geometry.canonical_loader import ImageCanonicalizer, canonicalize_image
from target_scorer.geometry.rectifier import RectificationResult, TargetRectifier

__all__ = ["ImageCanonicalizer", "canonicalize_image", "RectificationResult", "TargetRectifier"]
