"""
Metrology, sub-pixel coordinate extraction, NRA inward scoring, and scorecard rendering.
"""

from target_scorer.metrology.peaks import extract_subpixel_peaks
from target_scorer.metrology.scoring import (
    GroupMetrics,
    ShotResult,
    calculate_moa,
    score_target_group,
)
from target_scorer.metrology.visualization import render_scoring_inspection_overlay

__all__ = [
    "extract_subpixel_peaks",
    "ShotResult",
    "GroupMetrics",
    "calculate_moa",
    "score_target_group",
    "render_scoring_inspection_overlay",
]
