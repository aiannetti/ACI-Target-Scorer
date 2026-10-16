"""
Legacy compatibility module for target_inference_gauge_overlay_scoring_engine.py.
Delegates to target_scorer.infer and target_scorer metrology modules.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from target_scorer.config import TargetRingsConfig, TrainConfig
from target_scorer.geometry.rectifier import RectificationResult, TargetRectifier
from target_scorer.infer import generate_synthetic_demo_target, main, run_inference_and_scoring
from target_scorer.metrology.peaks import extract_subpixel_peaks
from target_scorer.metrology.scoring import (
    GroupMetrics,
    ShotResult,
    calculate_moa,
    score_target_group,
)
from target_scorer.metrology.visualization import render_scoring_inspection_overlay
from target_scorer.models.unet import ConvBlock, HighResHeatmapUNet

__all__ = [
    "TrainConfig",
    "TargetRingsConfig",
    "ShotResult",
    "GroupMetrics",
    "ConvBlock",
    "HighResHeatmapUNet",
    "extract_subpixel_peaks",
    "calculate_moa",
    "score_target_group",
    "render_scoring_inspection_overlay",
    "generate_synthetic_demo_target",
    "run_inference_and_scoring",
    "TargetRectifier",
    "RectificationResult",
]

if __name__ == "__main__":
    main()