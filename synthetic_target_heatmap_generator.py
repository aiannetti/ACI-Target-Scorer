"""
Legacy compatibility module for synthetic_target_heatmap_generator.py.
Delegates to target_scorer.synthetic.generator.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from target_scorer.config import ShotAnnotation, TargetConfig
from target_scorer.synthetic.generator import (
    draw_standard_bullseye,
    generate_cardboard_texture,
    generate_ground_truth_heatmap,
    generate_ragged_contour,
    generate_synthetic_target_sample,
    stamp_bullet_hole,
    stamp_double_shot,
    stamp_keyhole_shot,
)

__all__ = [
    "TargetConfig",
    "ShotAnnotation",
    "generate_cardboard_texture",
    "draw_standard_bullseye",
    "generate_ragged_contour",
    "stamp_bullet_hole",
    "stamp_double_shot",
    "stamp_keyhole_shot",
    "generate_ground_truth_heatmap",
    "generate_synthetic_target_sample",
]