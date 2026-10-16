"""
Procedural target generation and PyTorch synthetic datasets.
"""

from target_scorer.synthetic.dataset import SyntheticTargetDataset, custom_collate_fn
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
    "generate_cardboard_texture",
    "draw_standard_bullseye",
    "generate_ragged_contour",
    "stamp_bullet_hole",
    "stamp_double_shot",
    "stamp_keyhole_shot",
    "generate_ground_truth_heatmap",
    "generate_synthetic_target_sample",
    "SyntheticTargetDataset",
    "custom_collate_fn",
]
