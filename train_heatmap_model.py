"""
Procedural Heatmap Training Pipeline CLI Wrapper (NRA Target Metrology & Scoring Engine).
Backward-compatible shim importing from the target_scorer package.
"""

import sys
import os

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from target_scorer.config import (
    ShotAnnotation,
    TargetConfig,
    TargetRingsConfig,
    TrainConfig,
)
from target_scorer.metrology.peaks import extract_subpixel_peaks
from target_scorer.models.loss import AdaptiveWingLoss
from target_scorer.models.unet import ConvBlock, HighResHeatmapUNet
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
from target_scorer.train import (
    calculate_localization_metrics,
    main,
    run_training,
    train_one_epoch,
    validate_one_epoch,
)

__all__ = [
    "TargetConfig",
    "TrainConfig",
    "ShotAnnotation",
    "generate_cardboard_texture",
    "draw_standard_bullseye",
    "generate_ragged_contour",
    "stamp_bullet_hole",
    "stamp_double_shot",
    "stamp_keyhole_shot",
    "generate_ground_truth_heatmap",
    "generate_synthetic_target_sample",
    "SyntheticTargetDataset",
    "ConvBlock",
    "HighResHeatmapUNet",
    "extract_subpixel_peaks",
    "calculate_localization_metrics",
    "AdaptiveWingLoss",
    "train_one_epoch",
    "validate_one_epoch",
    "custom_collate_fn",
    "run_training",
]

if __name__ == "__main__":
    main()
