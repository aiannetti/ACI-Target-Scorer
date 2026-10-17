"""
Target Scorer: Camera-Independent NRA Pistol Target Metrology & Scoring Engine.
"""

from target_scorer.config import (
    ShotAnnotation,
    TargetConfig,
    TargetRingsConfig,
    TrainConfig,
)
from target_scorer.geometry.rectifier import RectificationResult, TargetRectifier
from target_scorer.metrology.peaks import extract_subpixel_peaks
from target_scorer.metrology.scoring import (
    GroupMetrics,
    ShotResult,
    calculate_moa,
    score_target_group,
)
from target_scorer.metrology.visualization import render_scoring_inspection_overlay
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

# Lazy exports for CLI/entrypoint modules (prevents runpy duplicate import RuntimeWarning)
_LAZY_EXPORTS = {
    "generate_synthetic_demo_target": ("target_scorer.infer", "generate_synthetic_demo_target"),
    "run_inference_and_scoring": ("target_scorer.infer", "run_inference_and_scoring"),
    "run_training": ("target_scorer.train", "run_training"),
    "train_one_epoch": ("target_scorer.train", "train_one_epoch"),
    "validate_one_epoch": ("target_scorer.train", "validate_one_epoch"),
    "calculate_localization_metrics": ("target_scorer.train", "calculate_localization_metrics"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        import importlib
        mod_name, attr_name = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__version__ = "0.1.0"

__all__ = [
    "TargetConfig",
    "TrainConfig",
    "TargetRingsConfig",
    "ShotAnnotation",
    "TargetRectifier",
    "RectificationResult",
    "HighResHeatmapUNet",
    "ConvBlock",
    "AdaptiveWingLoss",
    "extract_subpixel_peaks",
    "score_target_group",
    "calculate_moa",
    "render_scoring_inspection_overlay",
    "ShotResult",
    "GroupMetrics",
    "generate_synthetic_demo_target",
    "run_inference_and_scoring",
    "run_training",
    "train_one_epoch",
    "validate_one_epoch",
    "calculate_localization_metrics",
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
