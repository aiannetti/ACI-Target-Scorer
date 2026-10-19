"""
Procedural target generation and PyTorch synthetic datasets.
"""

from target_scorer.synthetic.dataset import SyntheticTargetDataset, custom_collate_fn
from target_scorer.synthetic.canvas import (
    draw_standard_bullseye,
    generate_cardboard_texture,
    generate_ground_truth_heatmap,
)
from target_scorer.synthetic.holes import (
    generate_ragged_contour,
    sample_bullet_void_bgr,
    sample_shot_position,
    stamp_bullet_hole,
    stamp_double_shot,
    stamp_keyhole_shot,
)
from target_scorer.synthetic.markings import (
    draw_handwritten_score_string,
    draw_procedural_looped_digit,
    get_random_ink_style,
    render_official_nra_seal,
    render_printed_target_header,
    render_procedural_handwriting,
    render_scorer_annotations,
)
from target_scorer.synthetic.generator import generate_synthetic_target_sample

__all__ = [
    # Canvas
    "generate_cardboard_texture",
    "draw_standard_bullseye",
    "generate_ground_truth_heatmap",
    # Holes & Voids
    "generate_ragged_contour",
    "sample_bullet_void_bgr",
    "stamp_bullet_hole",
    "stamp_double_shot",
    "stamp_keyhole_shot",
    "sample_shot_position",
    # Markings & Seals
    "get_random_ink_style",
    "render_printed_target_header",
    "render_official_nra_seal",
    "render_procedural_handwriting",
    "draw_procedural_looped_digit",
    "draw_handwritten_score_string",
    "render_scorer_annotations",
    # Orchestrator & Dataset
    "generate_synthetic_target_sample",
    "SyntheticTargetDataset",
    "custom_collate_fn",
]
