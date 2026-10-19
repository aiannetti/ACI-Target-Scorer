"""
Procedural synthesis of realistic cardboard backing textures, official NRA targets, and bullet holes.
Acts as a unified facade re-exporting canvas, holes, and markings submodules.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from target_scorer.config import ShotAnnotation, TargetConfig
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

__all__ = [
    # Canvas & Heatmap
    "generate_cardboard_texture",
    "draw_standard_bullseye",
    "generate_ground_truth_heatmap",
    # Physical Voids & Holes
    "generate_ragged_contour",
    "sample_bullet_void_bgr",
    "stamp_bullet_hole",
    "stamp_double_shot",
    "stamp_keyhole_shot",
    "sample_shot_position",
    # Markings, Headers & Handwriting
    "get_random_ink_style",
    "render_printed_target_header",
    "render_official_nra_seal",
    "render_procedural_handwriting",
    "draw_procedural_looped_digit",
    "draw_handwritten_score_string",
    "render_scorer_annotations",
    # Full Sample Orchestrator
    "generate_synthetic_target_sample",
]


def generate_synthetic_target_sample(
    config: Optional[TargetConfig] = None,
    num_singles: int = 5,
    num_doubles: int = 1,
    num_keyholes: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[ShotAnnotation], Tuple[float, float], Dict[str, float]]:
    """Generates complete procedural target sample with backing, rings, markings, and stratified holes."""
    if config is None:
        config = TargetConfig()

    canvas = generate_cardboard_texture(config)
    canvas, bullseye_center, ring_radii_px = draw_standard_bullseye(canvas, config)

    if config.include_header_text:
        canvas = render_printed_target_header(canvas, config)

    if config.include_nra_seal:
        canvas = render_official_nra_seal(canvas, config)

    if config.include_handwriting:
        canvas = render_procedural_handwriting(canvas, config, bullseye_center, ring_radii_px)

    annotations: List[ShotAnnotation] = []

    # 70% of targets feature mixed per-shot void colors; 30% have uniform backing
    uniform_void = sample_bullet_void_bgr(config) if random.random() < 0.30 else None

    def get_void() -> Tuple[int, int, int]:
        return uniform_void if uniform_void is not None else sample_bullet_void_bgr(config)

    for _ in range(num_singles):
        pos = sample_shot_position(bullseye_center, ring_radii_px, config)
        annotations.append(stamp_bullet_hole(canvas, pos, config, raggedness=random.uniform(0.9, 1.1), void_bgr=get_void()))
    for _ in range(num_doubles):
        pos = sample_shot_position(bullseye_center, ring_radii_px, config)
        annotations.extend(stamp_double_shot(canvas, pos, config, void_bgr=get_void()))
    for _ in range(num_keyholes):
        pos = sample_shot_position(bullseye_center, ring_radii_px, config)
        annotations.append(stamp_keyhole_shot(canvas, pos, config, void_bgr=get_void()))

    if config.include_scorer_marks:
        canvas = render_scorer_annotations(canvas, annotations, config, bullseye_center, ring_radii_px)

    heatmap = generate_ground_truth_heatmap(annotations, config)
    return canvas, heatmap, annotations, bullseye_center, ring_radii_px
