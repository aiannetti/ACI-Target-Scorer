"""
Procedural synthesis of cardboard backing textures, official NRA target rings, and ground truth heatmaps.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from target_scorer.config import ShotAnnotation, TargetConfig


def generate_cardboard_texture(config: TargetConfig) -> np.ndarray:
    """Synthesizes realistic corrugated cardboard backing texture with fluting grain, noise, and lighting vignette."""
    height, width = config.canvas_size, config.canvas_size
    base_color = np.array(config.cardboard_base_bgr, dtype=np.float32)
    canvas = np.ones((height, width, 3), dtype=np.float32) * base_color

    flute_period = max(12.0, 8.0 * config.pixels_per_mm)
    if config.flute_direction == "vertical":
        x_coords = np.arange(width, dtype=np.float32)
        flute_wave = np.sin(2.0 * np.pi * x_coords / flute_period) * 12.0
        canvas += flute_wave[np.newaxis, :, np.newaxis]
    else:
        y_coords = np.arange(height, dtype=np.float32)
        flute_wave = np.sin(2.0 * np.pi * y_coords / flute_period) * 12.0
        canvas += flute_wave[:, np.newaxis, np.newaxis]

    fiber_noise = np.random.normal(0.0, 6.0, (height, width, 1)).astype(np.float32)
    canvas += fiber_noise

    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    lighting_vignette = 1.0 - 0.15 * (xx**2 + yy**2)
    canvas *= lighting_vignette[:, :, np.newaxis]
    return np.clip(canvas, 0, 255).astype(np.uint8)


def draw_standard_bullseye(canvas: np.ndarray, config: TargetConfig) -> Tuple[np.ndarray, Tuple[float, float], Dict[str, float]]:
    """Renders authentic NRA B-2 or NRA B-3 target rings and solid aiming black."""
    center_x = config.canvas_size / 2.0
    center_y = config.canvas_size / 2.0
    c_int = (int(round(center_x)), int(round(center_y)))

    target_type = config.target_type.upper()
    ring_radii_px: Dict[str, float] = {}

    if target_type == "B-3":
        # NRA B-3: 9 and 10 rings black (3.06 in dia)
        black_radius_mm = 38.862
        inner_rings = [("X", 11.43), ("10", 22.86)]
        outer_rings = [("8", 56.642), ("7", 77.978), ("6", 105.664)]
    else:
        # NRA B-2: 7, 8, 9, 10 rings black (3.07 in dia) - Innermost ring is 10 (no X-ring)
        black_radius_mm = 38.989
        inner_rings = [("10", 11.43), ("9", 19.558), ("8", 28.321)]
        outer_rings = [("6", 52.832), ("5", 70.612), ("4", 93.091)]

    black_bull_radius_px = int(round(black_radius_mm * config.pixels_per_mm))
    cv2.circle(canvas, c_int, black_bull_radius_px, (30, 30, 35), -1, lineType=cv2.LINE_AA)

    cross_len = max(3, int(round(1.5 * config.pixels_per_mm)))
    for label, r_mm in inner_rings:
        r_px = int(round(r_mm * config.pixels_per_mm))
        ring_radii_px[label] = r_mm * config.pixels_per_mm
        if label == "X":
            cv2.circle(canvas, c_int, r_px, (220, 220, 220), 1, lineType=cv2.LINE_AA)
            cv2.line(canvas, (c_int[0] - cross_len, c_int[1]), (c_int[0] + cross_len, c_int[1]), (220, 220, 220), 1, lineType=cv2.LINE_AA)
            cv2.line(canvas, (c_int[0], c_int[1] - cross_len), (c_int[0], c_int[1] + cross_len), (220, 220, 220), 1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(canvas, c_int, r_px, (220, 220, 220), config.ring_line_width_px, lineType=cv2.LINE_AA)

    ring_radii_px["9" if target_type == "B-3" else "7"] = black_radius_mm * config.pixels_per_mm

    for label, r_mm in outer_rings:
        r_px = int(round(r_mm * config.pixels_per_mm))
        ring_radii_px[label] = r_mm * config.pixels_per_mm
        cv2.circle(canvas, c_int, r_px, (40, 40, 45), config.ring_line_width_px, lineType=cv2.LINE_AA)

    return canvas, (center_x, center_y), ring_radii_px


def generate_ground_truth_heatmap(
    annotations: List[ShotAnnotation],
    config: TargetConfig,
    sigma_override: Optional[float] = None
) -> np.ndarray:
    """Generates continuous 2D Gaussian heatmaps with max-composition."""
    height, width = config.canvas_size, config.canvas_size
    heatmap = np.zeros((height, width), dtype=np.float32)
    sigma = sigma_override if sigma_override is not None else config.heatmap_sigma_px
    two_sigma_sq = 2.0 * (sigma ** 2)
    y_coords = np.arange(height, dtype=np.float32)
    x_coords = np.arange(width, dtype=np.float32)

    for shot in annotations:
        cx, cy = shot.center_x, shot.center_y
        radius_cutoff = int(math.ceil(4.0 * sigma))
        x_min = max(0, int(math.floor(cx - radius_cutoff)))
        x_max = min(width, int(math.ceil(cx + radius_cutoff + 1)))
        y_min = max(0, int(math.floor(cy - radius_cutoff)))
        y_max = min(height, int(math.ceil(cy + radius_cutoff + 1)))

        if x_min >= x_max or y_min >= y_max:
            continue

        dx = x_coords[x_min:x_max] - cx
        dy = y_coords[y_min:y_max] - cy
        dx_grid, dy_grid = np.meshgrid(dx, dy)
        gaussian_patch = np.exp(-(dx_grid ** 2 + dy_grid ** 2) / two_sigma_sq)
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], gaussian_patch)

    return np.clip(heatmap, 0.0, 1.0)
