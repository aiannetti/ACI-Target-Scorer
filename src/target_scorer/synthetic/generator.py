"""
Procedural synthesis of realistic cardboard backing textures, official NRA targets, and bullet holes.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from target_scorer.config import ShotAnnotation, TargetConfig


def generate_cardboard_texture(config: TargetConfig) -> np.ndarray:
    """Synthesizes realistic corrugated cardboard backing texture."""
    height, width = config.canvas_size, config.canvas_size
    base_color = np.array(config.cardboard_base_bgr, dtype=np.float32)
    canvas = np.ones((height, width, 3), dtype=np.float32) * base_color

    flute_period = 32.0
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
        # NRA B-2: 7, 8, 9, 10 rings black (3.07 in dia)
        black_radius_mm = 38.989
        inner_rings = [("X", 5.715), ("10", 11.43), ("9", 19.558), ("8", 28.321)]
        outer_rings = [("6", 52.832), ("5", 70.612), ("4", 93.091)]

    black_bull_radius_px = int(round(black_radius_mm * config.pixels_per_mm))
    cv2.circle(canvas, c_int, black_bull_radius_px, (30, 30, 35), -1, lineType=cv2.LINE_AA)

    for label, r_mm in inner_rings:
        r_px = int(round(r_mm * config.pixels_per_mm))
        ring_radii_px[label] = r_mm * config.pixels_per_mm
        if label == "X":
            cv2.circle(canvas, c_int, r_px, (220, 220, 220), 1, lineType=cv2.LINE_AA)
            cv2.line(canvas, (c_int[0] - 5, c_int[1]), (c_int[0] + 5, c_int[1]), (220, 220, 220), 1, lineType=cv2.LINE_AA)
            cv2.line(canvas, (c_int[0], c_int[1] - 5), (c_int[0], c_int[1] + 5), (220, 220, 220), 1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(canvas, c_int, r_px, (220, 220, 220), config.ring_line_width_px, lineType=cv2.LINE_AA)

    ring_radii_px["9" if target_type == "B-3" else "7"] = black_radius_mm * config.pixels_per_mm

    for label, r_mm in outer_rings:
        r_px = int(round(r_mm * config.pixels_per_mm))
        ring_radii_px[label] = r_mm * config.pixels_per_mm
        cv2.circle(canvas, c_int, r_px, (40, 40, 45), config.ring_line_width_px, lineType=cv2.LINE_AA)

    return canvas, (center_x, center_y), ring_radii_px


def generate_ragged_contour(center: Tuple[float, float], base_radius_px: float, 
                            config: TargetConfig, anisotropy: float = 1.0) -> np.ndarray:
    """Generates perturbed ragged polygon edges simulating torn paper fibers."""
    cx, cy = center
    num_vertices = 48
    angles = np.linspace(0, 2.0 * np.pi, num_vertices, endpoint=False)
    phase1, phase2, phase3 = random.uniform(0, 2*np.pi), random.uniform(0, 2*np.pi), random.uniform(0, 2*np.pi)
    flute_bias = 0.18 if config.flute_direction == "vertical" else -0.18

    radii = []
    for theta in angles:
        perturbation = (
            0.12 * np.sin(3.0 * theta + phase1) +
            0.08 * np.sin(7.0 * theta + phase2) +
            0.05 * np.cos(11.0 * theta + phase3) +
            flute_bias * np.cos(2.0 * theta)
        )
        r = base_radius_px * (1.0 + perturbation + random.uniform(-0.06, 0.06)) * anisotropy
        radii.append(r)

    points = [[cx + r * np.cos(t), cy + r * np.sin(t)] for r, t in zip(radii, angles)]
    return np.array(points, dtype=np.int32)


def stamp_bullet_hole(canvas: np.ndarray, center: Tuple[float, float], 
                      config: TargetConfig, raggedness: float = 1.0) -> ShotAnnotation:
    """Stamps realistic single bullet penetration hole with lead wipe grease ring."""
    cx, cy = center
    r_bullet_px = (config.bullet_diameter_mm / 2.0) * config.pixels_per_mm
    grease_width_px = config.grease_ring_width_mm * config.pixels_per_mm

    grease_radius_outer = int(round(r_bullet_px + grease_width_px * 0.5))
    cv2.circle(canvas, (int(round(cx)), int(round(cy))), grease_radius_outer, (25, 25, 28), -1, lineType=cv2.LINE_AA)

    effective_void_radius = r_bullet_px * (0.85 * raggedness)
    hole_poly = generate_ragged_contour((cx, cy), effective_void_radius, config)
    cv2.fillPoly(canvas, [hole_poly], (12, 10, 10), lineType=cv2.LINE_AA)

    num_flaps = random.randint(2, 4)
    for _ in range(num_flaps):
        flap_angle = random.uniform(0, 2.0 * np.pi)
        flap_length = random.uniform(r_bullet_px * 0.35, r_bullet_px * 0.70)
        flap_width = random.uniform(r_bullet_px * 0.30, r_bullet_px * 0.55)

        bx = cx + r_bullet_px * 0.9 * np.cos(flap_angle)
        by = cy + r_bullet_px * 0.9 * np.sin(flap_angle)
        tip_x = cx + (r_bullet_px - flap_length) * np.cos(flap_angle)
        tip_y = cy + (r_bullet_px - flap_length) * np.sin(flap_angle)

        tangent_angle = flap_angle + np.pi / 2.0
        p1 = (int(bx + (flap_width / 2.0) * np.cos(tangent_angle)), int(by + (flap_width / 2.0) * np.sin(tangent_angle)))
        p2 = (int(bx - (flap_width / 2.0) * np.cos(tangent_angle)), int(by - (flap_width / 2.0) * np.sin(tangent_angle)))
        p_tip = (int(tip_x), int(tip_y))

        fiber_tone = (
            min(255, int(config.cardboard_base_bgr[0] + random.randint(10, 30))),
            min(255, int(config.cardboard_base_bgr[1] + random.randint(10, 30))),
            min(255, int(config.cardboard_base_bgr[2] + random.randint(10, 30)))
        )
        cv2.fillPoly(canvas, [np.array([p1, p2, p_tip], dtype=np.int32)], fiber_tone, lineType=cv2.LINE_AA)

    return ShotAnnotation(center_x=cx, center_y=cy, shot_type="single", caliber_radius_px=r_bullet_px)


def stamp_double_shot(canvas: np.ndarray, primary_center: Tuple[float, float],
                      config: TargetConfig) -> List[ShotAnnotation]:
    """Stamps overlapping double shot with shared ragged boundary."""
    r_bullet_px = (config.bullet_diameter_mm / 2.0) * config.pixels_per_mm
    cx1, cy1 = primary_center
    separation_dist = random.uniform(0.65 * (2.0 * r_bullet_px), 1.25 * (2.0 * r_bullet_px))
    angle = random.uniform(0, 2.0 * np.pi)
    cx2, cy2 = cx1 + separation_dist * np.cos(angle), cy1 + separation_dist * np.sin(angle)

    shot1 = stamp_bullet_hole(canvas, (cx1, cy1), config, raggedness=0.95)
    shot1.shot_type = "double_primary"
    shot2 = stamp_bullet_hole(canvas, (cx2, cy2), config, raggedness=1.0)
    shot2.shot_type = "double_secondary"
    return [shot1, shot2]


def stamp_keyhole_shot(canvas: np.ndarray, center: Tuple[float, float],
                       config: TargetConfig) -> ShotAnnotation:
    """Stamps elongated tumbling keyhole shot."""
    cx, cy = center
    r_bullet_px = (config.bullet_diameter_mm / 2.0) * config.pixels_per_mm
    yaw_deg = random.uniform(0.0, 360.0)
    aspect_ratio = random.uniform(1.4, 2.2)

    major_len_px = r_bullet_px * aspect_ratio
    minor_len_px = r_bullet_px * 1.05
    center_int = (int(round(cx)), int(round(cy)))
    axes = (int(round(major_len_px + 2.0)), int(round(minor_len_px + 2.0)))
    cv2.ellipse(canvas, center_int, axes, yaw_deg, 0, 360, (25, 25, 28), -1, lineType=cv2.LINE_AA)

    void_axes = (int(round(major_len_px * 0.85)), int(round(minor_len_px * 0.80)))
    cv2.ellipse(canvas, center_int, void_axes, yaw_deg, 0, 360, (12, 10, 10), -1, lineType=cv2.LINE_AA)

    return ShotAnnotation(
        center_x=cx, center_y=cy, shot_type="keyhole",
        caliber_radius_px=r_bullet_px, yaw_angle_deg=yaw_deg, aspect_ratio=aspect_ratio
    )


def generate_ground_truth_heatmap(annotations: List[ShotAnnotation], config: TargetConfig) -> np.ndarray:
    """Generates continuous 2D Gaussian heatmaps with max-composition."""
    height, width = config.canvas_size, config.canvas_size
    heatmap = np.zeros((height, width), dtype=np.float32)
    sigma = config.heatmap_sigma_px
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

        dx = x_coords[x_min:x_max] - cx
        dy = y_coords[y_min:y_max] - cy
        dx_grid, dy_grid = np.meshgrid(dx, dy)
        gaussian_patch = np.exp(-(dx_grid ** 2 + dy_grid ** 2) / two_sigma_sq)
        heatmap[y_min:y_max, x_min:x_max] = np.maximum(heatmap[y_min:y_max, x_min:x_max], gaussian_patch)

    return np.clip(heatmap, 0.0, 1.0)


def generate_synthetic_target_sample(
    config: Optional[TargetConfig] = None,
    num_singles: int = 5,
    num_doubles: int = 1,
    num_keyholes: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[ShotAnnotation], Tuple[float, float], Dict[str, float]]:
    """Generates complete procedural target sample."""
    if config is None:
        config = TargetConfig()

    canvas = generate_cardboard_texture(config)
    canvas, bullseye_center, ring_radii_px = draw_standard_bullseye(canvas, config)
    annotations: List[ShotAnnotation] = []
    bull_x, bull_y = bullseye_center
    outer_key = "6" if config.target_type.upper() == "B-3" else "5"
    max_dispersion_px = ring_radii_px[outer_key] * 0.85

    def sample_shot_position() -> Tuple[float, float]:
        r = np.random.exponential(scale=max_dispersion_px * 0.35)
        theta = random.uniform(0, 2.0 * np.pi)
        sx = max(64.0, min(config.canvas_size - 64.0, bull_x + r * np.cos(theta)))
        sy = max(64.0, min(config.canvas_size - 64.0, bull_y + r * np.sin(theta)))
        return (float(sx), float(sy))

    for _ in range(num_singles):
        annotations.append(stamp_bullet_hole(canvas, sample_shot_position(), config, raggedness=random.uniform(0.9, 1.1)))
    for _ in range(num_doubles):
        annotations.extend(stamp_double_shot(canvas, sample_shot_position(), config))
    for _ in range(num_keyholes):
        annotations.append(stamp_keyhole_shot(canvas, sample_shot_position(), config))

    heatmap = generate_ground_truth_heatmap(annotations, config)
    return canvas, heatmap, annotations, bullseye_center, ring_radii_px
