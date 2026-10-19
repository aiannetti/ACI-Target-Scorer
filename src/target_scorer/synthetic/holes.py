"""
Procedural bullet hole stamping, ragged paper fiber simulation, multi-tone voids, and stratified shot placement.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from target_scorer.config import ShotAnnotation, TargetConfig


def generate_ragged_contour(
    center: Tuple[float, float],
    base_radius_px: float,
    config: TargetConfig,
    anisotropy: float = 1.0
) -> np.ndarray:
    """Generates perturbed ragged polygon edges simulating torn paper fibers."""
    cx, cy = center
    num_vertices = 48
    angles = np.linspace(0, 2.0 * np.pi, num_vertices, endpoint=False)
    phase1, phase2, phase3 = random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi), random.uniform(0, 2 * np.pi)
    flute_bias = 0.08 if config.flute_direction == "vertical" else -0.08

    radii = []
    for theta in angles:
        perturbation = (
            0.06 * np.sin(3.0 * theta + phase1) +
            0.04 * np.sin(7.0 * theta + phase2) +
            0.03 * np.cos(11.0 * theta + phase3) +
            flute_bias * np.cos(2.0 * theta)
        )
        scale_val = float(np.clip(1.0 + perturbation + random.uniform(-0.04, 0.04), 0.75, 1.12))
        r = base_radius_px * scale_val * anisotropy
        radii.append(r)

    points = [[cx + r * np.cos(t), cy + r * np.sin(t)] for r, t in zip(radii, angles)]
    return np.array(points, dtype=np.int32)


def sample_bullet_void_bgr(config: Optional[TargetConfig] = None) -> Tuple[int, int, int]:
    """
    Samples realistic bullet hole penetration void background color:
      - 1/3 (33.3%) bright white/cream (flatbed document scanner lid backing, white card backer)
      - 1/3 (33.3%) dark range void / rubber bullet trap shadow
      - 1/3 (33.4%) tan/kraft corrugated cardboard backing
    """
    mode = getattr(config, "void_color_mode", "mixed") if config else "mixed"
    if mode == "scanner_white":
        val = random.randint(225, 252)
        return (val - random.randint(0, 6), val, val - random.randint(0, 4))
    elif mode == "dark_void":
        return (random.randint(12, 28), random.randint(10, 26), random.randint(10, 26))
    elif mode == "tan_backer":
        return (random.randint(140, 175), random.randint(160, 195), random.randint(180, 215))

    p = random.random()
    if p < 1.0 / 3.0:
        val = random.randint(225, 252)
        return (val - random.randint(0, 6), val, val - random.randint(0, 4))
    elif p < 2.0 / 3.0:
        return (random.randint(12, 28), random.randint(10, 26), random.randint(10, 26))
    else:
        return (random.randint(140, 175), random.randint(160, 195), random.randint(180, 215))


def stamp_bullet_hole(
    canvas: np.ndarray,
    center: Tuple[float, float],
    config: TargetConfig,
    raggedness: float = 1.0,
    void_bgr: Optional[Tuple[int, int, int]] = None
) -> ShotAnnotation:
    """Stamps realistic single bullet penetration hole with lead wipe grease ring and multi-tone void."""
    cx, cy = center
    r_bullet_px = (config.bullet_diameter_mm / 2.0) * config.pixels_per_mm
    grease_width_px = config.grease_ring_width_mm * config.pixels_per_mm

    # 1. Outer lead wipe grease ring (NRA Rule 14.3)
    grease_radius_outer = int(round(r_bullet_px + grease_width_px * 0.5))
    cv2.circle(canvas, (int(round(cx)), int(round(cy))), grease_radius_outer, (22, 22, 25), -1, lineType=cv2.LINE_AA)

    # 2. Perforated void with simulated illumination (scanner white, dark range, or tan backer)
    effective_void_radius = r_bullet_px * (0.80 * raggedness)
    hole_poly = generate_ragged_contour((cx, cy), effective_void_radius, config)
    if void_bgr is None:
        void_bgr = sample_bullet_void_bgr(config)
    cv2.fillPoly(canvas, [hole_poly], void_bgr, lineType=cv2.LINE_AA)

    # 3. Torn paper fibers around penetration boundary
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

        if void_bgr[0] > 200:
            fiber_tone = (random.randint(40, 75), random.randint(40, 75), random.randint(45, 80)) if random.random() < 0.5 else (205, 205, 212)
        elif void_bgr[0] < 40:
            fiber_tone = (
                min(255, int(config.cardboard_base_bgr[0] + random.randint(10, 30))),
                min(255, int(config.cardboard_base_bgr[1] + random.randint(10, 30))),
                min(255, int(config.cardboard_base_bgr[2] + random.randint(10, 30)))
            )
        else:
            if random.random() < 0.5:
                fiber_tone = (
                    min(255, int(config.cardboard_base_bgr[0] + random.randint(25, 45))),
                    min(255, int(config.cardboard_base_bgr[1] + random.randint(25, 45))),
                    min(255, int(config.cardboard_base_bgr[2] + random.randint(25, 45)))
                )
            else:
                fiber_tone = (random.randint(40, 65), random.randint(40, 65), random.randint(45, 70))
        cv2.fillPoly(canvas, [np.array([p1, p2, p_tip], dtype=np.int32)], fiber_tone, lineType=cv2.LINE_AA)

    return ShotAnnotation(center_x=cx, center_y=cy, shot_type="single", caliber_radius_px=r_bullet_px)


def stamp_double_shot(
    canvas: np.ndarray,
    primary_center: Tuple[float, float],
    config: TargetConfig,
    void_bgr: Optional[Tuple[int, int, int]] = None
) -> List[ShotAnnotation]:
    """Stamps overlapping double shot with shared ragged boundary."""
    r_bullet_px = (config.bullet_diameter_mm / 2.0) * config.pixels_per_mm
    margin_px = max(16.0, config.canvas_size * 0.06)
    min_y = (20.0 * (config.canvas_size / 256.0) + 4.8 * config.heatmap_sigma_px) if config.include_header_text else margin_px
    cx1, cy1 = primary_center
    separation_dist = random.uniform(0.65 * (2.0 * r_bullet_px), 1.25 * (2.0 * r_bullet_px))
    angle = random.uniform(0, 2.0 * np.pi)
    cx2 = max(margin_px, min(config.canvas_size - margin_px, cx1 + separation_dist * np.cos(angle)))
    cy2 = max(min_y, min(config.canvas_size - margin_px, cy1 + separation_dist * np.sin(angle)))

    if void_bgr is None:
        void_bgr = sample_bullet_void_bgr(config)
    shot1 = stamp_bullet_hole(canvas, (cx1, cy1), config, raggedness=0.95, void_bgr=void_bgr)
    shot1.shot_type = "double_primary"
    shot2 = stamp_bullet_hole(canvas, (cx2, cy2), config, raggedness=1.0, void_bgr=void_bgr)
    shot2.shot_type = "double_secondary"
    return [shot1, shot2]


def stamp_keyhole_shot(
    canvas: np.ndarray,
    center: Tuple[float, float],
    config: TargetConfig,
    void_bgr: Optional[Tuple[int, int, int]] = None
) -> ShotAnnotation:
    """Stamps elongated tumbling keyhole shot with lead wipe and void backing."""
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
    if void_bgr is None:
        void_bgr = sample_bullet_void_bgr(config)
    cv2.ellipse(canvas, center_int, void_axes, yaw_deg, 0, 360, void_bgr, -1, lineType=cv2.LINE_AA)

    return ShotAnnotation(
        center_x=cx, center_y=cy, shot_type="keyhole",
        caliber_radius_px=r_bullet_px, yaw_angle_deg=yaw_deg, aspect_ratio=aspect_ratio
    )


def sample_shot_position(
    bullseye_center: Tuple[float, float],
    ring_radii_px: Dict[str, float],
    config: TargetConfig
) -> Tuple[float, float]:
    """
    Samples bullet hole position across 4 stratified physical zones:
      - Zone 1 (40%): Inner Aiming Black (8, 9, 10, X rings)
      - Zone 2 (20%): Outer Scoring Rings (4, 5, 6, 7 on B-2; 6, 7, 8 on B-3)
      - Zone 3 (20%): On-Paper Margins Outside Rings (near headers, handwriting, tallies)
      - Zone 4 (20%): Off-Paper / Cardboard Backer & Edge-Clipping
    """
    bull_x, bull_y = bullseye_center
    target_type = config.target_type.upper()
    w = float(config.canvas_size)
    h = float(config.canvas_size)

    bull_radius_px = ring_radii_px.get("7" if target_type == "B-2" else "9", 38.989 * config.pixels_per_mm)
    outer_ring_key = "4" if target_type == "B-2" else "6"
    outer_ring_radius_px = ring_radii_px.get(
        outer_ring_key,
        93.091 * config.pixels_per_mm if target_type == "B-2" else 105.664 * config.pixels_per_mm
    )

    pz = random.random()
    if pz < 0.40:
        # Zone 1: Inner black bullseye
        r = float(np.random.exponential(scale=bull_radius_px * 0.40))
        r = min(bull_radius_px * 0.95, r)
        theta = random.uniform(0, 2.0 * np.pi)
        sx = bull_x + r * np.cos(theta)
        sy = bull_y + r * np.sin(theta)
    elif pz < 0.60:
        # Zone 2: Outer scoring rings on tan cardstock
        r = random.uniform(bull_radius_px * 1.02, outer_ring_radius_px * 0.98)
        theta = random.uniform(0, 2.0 * np.pi)
        sx = bull_x + r * np.cos(theta)
        sy = bull_y + r * np.sin(theta)
    elif pz < 0.80:
        # Zone 3: On-paper margin outside printed scoring rings
        r_min = outer_ring_radius_px * 1.04
        r_max = max(r_min + 10.0, min(w, h) / 2.0 - 25.0 * max(0.5, config.canvas_size / 512.0))
        r = random.uniform(r_min, r_max)
        theta = random.uniform(0, 2.0 * np.pi)
        sx = bull_x + r * np.cos(theta)
        sy = bull_y + r * np.sin(theta)
    else:
        # Zone 4: Off-paper / cardboard backer & card border edges
        edge = random.choice(["bottom", "bottom", "left", "right", "top"])
        sf = max(0.5, config.canvas_size / 512.0)
        edge_margin = random.uniform(
            max(8.0, 2.0 * config.pixels_per_mm),
            max(20.0, 6.0 * config.pixels_per_mm)
        )
        if edge == "bottom":
            sx = random.uniform(20.0 * sf, w - 20.0 * sf)
            sy = h - edge_margin
        elif edge == "left":
            sx = edge_margin
            sy = random.uniform(30.0 * sf, h - 20.0 * sf)
        elif edge == "right":
            sx = w - edge_margin
            sy = random.uniform(30.0 * sf, h - 20.0 * sf)
        else:
            sx = random.choice([
                random.uniform(20.0 * sf, 80.0 * sf),
                random.uniform(w - 80.0 * sf, w - 20.0 * sf)
            ])
            sy = max(28.0 * (config.canvas_size / 256.0), edge_margin * 1.5)

    safe_m = max(6.0, 1.5 * config.pixels_per_mm)
    min_y = (20.0 * (config.canvas_size / 256.0) + 4.8 * config.heatmap_sigma_px) if config.include_header_text else safe_m
    sx = max(safe_m, min(w - safe_m, sx))
    sy = max(min_y, min(h - safe_m, sy))
    return (float(sx), float(sy))
