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


def generate_ragged_contour(center: Tuple[float, float], base_radius_px: float, 
                            config: TargetConfig, anisotropy: float = 1.0) -> np.ndarray:
    """Generates perturbed ragged polygon edges simulating torn paper fibers."""
    cx, cy = center
    num_vertices = 48
    angles = np.linspace(0, 2.0 * np.pi, num_vertices, endpoint=False)
    phase1, phase2, phase3 = random.uniform(0, 2*np.pi), random.uniform(0, 2*np.pi), random.uniform(0, 2*np.pi)
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
        # Scanner backing / white card
        val = random.randint(225, 252)
        return (val - random.randint(0, 6), val, val - random.randint(0, 4))
    elif p < 2.0 / 3.0:
        # Dark range void / rubber bullet trap
        return (random.randint(12, 28), random.randint(10, 26), random.randint(10, 26))
    else:
        # Tan/kraft corrugated cardboard backer
        return (random.randint(140, 175), random.randint(160, 195), random.randint(180, 215))


def stamp_bullet_hole(canvas: np.ndarray, center: Tuple[float, float], 
                      config: TargetConfig, raggedness: float = 1.0,
                      void_bgr: Optional[Tuple[int, int, int]] = None) -> ShotAnnotation:
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
            # White scanner void: flaps can be dark target paper or paper fiber tones
            fiber_tone = (random.randint(40, 75), random.randint(40, 75), random.randint(45, 80)) if random.random() < 0.5 else (205, 205, 212)
        elif void_bgr[0] < 40:
            # Dark range void: flaps can be tan paper fibers or lighter torn card edges
            fiber_tone = (
                min(255, int(config.cardboard_base_bgr[0] + random.randint(10, 30))),
                min(255, int(config.cardboard_base_bgr[1] + random.randint(10, 30))),
                min(255, int(config.cardboard_base_bgr[2] + random.randint(10, 30)))
            )
        else:
            # Tan cardboard backer: flaps are torn lighter fibers or lead-shadowed paper edges
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


def stamp_double_shot(canvas: np.ndarray, primary_center: Tuple[float, float],
                      config: TargetConfig,
                      void_bgr: Optional[Tuple[int, int, int]] = None) -> List[ShotAnnotation]:
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


def stamp_keyhole_shot(canvas: np.ndarray, center: Tuple[float, float],
                       config: TargetConfig,
                       void_bgr: Optional[Tuple[int, int, int]] = None) -> ShotAnnotation:
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


def get_random_ink_style() -> Tuple[Tuple[int, int, int], int, str]:
    """Returns random authentic pen/pencil style: (bgr_color, thickness, medium_type)."""
    medium = random.choice(["red", "blue", "black", "pencil"])
    if medium == "red":
        color = (random.randint(15, 30), random.randint(15, 35), random.randint(180, 220))
        thickness = 1
    elif medium == "blue":
        color = (random.randint(120, 160), random.randint(35, 60), random.randint(15, 30))
        thickness = 1
    elif medium == "black":
        color = (random.randint(25, 35), random.randint(25, 35), random.randint(25, 35))
        thickness = random.choice([1, 2])
    else:  # graphite pencil
        color = (random.randint(75, 95), random.randint(78, 98), random.randint(80, 100))
        thickness = 1
    return color, thickness, medium


def render_printed_target_header(canvas: np.ndarray, config: TargetConfig) -> np.ndarray:
    """Renders authentic official NRA printed target header and registration fields."""
    h, w = config.canvas_size, config.canvas_size
    sf = max(0.5, config.canvas_size / 512.0)
    top_y = int(round(18 * sf))

    bull_radius_px = 38.989 * config.pixels_per_mm
    margin_top = (h / 2.0) - bull_radius_px
    if margin_top < 30 * sf:
        return canvas

    header_font = cv2.FONT_HERSHEY_SIMPLEX
    font_s = 0.36 * sf
    color = (45, 45, 50)

    target_title = (
        "OFFICIAL 50 FT. SLOW FIRE PISTOL TARGET"
        if config.target_type.upper() == "B-2"
        else "OFFICIAL 50 FT. TIMED & RAPID FIRE PISTOL TARGET"
    )
    cv2.putText(canvas, target_title, (int(round(20 * sf)), top_y), header_font, font_s, color, 1, cv2.LINE_AA)

    fields_y = int(round(top_y + 15 * sf))
    field_text = "MATCH NO. ______  STAGE ______  RELAY ______  NAME ____________________  NRA# ________"
    cv2.putText(canvas, field_text, (int(round(20 * sf)), fields_y), header_font, font_s * 0.85, (70, 70, 75), 1, cv2.LINE_AA)
    return canvas


def render_official_nra_seal(canvas: np.ndarray, config: TargetConfig) -> np.ndarray:
    """
    Renders authentic official NRA circular seal in the lower-left target margin.
    Contains:
      - 'NONE OFFICIAL WITHOUT THIS SEAL' notice
      - Concentric high-contrast black border circles
      - Arched 'NATIONAL RIFLE ASSOCIATION' uppercase text along upper arc
      - Arched 'WASHINGTON, D.C.' and stars along inner arc
      - Eagle silhouette with spread wings and crossed rifles
    Ground truth heatmap for the seal is strictly 0.0 (negative hard mining).
    """
    sf = config.canvas_size / 1024.0

    cx = int(round(random.uniform(142.0, 168.0) * sf))
    cy = int(round(random.uniform(955.0, 985.0) * sf))
    r_outer = int(round(random.uniform(52.0, 58.0) * sf))

    if r_outer < 8:
        return canvas

    color = (random.randint(25, 45), random.randint(25, 45), random.randint(30, 50))
    thick = max(1, int(round(1.0 * sf)))
    thick_outer = max(1, int(round(1.8 * sf)))

    # Notice text: "NONE OFFICIAL WITHOUT THIS SEAL" (if space permits)
    if sf >= 0.40:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.28 * (config.canvas_size / 512.0)
        t_thick = max(1, int(round(0.9 * sf)))
        notice1 = "NONE OFFICIAL"
        notice2 = "WITHOUT THIS SEAL"
        (w1, h1), _ = cv2.getTextSize(notice1, font, font_scale, t_thick)
        (w2, h2), _ = cv2.getTextSize(notice2, font, font_scale, t_thick)
        y1 = max(12, cy - r_outer - int(round(20 * sf)))
        y2 = max(18, cy - r_outer - int(round(8 * sf)))
        cv2.putText(canvas, notice1, (max(2, cx - w1 // 2), y1), font, font_scale, color, t_thick, cv2.LINE_AA)
        cv2.putText(canvas, notice2, (max(2, cx - w2 // 2), y2), font, font_scale, color, t_thick, cv2.LINE_AA)

    # Concentric circles
    cv2.circle(canvas, (cx, cy), r_outer, color, thick_outer, lineType=cv2.LINE_AA)
    r_ring2 = max(2, r_outer - int(round(4.0 * sf)))
    cv2.circle(canvas, (cx, cy), r_ring2, color, thick, lineType=cv2.LINE_AA)

    r_inner = max(2, r_outer - int(round(20.0 * sf)))
    if r_inner > 4:
        cv2.circle(canvas, (cx, cy), r_inner, color, thick, lineType=cv2.LINE_AA)

    # Circular text along arc if resolution permits
    if sf >= 0.50:
        font = cv2.FONT_HERSHEY_SIMPLEX
        char_scale = 0.27 * (config.canvas_size / 512.0)
        r_text = (r_ring2 + r_inner) / 2.0
        text_nra = "NATIONAL RIFLE ASSOCIATION"
        for ch, ang_deg in zip(text_nra, np.linspace(-172.0, -8.0, len(text_nra))):
            if ch == ' ':
                continue
            ang_rad = math.radians(ang_deg)
            px = cx + r_text * math.cos(ang_rad)
            py = cy + r_text * math.sin(ang_rad)
            (cw, ch_h), _ = cv2.getTextSize(ch, font, char_scale, thick)
            cv2.putText(canvas, ch, (int(round(px - cw / 2)), int(round(py + ch_h / 2))), font, char_scale, color, thick, cv2.LINE_AA)

        text_dc = "WASHINGTON D.C."
        for ch, ang_deg in zip(text_dc, np.linspace(160.0, 20.0, len(text_dc))):
            if ch == ' ':
                continue
            ang_rad = math.radians(ang_deg)
            px = cx + r_text * math.cos(ang_rad)
            py = cy + r_text * math.sin(ang_rad)
            (cw, ch_h), _ = cv2.getTextSize(ch, font, char_scale * 0.9, thick)
            cv2.putText(canvas, ch, (int(round(px - cw / 2)), int(round(py + ch_h / 2))), font, char_scale * 0.9, color, thick, cv2.LINE_AA)

    # Eagle silhouette & crossed rifles
    if r_inner > 6:
        r_len = int(round(r_inner * 0.75))
        cv2.line(canvas, (cx - r_len, cy + int(round(r_len * 0.6))), (cx + r_len, cy - int(round(r_len * 0.4))), color, thick, cv2.LINE_AA)
        cv2.line(canvas, (cx - r_len, cy - int(round(r_len * 0.4))), (cx + r_len, cy + int(round(r_len * 0.6))), color, thick, cv2.LINE_AA)

        wing_w = int(round(24 * sf))
        wing_h = int(round(12 * sf))
        if wing_w > 2 and wing_h > 2:
            cv2.ellipse(canvas, (cx, cy - int(round(4 * sf))), (wing_w, wing_h), 0, 180, 360, color, -1, lineType=cv2.LINE_AA)
            cv2.ellipse(canvas, (cx, cy - int(round(8 * sf))), (max(1, int(round(4 * sf))), max(1, int(round(6 * sf)))), 0, 0, 360, color, -1, lineType=cv2.LINE_AA)
            shield_pts = np.array([
                [cx - int(round(7 * sf)), cy - int(round(2 * sf))],
                [cx + int(round(7 * sf)), cy - int(round(2 * sf))],
                [cx + int(round(5 * sf)), cy + int(round(9 * sf))],
                [cx, cy + int(round(14 * sf))],
                [cx - int(round(5 * sf)), cy + int(round(9 * sf))]
            ], dtype=np.int32)
            cv2.fillPoly(canvas, [shield_pts], color, lineType=cv2.LINE_AA)

    return canvas


def render_procedural_handwriting(
    canvas: np.ndarray,
    config: TargetConfig,
    bullseye_center: Tuple[float, float],
    ring_radii_px: Dict[str, float]
) -> np.ndarray:
    """Renders competitor handwriting (names, NRA numbers, relay details, dates, calibers)."""
    h, w = config.canvas_size, config.canvas_size
    sf = max(0.5, config.canvas_size / 512.0)
    outer_key = "6" if config.target_type.upper() == "B-3" else "4"
    outer_r = ring_radii_px.get(outer_key, config.canvas_size * 0.45)

    names = ["J. Miller", "D. Reynolds", "T. Vance", "Rob Davis", "S. Kowalski", "W. H. Smith", "M. Cooper", "B. Anderson"]
    nra_nums = [f"NRA #{random.randint(100000, 999999)}", f"NRA {random.randint(1000000, 9999999)}", f"#{random.randint(10000, 99999)}"]
    match_details = [f"M-{random.randint(1, 5)} R-{random.randint(1, 4)}", f"Match {random.randint(1, 8)}", f"Stage {random.randint(1, 3)}"]
    dates = ["10/16/26", "Oct 16", "10-16", "2026-10", "11/04/26"]
    calibers = [".22 LR", ".22 Cal", "SW 41", "Pistol .22", "Ruger MK4", "Hammerli"]

    entries = [
        random.choice(names),
        random.choice(nra_nums),
        random.choice(match_details),
        random.choice(dates),
        random.choice(calibers)
    ]
    random.shuffle(entries)
    chosen_entries = entries[:random.randint(2, 4)]

    color, thickness, _ = get_random_ink_style()
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX if random.random() < 0.60 else cv2.FONT_HERSHEY_SIMPLEX

    cx, cy = bullseye_center
    margin_candidates = [
        (int(round(250 * sf)), int(round(33 * sf))),
        (int(round(35 * sf)), int(round(33 * sf))),
        (int(round(w - 180 * sf)), int(round(33 * sf))),
        (int(round(30 * sf)), int(round(h - 22 * sf))),
        (int(round(w - 170 * sf)), int(round(h - 22 * sf))),
        (int(round(cx + outer_r * 0.70)), int(round(cy - outer_r * 0.75))),
        (int(round(cx - outer_r * 0.85)), int(round(cy + outer_r * 0.70))),
    ]
    random.shuffle(margin_candidates)

    for i, text in enumerate(chosen_entries):
        if i >= len(margin_candidates):
            break
        pos_x, pos_y = margin_candidates[i]
        pos_x = max(10, min(w - 140, pos_x))
        pos_y = max(20, min(h - 10, pos_y))
        font_scale = random.uniform(0.38, 0.52) * sf
        cv2.putText(canvas, text, (pos_x, pos_y), font, font_scale, color, thickness, cv2.LINE_AA)

    return canvas


def draw_procedural_looped_digit(
    canvas: np.ndarray,
    digit_char: str,
    center: Tuple[int, int],
    scale_px: float,
    color: Tuple[int, int, int],
    thickness: int
) -> None:
    """Draws realistic handwritten digits with closed circular loops (mimicking .22 LR holes)."""
    cx, cy = center
    sf = scale_px / 40.0
    ang = random.uniform(-18.0, 18.0)

    if digit_char == '8':
        r_top = max(2, int(round(9 * sf)))
        r_bot = max(3, int(round(11 * sf)))
        cv2.ellipse(canvas, (cx, cy - r_top + int(round(2 * sf))), (max(2, int(round(7 * sf))), r_top), ang, 0, 360, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy + r_bot - int(round(2 * sf))), (max(2, int(round(9 * sf))), r_bot), -ang, 0, 360, color, thickness, cv2.LINE_AA)
    elif digit_char == '0':
        rx = max(2, int(round(8 * sf)))
        ry = max(3, int(round(13 * sf)))
        cv2.ellipse(canvas, (cx, cy), (rx, ry), ang, 0, 360, color, thickness, cv2.LINE_AA)
    elif digit_char == '9':
        rx = max(2, int(round(7 * sf)))
        ry = max(2, int(round(8 * sf)))
        cv2.ellipse(canvas, (cx, cy - ry + int(round(2 * sf))), (rx, ry), ang, 0, 360, color, thickness, cv2.LINE_AA)
        pt1 = (int(round(cx + rx * 0.9)), cy - ry)
        pt2 = (int(round(cx + rx * 0.5)), cy + int(round(14 * sf)))
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
    elif digit_char == '6':
        rx = max(2, int(round(7 * sf)))
        ry = max(2, int(round(8 * sf)))
        cv2.ellipse(canvas, (cx, cy + ry - int(round(2 * sf))), (rx, ry), ang, 0, 360, color, thickness, cv2.LINE_AA)
        pt1 = (int(round(cx - rx * 0.9)), cy + ry)
        pt2 = (int(round(cx + rx * 0.2)), cy - int(round(14 * sf)))
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)
    elif digit_char == '3':
        r = max(2, int(round(7 * sf)))
        cv2.ellipse(canvas, (cx, cy - r + int(round(2 * sf))), (r, r), ang, -80, 140, color, thickness, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy + r - int(round(2 * sf))), (max(2, int(round(r * 1.1))), max(2, int(round(r * 1.1)))), ang, -120, 120, color, thickness, cv2.LINE_AA)
    elif digit_char == '4':
        cv2.line(canvas, (cx + int(round(3 * sf)), cy - int(round(14 * sf))), (cx - int(round(8 * sf)), cy + int(round(4 * sf))), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (cx - int(round(10 * sf)), cy + int(round(4 * sf))), (cx + int(round(8 * sf)), cy + int(round(4 * sf))), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (cx + int(round(3 * sf)), cy - int(round(6 * sf))), (cx + int(round(3 * sf)), cy + int(round(14 * sf))), color, thickness, cv2.LINE_AA)
    elif digit_char in ['X', 'x']:
        r = max(2, int(round(8 * sf)))
        cv2.line(canvas, (cx - r, cy - r), (cx + r, cy + r), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (cx - r, cy + r), (cx + r, cy - r), color, thickness, cv2.LINE_AA)
    elif digit_char == '-':
        r = max(2, int(round(6 * sf)))
        cv2.line(canvas, (cx - r, cy), (cx + r, cy), color, thickness, cv2.LINE_AA)
    else:
        cv2.putText(canvas, digit_char, (cx - int(round(6 * sf)), cy + int(round(6 * sf))), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6 * sf, color, thickness, cv2.LINE_AA)


def draw_handwritten_score_string(
    canvas: np.ndarray,
    score_str: str,
    origin: Tuple[int, int],
    scale_px: float,
    color: Tuple[int, int, int],
    thickness: int
) -> None:
    """Draws handwritten multi-character score strings with authentic looped strokes."""
    ox, oy = origin
    spacing = int(round(scale_px * 0.55))
    for i, ch in enumerate(score_str):
        px = ox + i * spacing + random.randint(-1, 1)
        py = oy + random.randint(-1, 1)
        draw_procedural_looped_digit(canvas, ch, (px, py), scale_px, color, thickness)


def render_scorer_annotations(
    canvas: np.ndarray,
    annotations: List[ShotAnnotation],
    config: TargetConfig,
    bullseye_center: Tuple[float, float],
    ring_radii_px: Dict[str, float]
) -> np.ndarray:
    """
    Renders realistic range officer / scorer markings:
      1. Hand-drawn circles around bullet holes in Red, Blue, Black, or Graphite Pencil.
      2. Written score values ("10", "9", "X", "8") adjacent to bullet holes with looped digits.
      3. Scorer score tallies in upper-right margin, outer rings, and lower margin.
      4. Ground-truth heatmap for all scorer markings is strictly 0.0 (negative hard mining).
    """
    h, w = config.canvas_size, config.canvas_size
    sf = max(0.5, config.canvas_size / 512.0)
    r_bullet_px = (config.bullet_diameter_mm / 2.0) * config.pixels_per_mm

    # 1. Circle a subset of bullet holes (simulates range officer scoring overlay check)
    if annotations:
        num_to_circle = max(1, random.randint(len(annotations) // 4, max(1, len(annotations) // 2)))
        shots_to_circle = random.sample(annotations, min(num_to_circle, len(annotations)))

        for shot in shots_to_circle:
            color, thickness, _ = get_random_ink_style()
            cx, cy = shot.center_x, shot.center_y

            circle_radius = r_bullet_px * random.uniform(1.30, 2.10)
            rx = int(round(circle_radius * random.uniform(0.90, 1.18)))
            ry = int(round(circle_radius * random.uniform(0.88, 1.15)))
            ox = int(round(cx + random.uniform(-r_bullet_px * 0.25, r_bullet_px * 0.25)))
            oy = int(round(cy + random.uniform(-r_bullet_px * 0.25, r_bullet_px * 0.25)))
            angle = random.uniform(-40.0, 40.0)
            end_angle = random.uniform(365.0, 395.0)

            cv2.ellipse(canvas, (ox, oy), (rx, ry), angle, 0, end_angle, color, thickness, cv2.LINE_AA)

            # 50% chance of adjacent ring score notation with looped digits
            if random.random() < 0.50:
                dx = cx - bullseye_center[0]
                dy = cy - bullseye_center[1]
                dist_mm = math.hypot(dx, dy) / config.pixels_per_mm
                if dist_mm <= 12.0:
                    score_str = "X" if dist_mm <= 6.0 else "10"
                elif dist_mm <= 20.0:
                    score_str = "9"
                elif dist_mm <= 30.0:
                    score_str = "8"
                elif dist_mm <= 40.0:
                    score_str = "7"
                else:
                    score_str = "6"

                offset_ang = random.uniform(0, 2 * math.pi)
                txt_x = int(round(cx + (rx + 8.0 * sf) * math.cos(offset_ang)))
                txt_y = int(round(cy + (ry + 8.0 * sf) * math.sin(offset_ang)))
                txt_x = max(10, min(config.canvas_size - 40, txt_x))
                txt_y = max(20, min(config.canvas_size - 15, txt_y))
                digit_scale = random.uniform(22.0, 32.0) * (config.canvas_size / 512.0)
                draw_handwritten_score_string(canvas, score_str, (txt_x, txt_y), digit_scale, color, thickness)

    # 2. Authentic Scorer Total Tally in Margins (e.g. "83", "94-2X", "49", "78", "96-3X")
    # 85% of targets feature authentic handwritten scores
    if random.random() < 0.85:
        tally_color, tally_thick, _ = get_random_ink_style()
        total_pts = random.choice([83, 94, 96, 49, 78, 88, 91, 95, 89, 90, 85, 98, 100])
        x_pts = random.randint(1, 4)
        tally_text = f"{total_pts}-{x_pts}X" if random.random() < 0.60 else str(total_pts)

        # Place in Upper-Right Margin (common for competition scorers), or Lower Margin
        if random.random() < 0.65:
            corner_x = int(round(w * random.uniform(0.60, 0.80)))
            corner_y = int(round(h * random.uniform(0.08, 0.22)))
        else:
            corner_x = int(round(w * random.choice([0.08, 0.70])))
            corner_y = int(round(h * random.choice([0.88, 0.93])))

        tally_scale = random.uniform(28.0, 42.0) * (config.canvas_size / 512.0)
        draw_handwritten_score_string(canvas, tally_text, (corner_x, corner_y), tally_scale, tally_color, max(1, int(round(tally_thick * 1.5))))

        if random.random() < 0.50:
            init_text = random.choice(["OK", "SC", "R.O.", "MK", "CK", "#1", "#2"])
            init_x = min(w - 30, corner_x + int(round(len(tally_text) * tally_scale * 0.60)))
            cv2.putText(canvas, init_text, (init_x, corner_y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.42 * sf, tally_color, 1, cv2.LINE_AA)

    # 3. Stray handwritten circle or checkmark on paper (simulates scorer inspecting target margin)
    if random.random() < 0.35:
        stray_color, stray_thick, _ = get_random_ink_style()
        stray_x = int(round(w * random.uniform(0.15, 0.85)))
        stray_y = int(round(h * random.uniform(0.15, 0.85)))
        stray_r = int(round(r_bullet_px * random.uniform(1.2, 1.8)))
        cv2.ellipse(canvas, (stray_x, stray_y), (stray_r, int(round(stray_r * 1.1))), random.uniform(-30, 30), 0, random.uniform(330, 390), stray_color, stray_thick, cv2.LINE_AA)

    return canvas


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

    if config.include_header_text:
        canvas = render_printed_target_header(canvas, config)

    if config.include_nra_seal:
        canvas = render_official_nra_seal(canvas, config)

    if config.include_handwriting:
        canvas = render_procedural_handwriting(canvas, config, bullseye_center, ring_radii_px)

    annotations: List[ShotAnnotation] = []
    bull_x, bull_y = bullseye_center
    target_type = config.target_type.upper()
    w = float(config.canvas_size)
    h = float(config.canvas_size)

    # Physical radial boundaries for 4-zone placement
    bull_radius_px = ring_radii_px.get("7" if target_type == "B-2" else "9", 38.989 * config.pixels_per_mm)
    outer_ring_key = "4" if target_type == "B-2" else "6"
    outer_ring_radius_px = ring_radii_px.get(
        outer_ring_key,
        93.091 * config.pixels_per_mm if target_type == "B-2" else 105.664 * config.pixels_per_mm
    )

    def sample_shot_position() -> Tuple[float, float]:
        """
        Samples bullet hole position across 4 stratified physical zones:
          - Zone 1 (40%): Inner Aiming Black (8, 9, 10, X rings)
          - Zone 2 (20%): Outer Scoring Rings (4, 5, 6, 7 on B-2; 6, 7, 8 on B-3)
          - Zone 3 (20%): On-Paper Margins Outside Rings (near headers, handwriting, tallies)
          - Zone 4 (20%): Off-Paper / Cardboard Backer & Edge-Clipping
        """
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
            else:  # top edge / corners
                sx = random.choice([
                    random.uniform(20.0 * sf, 80.0 * sf),
                    random.uniform(w - 80.0 * sf, w - 20.0 * sf)
                ])
                sy = max(28.0 * (config.canvas_size / 256.0), edge_margin * 1.5)

        # Clamp within safe canvas boundaries
        safe_m = max(6.0, 1.5 * config.pixels_per_mm)
        min_y = (20.0 * (config.canvas_size / 256.0) + 4.8 * config.heatmap_sigma_px) if config.include_header_text else safe_m
        sx = max(safe_m, min(w - safe_m, sx))
        sy = max(min_y, min(h - safe_m, sy))
        return (float(sx), float(sy))

    # 70% of targets feature mixed per-shot void colors; 30% have uniform backing
    uniform_void = sample_bullet_void_bgr(config) if random.random() < 0.30 else None

    def get_void() -> Tuple[int, int, int]:
        return uniform_void if uniform_void is not None else sample_bullet_void_bgr(config)

    for _ in range(num_singles):
        annotations.append(stamp_bullet_hole(canvas, sample_shot_position(), config, raggedness=random.uniform(0.9, 1.1), void_bgr=get_void()))
    for _ in range(num_doubles):
        annotations.extend(stamp_double_shot(canvas, sample_shot_position(), config, void_bgr=get_void()))
    for _ in range(num_keyholes):
        annotations.append(stamp_keyhole_shot(canvas, sample_shot_position(), config, void_bgr=get_void()))

    if config.include_scorer_marks:
        canvas = render_scorer_annotations(canvas, annotations, config, bullseye_center, ring_radii_px)

    heatmap = generate_ground_truth_heatmap(annotations, config)
    return canvas, heatmap, annotations, bullseye_center, ring_radii_px

