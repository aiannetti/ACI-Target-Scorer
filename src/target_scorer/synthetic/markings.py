"""
Procedural synthesis of official printed target headers, NRA official seals, and scorer handwriting.
"""

import math
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np

from target_scorer.config import ShotAnnotation, TargetConfig


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
    if random.random() < 0.85:
        tally_color, tally_thick, _ = get_random_ink_style()
        total_pts = random.choice([83, 94, 96, 49, 78, 88, 91, 95, 89, 90, 85, 98, 100])
        x_pts = random.randint(1, 4)
        tally_text = f"{total_pts}-{x_pts}X" if random.random() < 0.60 else str(total_pts)

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

    # 3. Stray handwritten circle or checkmark on paper
    if random.random() < 0.35:
        stray_color, stray_thick, _ = get_random_ink_style()
        stray_x = int(round(w * random.uniform(0.15, 0.85)))
        stray_y = int(round(h * random.uniform(0.15, 0.85)))
        stray_r = int(round(r_bullet_px * random.uniform(1.2, 1.8)))
        cv2.ellipse(canvas, (stray_x, stray_y), (stray_r, int(round(stray_r * 1.1))), random.uniform(-30, 30), 0, random.uniform(330, 390), stray_color, stray_thick, cv2.LINE_AA)

    return canvas
