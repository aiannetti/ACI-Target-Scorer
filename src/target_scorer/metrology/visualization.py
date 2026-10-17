"""
Visual diagnostic HUD and scorecard rendering.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from target_scorer.config import TargetRingsConfig
from target_scorer.geometry.rectifier import RectificationResult
from target_scorer.metrology.scoring import GroupMetrics, ShotResult


def render_scoring_inspection_overlay(
    image_bgr: np.ndarray,
    bullet_heatmap: np.ndarray,
    shot_results: List[ShotResult],
    bullseye_center: Tuple[float, float],
    metrics: GroupMetrics,
    pixels_per_mm: float,
    bullet_diameter_mm: float = 5.69,
    rings_cfg: Optional[TargetRingsConfig] = None,
    rect_meta: Optional[RectificationResult] = None,
    redact_pii: bool = False
) -> np.ndarray:
    """Renders composite inspection image: canonical target on left, scoreboard HUD on right."""
    if rings_cfg is None:
        rings_cfg = TargetRingsConfig()

    canvas = image_bgr.copy()
    h, w, _ = canvas.shape
    bull_x, bull_y = int(round(bullseye_center[0])), int(round(bullseye_center[1]))
    bullet_radius_px = max(1, int(round((bullet_diameter_mm / 2.0) * pixels_per_mm)))

    if redact_pii:
        # Mask out outer margin registration zones where competitor PII is written
        margin_mask = np.zeros((h, w), dtype=np.uint8)
        rings_table = rings_cfg.get_rings_table()
        outer_margin_rad = int(round(rings_table[-1][1] * pixels_per_mm + 12.0 * (pixels_per_mm / 4.0)))
        cv2.circle(margin_mask, (bull_x, bull_y), outer_margin_rad, 255, -1)
        header_h = max(28, int(round(bull_y - outer_margin_rad + 8)))
        if header_h > 0:
            margin_mask[:header_h, :] = 0

        blurred_canvas = cv2.GaussianBlur(canvas, (45, 45), 25.0)
        pii_zone = (margin_mask == 0)[:, :, np.newaxis]
        canvas = np.where(pii_zone, blurred_canvas, canvas)

        badge_y = max(18, int(round(header_h * 0.6)))
        cv2.putText(
            canvas, "[COMPETITOR REGISTRATION REDACTED]", (18, badge_y),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.32, 0.38 * (h / 512.0)), (130, 130, 140), 1, cv2.LINE_AA
        )

    # Ensure bullet heatmap matches canvas dimensions
    if bullet_heatmap.shape[:2] != (h, w):
        bullet_heatmap = cv2.resize(bullet_heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # 1. Overlay semi-transparent heat trace
    hmap_uint8 = np.clip(bullet_heatmap * 255.0, 0, 255).astype(np.uint8)
    hmap_color = cv2.applyColorMap(hmap_uint8, cv2.COLORMAP_JET)
    heat_mask = (bullet_heatmap > 0.15).astype(np.float32)[:, :, np.newaxis]
    canvas = (canvas * (1.0 - 0.35 * heat_mask) + hmap_color * (0.35 * heat_mask)).astype(np.uint8)

    # 2. Draw official NRA target scoring ring contours
    black_rad_px = int(round(rings_cfg.get_black_radius_mm() * pixels_per_mm))
    cv2.circle(canvas, (bull_x, bull_y), black_rad_px, (100, 100, 110), 1, lineType=cv2.LINE_AA)

    for label, radius_mm, _ in rings_cfg.get_rings_table():
        r_px = int(round(radius_mm * pixels_per_mm))
        ring_color = (200, 200, 200) if r_px <= black_rad_px else (70, 70, 80)
        cv2.circle(canvas, (bull_x, bull_y), r_px, ring_color, 1, lineType=cv2.LINE_AA)

    # Bullseye center crosshair
    cv2.drawMarker(canvas, (bull_x, bull_y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=1, line_type=cv2.LINE_AA)

    # 3. Draw Extreme Spread chord line between farthest shots
    if metrics.total_shots >= 2 and metrics.extreme_spread_mm > 0:
        p1 = next((s.center_px for s in shot_results if s.shot_id == metrics.farthest_pair[0]), None)
        p2 = next((s.center_px for s in shot_results if s.shot_id == metrics.farthest_pair[1]), None)
        if p1 is not None and p2 is not None:
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0])), int(round(p2[1])))
            cv2.line(canvas, pt1, pt2, (0, 140, 255), 2, lineType=cv2.LINE_AA)

    # 4. Render shot markers and score badges
    for shot in shot_results:
        cx, cy = int(round(shot.center_px[0])), int(round(shot.center_px[1]))

        if shot.is_inner_x:
            gauge_color = (0, 215, 255)    # Gold
        elif shot.score_value == 10:
            gauge_color = (0, 255, 0)      # Green
        elif shot.score_value == 9:
            gauge_color = (255, 200, 0)    # Cyan
        elif shot.score_value >= 7:
            gauge_color = (0, 165, 255)    # Orange
        else:
            gauge_color = (0, 0, 255)      # Red

        cv2.circle(canvas, (cx, cy), bullet_radius_px, gauge_color, 2, lineType=cv2.LINE_AA)
        cv2.drawMarker(canvas, (cx, cy), gauge_color, markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1, line_type=cv2.LINE_AA)

        badge_text = f"#{shot.shot_id}:{shot.score_label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        (t_w, t_h), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)

        bx = cx + bullet_radius_px + 4
        by = cy - 4
        bx = min(w - t_w - 8, max(4, bx))
        by = min(h - 4, max(t_h + 8, by))

        cv2.rectangle(canvas, (bx - 2, by - t_h - 4), (bx + t_w + 3, by + 3), (20, 20, 24), -1)
        cv2.rectangle(canvas, (bx - 2, by - t_h - 4), (bx + t_w + 3, by + 3), gauge_color, 1)
        cv2.putText(canvas, badge_text, (bx, by), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 5. Right-Hand HUD panel
    sf = max(0.5, min(1.45, h / 512.0))
    hud_width = int(round(390 * max(1.0, sf * 0.90)))
    hud = np.zeros((h, hud_width, 3), dtype=np.uint8)
    hud[:] = (24, 25, 28)

    target_name = f"NRA {rings_cfg.target_type.upper()} (50ft {'Slow' if rings_cfg.target_type.upper() == 'B-2' else 'Timed/Rapid'})"
    hdr_h = int(round(64 * sf))
    cv2.rectangle(hud, (0, 0), (hud_width, hdr_h), (36, 38, 44), -1)
    cv2.putText(hud, target_name, (18, int(round(26 * sf))), cv2.FONT_HERSHEY_SIMPLEX, 0.52 * sf, (0, 215, 255), max(1, int(round(2 * sf))), cv2.LINE_AA)

    if rect_meta is not None:
        sub_info = f"Tilt: {rect_meta.tilt_angle_deg:.1f} deg | Calib: {pixels_per_mm:.1f} px/mm | Blur: {rect_meta.blur_metric:.0f}"
    else:
        sub_info = f"Distance: 50.0 ft ({metrics.target_distance_yards:.1f} yd) | Cal: {bullet_diameter_mm:.2f}mm"
    cv2.putText(hud, sub_info, (18, int(round(48 * sf))), cv2.FONT_HERSHEY_SIMPLEX, max(0.30, 0.34 * sf), (160, 160, 170), 1, cv2.LINE_AA)

    # Score Summary Banner
    b_top = int(round(75 * sf))
    b_bot = int(round(145 * sf))
    cv2.rectangle(hud, (16, b_top), (hud_width - 16, b_bot), (32, 34, 40), -1)
    cv2.rectangle(hud, (16, b_top), (hud_width - 16, b_bot), (60, 65, 75), 1)

    if rings_cfg.target_type.upper() == "B-2":
        score_str = f"{metrics.total_score}"
        max_x = int(round(30 + 115 * sf))
    else:
        score_str = f"{metrics.total_score} - {metrics.x_count}X"
        max_x = int(round(30 + 185 * sf))

    cv2.putText(hud, "TOTAL SCORE", (30, int(round(95 * sf))), cv2.FONT_HERSHEY_SIMPLEX, max(0.32, 0.40 * sf), (180, 180, 190), 1, cv2.LINE_AA)
    cv2.putText(hud, score_str, (30, int(round(132 * sf))), cv2.FONT_HERSHEY_SIMPLEX, max(0.65, 1.15 * sf), (0, 255, 120), max(1, int(round(2 * sf))), cv2.LINE_AA)

    max_str = f"/ {metrics.max_possible_score}"
    cv2.putText(hud, max_str, (max_x, int(round(130 * sf))), cv2.FONT_HERSHEY_SIMPLEX, max(0.42, 0.60 * sf), (140, 140, 150), 1, cv2.LINE_AA)

    # Ballistic Group Statistics block
    y_pos = int(round(172 * sf))
    cv2.putText(hud, "BALLISTIC GROUP METRICS", (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.48 * sf), (0, 215, 255), 1, cv2.LINE_AA)
    cv2.line(hud, (18, y_pos + int(round(5 * sf))), (hud_width - 18, y_pos + int(round(5 * sf))), (50, 55, 65), 1)
    row_h = int(round(22 * sf))
    y_pos += int(round(24 * sf))

    val_x = int(round(hud_width * 0.48))

    def draw_stat_row(label: str, val_str: str, curr_y: int) -> int:
        font_s = max(0.30, 0.40 * sf)
        cv2.putText(hud, label, (22, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_s, (170, 175, 185), 1, cv2.LINE_AA)
        cv2.putText(hud, val_str, (val_x, curr_y), cv2.FONT_HERSHEY_SIMPLEX, font_s, (255, 255, 255), 1, cv2.LINE_AA)
        return curr_y + row_h

    y_pos = draw_stat_row("Shot Count:", f"{metrics.total_shots}", y_pos)
    y_pos = draw_stat_row("Extreme Spread:", f"{metrics.extreme_spread_mm:.2f} mm ({metrics.extreme_spread_moa:.2f} MOA)", y_pos)
    y_pos = draw_stat_row("Mean Radius:", f"{metrics.mean_radius_mm:.2f} mm ({metrics.mean_radius_moa:.2f} MOA)", y_pos)
    y_pos = draw_stat_row("Farthest Pair:", f"Shot #{metrics.farthest_pair[0]} <-> #{metrics.farthest_pair[1]}", y_pos)
    y_pos = draw_stat_row("Windage Bias:", f"{metrics.windage_bias_mm:+.2f} mm ({'Right' if metrics.windage_bias_mm >= 0 else 'Left'})", y_pos)
    y_pos = draw_stat_row("Elevation Bias:", f"{metrics.elevation_bias_mm:+.2f} mm ({'High' if metrics.elevation_bias_mm >= 0 else 'Low'})", y_pos)

    if h >= 256:
        y_pos += int(round(8 * sf))
        cv2.putText(hud, "INDIVIDUAL SHOT LOG", (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, max(0.35, 0.48 * sf), (0, 215, 255), 1, cv2.LINE_AA)
        cv2.line(hud, (18, y_pos + int(round(5 * sf))), (hud_width - 18, y_pos + int(round(5 * sf))), (50, 55, 65), 1)
        y_pos += int(round(20 * sf))

        hdr_font_s = max(0.30, 0.38 * sf)
        c1, c2, c3, c4 = 22, int(round(hud_width * 0.22)), int(round(hud_width * 0.40)), int(round(hud_width * 0.65))
        cv2.putText(hud, "SHOT", (c1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, hdr_font_s, (130, 135, 145), 1, cv2.LINE_AA)
        cv2.putText(hud, "SCORE", (c2, y_pos), cv2.FONT_HERSHEY_SIMPLEX, hdr_font_s, (130, 135, 145), 1, cv2.LINE_AA)
        cv2.putText(hud, "DIST (MM)", (c3, y_pos), cv2.FONT_HERSHEY_SIMPLEX, hdr_font_s, (130, 135, 145), 1, cv2.LINE_AA)
        cv2.putText(hud, "GAUGE (MM)", (c4, y_pos), cv2.FONT_HERSHEY_SIMPLEX, hdr_font_s, (130, 135, 145), 1, cv2.LINE_AA)
        shot_row_h = int(round(18 * sf))
        y_pos += shot_row_h

        max_rows = 15 if h >= 800 else (12 if h >= 512 else 5)
        for s in shot_results[:max_rows]:
            if y_pos > h - 4:
                break
            row_color = (0, 215, 255) if s.is_inner_x else ((0, 255, 120) if s.score_value == 10 else (220, 225, 235))
            cv2.putText(hud, f"#{s.shot_id:02d}", (c1, y_pos), cv2.FONT_HERSHEY_SIMPLEX, hdr_font_s, row_color, 1, cv2.LINE_AA)
            cv2.putText(hud, f"{s.score_label:>3}", (c2 + 4, y_pos), cv2.FONT_HERSHEY_SIMPLEX, hdr_font_s, row_color, 1, cv2.LINE_AA)
            cv2.putText(hud, f"{s.radial_dist_mm:6.2f}", (c3 + 4, y_pos), cv2.FONT_HERSHEY_SIMPLEX, hdr_font_s, (200, 200, 210), 1, cv2.LINE_AA)
            cv2.putText(hud, f"{s.effective_gauge_dist_mm:6.2f}", (c4 + 4, y_pos), cv2.FONT_HERSHEY_SIMPLEX, hdr_font_s, (200, 200, 210), 1, cv2.LINE_AA)
            y_pos += shot_row_h

    return np.hstack([canvas, hud])

