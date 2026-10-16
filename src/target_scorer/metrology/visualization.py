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
    rect_meta: Optional[RectificationResult] = None
) -> np.ndarray:
    """Renders composite inspection image: canonical target on left, scoreboard HUD on right."""
    if rings_cfg is None:
        rings_cfg = TargetRingsConfig()

    canvas = image_bgr.copy()
    h, w, _ = canvas.shape
    bull_x, bull_y = int(round(bullseye_center[0])), int(round(bullseye_center[1]))
    bullet_radius_px = int(round((bullet_diameter_mm / 2.0) * pixels_per_mm))

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
    hud_width = 390
    hud = np.zeros((h, hud_width, 3), dtype=np.uint8)
    hud[:] = (24, 25, 28)

    target_name = f"NRA {rings_cfg.target_type.upper()} (50ft {'Slow' if rings_cfg.target_type.upper() == 'B-2' else 'Timed/Rapid'})"
    cv2.rectangle(hud, (0, 0), (hud_width, 64), (36, 38, 44), -1)
    cv2.putText(hud, target_name, (18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 215, 255), 2, cv2.LINE_AA)

    if rect_meta is not None:
        sub_info = f"Tilt: {rect_meta.tilt_angle_deg:.1f} deg | Calib: {pixels_per_mm:.1f} px/mm | Blur: {rect_meta.blur_metric:.0f}"
    else:
        sub_info = f"Distance: 50.0 ft ({metrics.target_distance_yards:.1f} yd) | Cal: {bullet_diameter_mm:.2f}mm"
    cv2.putText(hud, sub_info, (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 170), 1, cv2.LINE_AA)

    # Score Summary Banner
    cv2.rectangle(hud, (16, 75), (hud_width - 16, 145), (32, 34, 40), -1)
    cv2.rectangle(hud, (16, 75), (hud_width - 16, 145), (60, 65, 75), 1)

    score_str = f"{metrics.total_score} - {metrics.x_count}X"
    cv2.putText(hud, "TOTAL SCORE", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 190), 1, cv2.LINE_AA)
    cv2.putText(hud, score_str, (30, 132), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 255, 120), 2, cv2.LINE_AA)

    max_str = f"/ {metrics.max_possible_score}"
    cv2.putText(hud, max_str, (215, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (140, 140, 150), 1, cv2.LINE_AA)

    # Ballistic Group Statistics block
    y_pos = 172
    cv2.putText(hud, "BALLISTIC GROUP METRICS", (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.line(hud, (18, y_pos + 6), (hud_width - 18, y_pos + 6), (50, 55, 65), 1)
    y_pos += 26

    def draw_stat_row(label: str, val_str: str, curr_y: int) -> int:
        cv2.putText(hud, label, (22, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 175, 185), 1, cv2.LINE_AA)
        cv2.putText(hud, val_str, (hud_width - 190, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        return curr_y + 22

    y_pos = draw_stat_row("Shot Count:", f"{metrics.total_shots}", y_pos)
    y_pos = draw_stat_row("Extreme Spread:", f"{metrics.extreme_spread_mm:.2f} mm ({metrics.extreme_spread_moa:.2f} MOA)", y_pos)
    y_pos = draw_stat_row("Mean Radius:", f"{metrics.mean_radius_mm:.2f} mm ({metrics.mean_radius_moa:.2f} MOA)", y_pos)
    y_pos = draw_stat_row("Farthest Pair:", f"Shot #{metrics.farthest_pair[0]} <-> #{metrics.farthest_pair[1]}", y_pos)
    y_pos = draw_stat_row("Windage Bias:", f"{metrics.windage_bias_mm:+.2f} mm ({'Right' if metrics.windage_bias_mm >= 0 else 'Left'})", y_pos)
    y_pos = draw_stat_row("Elevation Bias:", f"{metrics.elevation_bias_mm:+.2f} mm ({'High' if metrics.elevation_bias_mm >= 0 else 'Low'})", y_pos)

    y_pos += 12
    cv2.putText(hud, "INDIVIDUAL SHOT LOG", (18, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 215, 255), 1, cv2.LINE_AA)
    cv2.line(hud, (18, y_pos + 6), (hud_width - 18, y_pos + 6), (50, 55, 65), 1)
    y_pos += 24

    cv2.putText(hud, "SHOT", (22, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 135, 145), 1, cv2.LINE_AA)
    cv2.putText(hud, "SCORE", (82, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 135, 145), 1, cv2.LINE_AA)
    cv2.putText(hud, "DIST (MM)", (152, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 135, 145), 1, cv2.LINE_AA)
    cv2.putText(hud, "GAUGE (MM)", (245, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 135, 145), 1, cv2.LINE_AA)
    y_pos += 18

    for s in shot_results[:12]:
        row_color = (0, 215, 255) if s.is_inner_x else ((0, 255, 120) if s.score_value == 10 else (220, 225, 235))
        cv2.putText(hud, f"#{s.shot_id:02d}", (22, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, row_color, 1, cv2.LINE_AA)
        cv2.putText(hud, f"{s.score_label:>3}", (85, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, row_color, 1, cv2.LINE_AA)
        cv2.putText(hud, f"{s.radial_dist_mm:6.2f}", (155, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 210), 1, cv2.LINE_AA)
        cv2.putText(hud, f"{s.effective_gauge_dist_mm:6.2f}", (250, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 210), 1, cv2.LINE_AA)
        y_pos += 18

    return np.hstack([canvas, hud])
