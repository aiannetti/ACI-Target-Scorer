"""
Sub-pixel peak extraction via 2D quadratic Taylor-series expansion.
"""

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


def extract_subpixel_peaks(
    heatmap: np.ndarray,
    threshold: float = 0.45,
    min_distance_px: float = 10.0,
    max_peaks: Optional[int] = 10,
    border_margin: int = 4
) -> List[Tuple[float, float, float]]:
    """
    Extracts continuous coordinate centers using:
      1. Fast Non-Maximum Suppression with 5x5 morphological dilation.
      2. Threshold filtering.
      3. Greedy spatial suppression.
      4. Sorting by confidence score.
      5. 2D quadratic Taylor series expansion for continuous sub-pixel coordinates.
    """
    if heatmap.dtype != np.float32:
        heatmap = heatmap.astype(np.float32)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(heatmap, kernel)
    peaks_mask = ((heatmap == dilated) & (heatmap > threshold)).astype(np.uint8)

    if not np.any(peaks_mask):
        return []

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks_mask)
    h, w = heatmap.shape

    candidates: List[Tuple[int, int, float, float, float, int]] = []
    for idx in range(1, num_labels):
        cx, cy = centroids[idx]
        x_int = int(round(cx))
        y_int = int(round(cy))
        if (
            y_int <= border_margin or y_int >= h - border_margin or
            x_int <= border_margin or x_int >= w - border_margin
        ):
            continue
        score = float(heatmap[y_int, x_int])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        candidates.append((y_int, x_int, score, float(cx - x_int), float(cy - y_int), area))

    candidates.sort(key=lambda c: c[2], reverse=True)

    suppressed_peaks: List[Tuple[int, int, float, float, float, int]] = []
    for y_c, x_c, score, cx_off, cy_off, area in candidates:
        too_close = False
        for y_acc, x_acc, _, _, _, _ in suppressed_peaks:
            dist = math.hypot(x_c - x_acc, y_c - y_acc)
            if dist < min_distance_px:
                too_close = True
                break
        if not too_close:
            suppressed_peaks.append((y_c, x_c, score, cx_off, cy_off, area))
            if max_peaks is not None and len(suppressed_peaks) >= max_peaks:
                break

    subpixel_points: List[Tuple[float, float, float]] = []
    for y_int, x_int, score, cx_off, cy_off, area in suppressed_peaks:
        patch = heatmap[y_int - 1:y_int + 2, x_int - 1:x_int + 2].astype(np.float64)

        dx = 0.5 * (patch[1, 2] - patch[1, 0])
        dy = 0.5 * (patch[2, 1] - patch[0, 1])

        dxx = patch[1, 2] - 2.0 * patch[1, 1] + patch[1, 0]
        dyy = patch[2, 1] - 2.0 * patch[1, 1] + patch[0, 1]
        dxy = 0.25 * (patch[2, 2] - patch[2, 0] - patch[0, 2] + patch[0, 0])

        det = dxx * dyy - dxy * dxy
        if area <= 4 and det > 1e-6 and dxx < 0 and dyy < 0:
            inv_hessian = (1.0 / det) * np.array([[dyy, -dxy], [-dxy, dxx]])
            grad = np.array([dx, dy], dtype=np.float64)
            offset = -inv_hessian @ grad
            offset_x = float(np.clip(offset[0], -0.9, 0.9))
            offset_y = float(np.clip(offset[1], -0.9, 0.9))
        else:
            offset_x, offset_y = cx_off, cy_off

        refined_x = x_int + offset_x
        refined_y = y_int + offset_y
        subpixel_points.append((refined_x, refined_y, score))

    return subpixel_points


def is_handwriting_pen_stroke(
    px: float,
    py: float,
    canonical_bgr: np.ndarray,
    pixels_per_mm: float = 4.0
) -> bool:
    """Checks if a candidate peak in margins is colored ballpoint pen or grading ink."""
    h, w = canonical_bgr.shape[:2]
    r_check = int(round(3.0 * pixels_per_mm))
    ix, iy = int(round(px)), int(round(py))
    if iy - r_check < 0 or iy + r_check >= h or ix - r_check < 0 or ix + r_check >= w:
        return False

    patch = canonical_bgr[iy - r_check:iy + r_check + 1, ix - r_check:ix + r_check + 1]
    b = patch[:, :, 0].astype(float)
    g = patch[:, :, 1].astype(float)
    r = patch[:, :, 2].astype(float)

    # Ballpoint blue / violet ink test:
    # Target cardstock has R > G > B. Blue ink has B noticeably higher than G.
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    ink_mask = gray < 165
    if np.sum(ink_mask) >= 15:
        ink_b = b[ink_mask]
        ink_g = g[ink_mask]
        ink_r = r[ink_mask]
        if np.mean(ink_b - ink_g) > 8.0:
            return True  # Blue / purple ballpoint ink handwriting

        # Red grading pen test:
        if np.mean(ink_r - ink_b) > 25.0 and np.mean(ink_r - ink_g) > 20.0:
            return True  # Red grading pen handwriting

    return False


def filter_spurious_peaks(
    peaks: List[Tuple[float, float, float]],
    canonical_bgr: Optional[np.ndarray] = None,
    canvas_size: int = 1024,
    pixels_per_mm: float = 4.0,
    target_type: str = "B-2",
    max_peaks: Optional[int] = 10,
    filter_seal: bool = True,
    filter_header: bool = True,
    filter_handwriting: bool = True
) -> List[Tuple[float, float, float]]:
    """
    Filters spurious false-positive detections occurring on printed target artifacts:
      1. Official NRA seal / emblem printed in lower-left margin (x: 50..260, y: 870..1024).
      2. Top header registration / title text (y < 42).
      3. Scorer handwritten score numerals in margins (ballpoint pen ink).
      4. Verifies true bullet hole radial void properties if canonical_bgr is provided.
    Returns cleaned list of peaks sorted by confidence, limited to max_peaks if specified.
    """
    if not peaks:
        return []

    sf = canvas_size / 1024.0
    seal_x0, seal_x1 = 50.0 * sf, 260.0 * sf
    seal_y0, seal_y1 = 870.0 * sf, float(canvas_size)
    header_y = 42.0 * sf

    cleaned: List[Tuple[float, float, float]] = []
    for px, py, conf in peaks:
        # 1. Top printed header exclusion
        if filter_header and py < header_y and conf < 0.90:
            continue

        # 2. Lower-left NRA seal exclusion
        if filter_seal and (seal_x0 <= px <= seal_x1 and seal_y0 <= py <= seal_y1):
            if canonical_bgr is not None:
                r_hole = int(round(2.85 * pixels_per_mm))
                ix, iy = int(round(px)), int(round(py))
                h, w = canonical_bgr.shape[:2]
                if iy - r_hole < 0 or iy + r_hole >= h or ix - r_hole < 0 or ix + r_hole >= w:
                    continue
                # In seal zone, spurious text/ring peaks have conf < 0.95
                if conf < 0.95:
                    continue
            else:
                if conf < 0.90:
                    continue

        # 3. Margin handwriting exclusion (upper-right score tally or outer margin)
        if filter_handwriting and canonical_bgr is not None:
            dx = px - (canvas_size / 2.0)
            dy = py - (canvas_size / 2.0)
            dist_center = math.hypot(dx, dy)
            outer_margin_px = 350.0 * sf
            if (dist_center > outer_margin_px or (py < 250.0 * sf and px > 520.0 * sf)):
                if is_handwriting_pen_stroke(px, py, canonical_bgr, pixels_per_mm):
                    continue

        cleaned.append((px, py, conf))

    cleaned.sort(key=lambda p: p[2], reverse=True)
    if max_peaks is not None and len(cleaned) > max_peaks:
        cleaned = cleaned[:max_peaks]

    return cleaned
