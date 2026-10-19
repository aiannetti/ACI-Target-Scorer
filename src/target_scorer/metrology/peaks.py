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


def is_spurious_margin_marking(
    px: float,
    py: float,
    canonical_bgr: np.ndarray,
    pixels_per_mm: float = 4.0
) -> bool:
    """
    Checks if a candidate peak in margins or outer regions is a spurious marking:
      - Colored ballpoint pen or grading ink (blue, violet, red)
      - Thin 1D stroke lines / checkmarks / underscores (aspect ratio > 2.2)
      - Blank paper / printed scoring ring line without authentic bullet penetration void
      - Preserves genuine 2D circular bullet hole penetration voids (dark, tan, or scanner-white).
    """
    h, w = canonical_bgr.shape[:2]
    border_px = int(round(4.0 * pixels_per_mm))
    if px < border_px or px > w - border_px or py < border_px or py > h - border_px:
        return True

    r_check = int(round(3.5 * pixels_per_mm))
    ix, iy = int(round(px)), int(round(py))
    if iy - r_check < 0 or iy + r_check >= h or ix - r_check < 0 or ix + r_check >= w:
        return True

    patch = canonical_bgr[iy - r_check:iy + r_check + 1, ix - r_check:ix + r_check + 1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    paper_median = float(np.median(gray))

    # Inside solid black bullseye, preserve candidate peaks
    if paper_median < 90:
        return False

    b = patch[:, :, 0].astype(float)
    g = patch[:, :, 1].astype(float)
    r = patch[:, :, 2].astype(float)

    # 1. Ballpoint blue / violet or red ink test
    ink_mask = gray < 165
    if np.sum(ink_mask) >= 15:
        ink_b = b[ink_mask]
        ink_g = g[ink_mask]
        ink_r = r[ink_mask]
        if np.mean(ink_b - ink_g) > 7.0:
            return True  # Blue / purple ballpoint ink handwriting
        if np.mean(ink_r - ink_b) > 25.0 and np.mean(ink_r - ink_g) > 20.0:
            return True  # Red grading pen handwriting

    # 2. Check for authentic bullet hole penetration void
    cy_p, cx_p = patch.shape[0] // 2, patch.shape[1] // 2
    center_disk = gray[max(0, cy_p - 3):min(patch.shape[0], cy_p + 4), max(0, cx_p - 3):min(patch.shape[1], cx_p + 4)]
    c_mean = float(center_disk.mean())
    c_min = int(center_disk.min())

    # 2a. Authentic dark void or tan backer void (center significantly darker than paper)
    if c_mean < paper_median - 25 or c_min < paper_median - 45:
        dark_mask = (gray < (paper_median - 25)).astype(np.uint8)
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 12:
                bx_c, by_c, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / max(1, min(bw, bh))
                if aspect > 2.2:
                    return True  # Elongated 1D stroke / text line
        return False  # Genuine circular dark bullet void!

    # 2b. Authentic scanner white void with grease ring
    if c_mean > paper_median + 4:
        r_lead = 2.6 * pixels_per_mm
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        perim_samples = [gray[int(round(cy_p + r_lead * np.sin(a))), int(round(cx_p + r_lead * np.cos(a)))]
                         for a in angles if 0 <= cy_p + r_lead * np.sin(a) < patch.shape[0] and 0 <= cx_p + r_lead * np.cos(a) < patch.shape[1]]
        if perim_samples and min(perim_samples) < paper_median - 25 and np.mean(perim_samples) < paper_median - 8:
            return False  # Authentic scanner white bullet hole!
        return True  # White patch lacking bullet grease ring

    # 2c. Untouched blank paper / faint printed ring line
    return True


is_handwriting_pen_stroke = is_spurious_margin_marking


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
      1. Extreme canvas borders (within 4mm / 16px of canvas edge).
      2. Official NRA seal / emblem printed in lower-left margin (x: 50..260, y: 870..1024).
      3. Top header registration / title text (y < 65).
      4. Scorer handwritten score numerals in margins (ballpoint pen ink).
      5. Blank paper / printed scoring ring line peaks lacking physical bullet void penetration.
    Returns cleaned list of peaks sorted by confidence, limited to max_peaks if specified.
    """
    if not peaks:
        return []

    sf = canvas_size / 1024.0
    border_px = 4.0 * pixels_per_mm
    seal_x0, seal_x1 = 50.0 * sf, 260.0 * sf
    seal_y0, seal_y1 = 870.0 * sf, float(canvas_size)
    header_y = 65.0 * sf

    cleaned: List[Tuple[float, float, float]] = []
    for px, py, conf in peaks:
        # 0. Extreme canvas boundary exclusion
        if px < border_px or px > canvas_size - border_px or py < border_px or py > canvas_size - border_px:
            continue

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

        # 3. Margin handwriting and void verification exclusion
        if filter_handwriting and canonical_bgr is not None:
            dx = px - (canvas_size / 2.0)
            dy = py - (canvas_size / 2.0)
            dist_center = math.hypot(dx, dy)
            outer_margin_px = 350.0 * sf
            if (dist_center > outer_margin_px or (py < 250.0 * sf and px > 520.0 * sf) or py < 120.0 * sf):
                if is_spurious_margin_marking(px, py, canonical_bgr, pixels_per_mm):
                    continue

        cleaned.append((px, py, conf))

    cleaned.sort(key=lambda p: p[2], reverse=True)
    if max_peaks is not None and len(cleaned) > max_peaks:
        cleaned = cleaned[:max_peaks]

    return cleaned
