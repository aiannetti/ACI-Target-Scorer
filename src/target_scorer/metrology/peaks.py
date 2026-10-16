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
