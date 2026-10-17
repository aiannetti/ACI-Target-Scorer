"""
Automated pre-inference geometric normalizer.
Locates the target aiming black in arbitrary smartphone photos, determines
perspective distortion and tilt via direct least-squares ellipse fitting,
calculates a closed-form 3x3 homography matrix, and warps the target into
an exact metric-calibrated canonical plane (e.g. 512x512 @ 4.0 px/mm or 256x256 @ 2.0 px/mm).
"""

import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class RectificationResult:
    """Diagnostic and transform metadata produced by TargetRectifier."""
    canonical_crop: np.ndarray
    homography_matrix: np.ndarray
    tilt_angle_deg: float
    raw_pixels_per_mm: float
    canonical_pixels_per_mm: float
    bullseye_center_raw: Tuple[float, float]
    semi_axes_raw: Tuple[float, float]
    ellipse_angle_deg: float
    circularity: float
    blur_metric: float
    is_fallback: bool = False
    warning: Optional[str] = None


class TargetRectifier:
    """
    Automated pre-inference geometric normalizer.
    """

    def __init__(
        self,
        target_type: str = "B-2",
        canonical_dim: int = 512,
        pixels_per_mm: float = 4.0,
        min_blur_metric: float = 75.0,
        max_tilt_deg: float = 48.0,
    ):
        self.target_type = target_type.upper()
        self.canonical_dim = canonical_dim
        self.pixels_per_mm = pixels_per_mm
        self.min_blur_metric = min_blur_metric
        self.max_tilt_deg = max_tilt_deg

        # Official NRA aiming black physical dimensions
        # B-2 (50ft Slow Fire): 3.07 in diameter (77.978 mm) -> 38.989 mm radius
        # B-3 (50ft Timed/Rapid): 3.06 in diameter (77.724 mm) -> 38.862 mm radius
        self.nominal_radius_mm = 38.862 if self.target_type == "B-3" else 38.989
        self.canonical_radius_px = self.nominal_radius_mm * self.pixels_per_mm
        self.canonical_center = (canonical_dim / 2.0, canonical_dim / 2.0)

    @staticmethod
    def load_image_with_exif(source: Union[str, Path, bytes, io.BytesIO, BinaryIO, np.ndarray]) -> np.ndarray:
        """
        Reads images across all formats (HEIC, JPEG, PNG, TIFF, BMP, WebP, in-memory arrays)
        while strictly obeying EXIF rotation metadata and sRGB color profile mapping.
        """
        from target_scorer.geometry.canonical_loader import canonicalize_image
        return canonicalize_image(source, target_format="BGR")

    # Convenient alias
    load_image = load_image_with_exif

    @staticmethod
    def evaluate_blur_metric(gray: np.ndarray) -> float:
        """Calculates variance of the 2D Laplacian operator to quantify motion blur."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def suppress_specular_glare(bgr: np.ndarray, dark_mask: np.ndarray) -> np.ndarray:
        """
        Suppresses bright flash or range lamp specular highlights inside dark target ink
        using fast Navier-Stokes morphological inpainting.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        # Specular glares are high brightness (V > 230) and low saturation (S < 45) inside black bull
        glare_mask = ((v_channel > 230) & (s_channel < 45) & (dark_mask > 0)).astype(np.uint8) * 255
        if cv2.countNonZero(glare_mask) > 10:
            # Dilate mask to consume anti-aliased perimeter pixels and restore solid dark ink
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated_mask = cv2.dilate(glare_mask, kernel, iterations=2)
            return cv2.inpaint(bgr, dilated_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        return bgr

    def _segment_bullseye_contour(self, raw_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Locates the outermost contour of the dark aiming black across variable lighting.
        Operates on an analysis pyramid for real-time responsiveness.
        """
        h, w = raw_bgr.shape[:2]
        max_dim = 1200.0
        scale = min(1.0, max_dim / max(h, w))
        work_w, work_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(raw_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur_metric = self.evaluate_blur_metric(gray)

        # Smooth high-frequency cardboard grain while preserving ring edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Segment aiming black using Otsu thresholding with adaptive fallback
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 1. Morphological opening with 5x5 kernel severs thin handwriting/pen strokes (< 5 px wide)
        # before closing, preventing handwriting from bridging to the bullseye contour
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened_mask = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, open_kernel, iterations=1)
        # 2. Morphological closing to bridge bullet holes inside the solid aiming black
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        processed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        total_pixels = work_w * work_h
        min_area = total_pixels * 0.012  # At least 1.2% of frame
        max_area = total_pixels * 0.700  # At most 70% of frame

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best_contour = None
        best_score = -1.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue

            if len(c) < 5:
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue

            solidity = area / hull_area
            if solidity < 0.88:
                continue

            # Contrast validation: mean intensity inside must be significantly darker than surround
            mask_c = np.zeros((work_h, work_w), dtype=np.uint8)
            cv2.drawContours(mask_c, [c], -1, 255, -1)
            mean_val = cv2.mean(gray, mask=mask_c)[0]

            score = (area / total_pixels) * (solidity ** 2) * (255.0 - mean_val)
            if score > best_score:
                best_score = score
                best_contour = c

        # If Otsu fails in harsh lighting, try adaptive thresholding
        if best_contour is None:
            adapt = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 41, 7
            )
            adapt_opened = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, open_kernel, iterations=1)
            adapt_closed = cv2.morphologyEx(adapt_opened, cv2.MORPH_CLOSE, close_kernel, iterations=2)
            contours_adapt, _ = cv2.findContours(adapt_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in contours_adapt:
                area = cv2.contourArea(c)
                if area < min_area or area > max_area:
                    continue
                if len(c) < 5:
                    continue
                solidity = cv2.contourArea(c) / max(1.0, cv2.contourArea(cv2.convexHull(c)))
                if solidity > 0.85:
                    best_contour = c
                    break

        if best_contour is None:
            return None, blur_metric

        # Map contour points back to raw full-resolution image coordinate space
        raw_contour = (best_contour.astype(np.float32) / scale).astype(np.float32)
        return raw_contour, blur_metric

    def _derive_homography_from_ellipse(
        self,
        center: Tuple[float, float],
        semi_major: float,
        semi_minor: float,
        angle_rad: float
    ) -> Tuple[np.ndarray, float]:
        """
        Computes the 3x3 projective homography matrix mapping the observed ellipse
        into an exact circle centered at canonical_center with radius = canonical_radius_px.
        Maintains directional basis vectors u and v to prevent spurious roll rotation.
        """
        xc, yc = center
        a = semi_major
        b = semi_minor

        # Major and minor axis orientation unit vectors
        u = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float64)
        v = np.array([-math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float64)

        # Synthesize N conjugate pairs distributed radially along the ellipse circumference
        num_conjugates = 16
        angles = np.linspace(0, 2.0 * np.pi, num_conjugates, endpoint=False)

        src_points = []
        dst_points = []

        canon_cx, canon_cy = self.canonical_center
        r_canon = self.canonical_radius_px

        for theta in angles:
            # Point on observed raw ellipse
            pt_raw = np.array([xc, yc], dtype=np.float64) + a * math.cos(theta) * u + b * math.sin(theta) * v
            src_points.append(pt_raw)

            # Corresponding point on canonical flat target circle (preserves orientation vectors u, v)
            pt_canon = np.array([
                canon_cx + r_canon * (math.cos(theta) * u[0] + math.sin(theta) * v[0]),
                canon_cy + r_canon * (math.cos(theta) * u[1] + math.sin(theta) * v[1])
            ], dtype=np.float64)
            dst_points.append(pt_canon)

        src_pts_arr = np.array(src_points, dtype=np.float32)
        dst_pts_arr = np.array(dst_points, dtype=np.float32)

        H, _ = cv2.findHomography(src_pts_arr, dst_pts_arr, cv2.RANSAC, 2.0)
        if H is None:
            H, _ = cv2.findHomography(src_pts_arr, dst_pts_arr, 0)

        # Estimate camera tilt angle (pitch off-normal)
        aspect_ratio = min(1.0, max(0.01, b / a))
        tilt_angle_deg = math.degrees(math.acos(aspect_ratio))

        return H, tilt_angle_deg

    def _fallback_aspect_preserving_crop(self, raw_bgr: np.ndarray, blur_metric: float) -> RectificationResult:
        """
        Graceful fallback when bullseye contouring fails (e.g. pre-cropped images or severe glare).
        Extracts a centered, aspect-ratio-preserved square crop without distorting circles into ovals.
        """
        h, w = raw_bgr.shape[:2]
        crop_size = min(h, w)
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        square_crop = raw_bgr[start_y:start_y + crop_size, start_x:start_x + crop_size]
        canonical_crop = cv2.resize(square_crop, (self.canonical_dim, self.canonical_dim), interpolation=cv2.INTER_AREA)

        scale_factor = self.canonical_dim / crop_size
        H_fallback = np.array([
            [scale_factor, 0.0, -start_x * scale_factor],
            [0.0, scale_factor, -start_y * scale_factor],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        return RectificationResult(
            canonical_crop=canonical_crop,
            homography_matrix=H_fallback,
            tilt_angle_deg=0.0,
            raw_pixels_per_mm=self.pixels_per_mm,
            canonical_pixels_per_mm=self.pixels_per_mm,
            bullseye_center_raw=(w / 2.0, h / 2.0),
            semi_axes_raw=(self.canonical_radius_px / scale_factor, self.canonical_radius_px / scale_factor),
            ellipse_angle_deg=0.0,
            circularity=1.0,
            blur_metric=blur_metric,
            is_fallback=True,
            warning="Target bullseye not isolated; used fallback square centering."
        )

    def process_raw_image(
        self,
        raw_bgr: Optional[np.ndarray] = None,
        image_path: Optional[Union[str, Path, bytes, io.BytesIO, BinaryIO, np.ndarray]] = None,
        image_input: Optional[Union[str, Path, bytes, io.BytesIO, BinaryIO, np.ndarray]] = None,
    ) -> RectificationResult:
        """
        End-to-end normalization pipeline. Accepts in-memory arrays, file paths, or byte streams.
        """
        source = image_input if image_input is not None else (image_path if image_path is not None else raw_bgr)
        if source is None:
            raise ValueError("TargetRectifier.process_raw_image requires raw_bgr, image_path, or image_input.")

        img_bgr = self.load_image_with_exif(source)

        h_raw, w_raw = img_bgr.shape[:2]

        contour, blur_metric = self._segment_bullseye_contour(img_bgr)
        if contour is None or len(contour) < 5:
            return self._fallback_aspect_preserving_crop(img_bgr, blur_metric)

        # Suppress specular flash hotspots inside dark target bullseye
        dark_mask = np.zeros((h_raw, w_raw), dtype=np.uint8)
        cv2.drawContours(dark_mask, [contour.astype(np.int32)], -1, 255, -1)
        img_bgr = self.suppress_specular_glare(img_bgr, dark_mask)

        # Direct Least-Squares Ellipse Fitting (Fitzgibbon-Pilu-Fisher formulation)
        try:
            pts_for_fit = contour.reshape(-1, 2).astype(np.float32)
            ellipse_box = cv2.fitEllipseDirect(pts_for_fit)
        except Exception:
            try:
                ellipse_box = cv2.fitEllipse(pts_for_fit)
            except Exception:
                return self._fallback_aspect_preserving_crop(img_bgr, blur_metric)

        (center_x, center_y), (axis_1, axis_2), angle_deg = ellipse_box

        a = max(axis_1, axis_2) / 2.0
        b = min(axis_1, axis_2) / 2.0

        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(center_x) and np.isfinite(center_y)):
            return self._fallback_aspect_preserving_crop(img_bgr, blur_metric)

        if a <= 5.0 or b <= 2.0:
            return self._fallback_aspect_preserving_crop(img_bgr, blur_metric)

        if axis_1 < axis_2:
            major_axis_angle_deg = (angle_deg + 90.0) % 180.0
        else:
            major_axis_angle_deg = angle_deg % 180.0
        major_axis_angle_rad = math.radians(major_axis_angle_deg)

        H, tilt_angle_deg = self._derive_homography_from_ellipse(
            center=(center_x, center_y),
            semi_major=a,
            semi_minor=b,
            angle_rad=major_axis_angle_rad
        )

        if H is None:
            return self._fallback_aspect_preserving_crop(img_bgr, blur_metric)

        try:
            canonical_crop = cv2.warpPerspective(
                img_bgr,
                H,
                (self.canonical_dim, self.canonical_dim),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(145, 180, 215)
            )
        except Exception:
            return self._fallback_aspect_preserving_crop(img_bgr, blur_metric)

        raw_pixels_per_mm = a / self.nominal_radius_mm
        circularity = b / a

        warning_msg = None
        if blur_metric < self.min_blur_metric:
            warning_msg = f"Target blurry (Var: {blur_metric:.1f} < {self.min_blur_metric}); steady phone."
        elif tilt_angle_deg > self.max_tilt_deg:
            warning_msg = f"Camera angle steep ({tilt_angle_deg:.1f} deg); hold camera flatter over target."

        return RectificationResult(
            canonical_crop=canonical_crop,
            homography_matrix=H,
            tilt_angle_deg=tilt_angle_deg,
            raw_pixels_per_mm=raw_pixels_per_mm,
            canonical_pixels_per_mm=self.pixels_per_mm,
            bullseye_center_raw=(center_x, center_y),
            semi_axes_raw=(a, b),
            ellipse_angle_deg=major_axis_angle_deg,
            circularity=circularity,
            blur_metric=blur_metric,
            is_fallback=False,
            warning=warning_msg
        )

    # Convenient alias for process_raw_image
    rectify = process_raw_image

    def render_diagnostic_panel(
        self,
        raw_bgr: np.ndarray,
        result: RectificationResult
    ) -> np.ndarray:
        """Creates a side-by-side diagnostic visualization panel."""
        h_raw, w_raw = raw_bgr.shape[:2]
        vis_raw = raw_bgr.copy()

        vis_scale = min(1.0, 768.0 / max(h_raw, w_raw))
        vis_w = int(round(w_raw * vis_scale))
        vis_h = int(round(h_raw * vis_scale))
        vis_raw = cv2.resize(vis_raw, (vis_w, vis_h), interpolation=cv2.INTER_AREA)

        xc, yc = result.bullseye_center_raw
        a, b = result.semi_axes_raw
        ang = result.ellipse_angle_deg

        xc_s, yc_s = int(round(xc * vis_scale)), int(round(yc * vis_scale))
        a_s, b_s = int(round(a * vis_scale)), int(round(b * vis_scale))

        if not result.is_fallback:
            cv2.ellipse(vis_raw, (xc_s, yc_s), (a_s, b_s), ang, 0, 360, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.drawMarker(vis_raw, (xc_s, yc_s), (0, 0, 255), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

            rad_ang = math.radians(ang)
            p_major_1 = (int(xc_s + a_s * math.cos(rad_ang)), int(yc_s + a_s * math.sin(rad_ang)))
            p_major_2 = (int(xc_s - a_s * math.cos(rad_ang)), int(yc_s - a_s * math.sin(rad_ang)))
            cv2.line(vis_raw, p_major_1, p_major_2, (255, 255, 0), 2, cv2.LINE_AA)

            rad_min = rad_ang + math.pi / 2.0
            p_min_1 = (int(xc_s + b_s * math.cos(rad_min)), int(yc_s + b_s * math.sin(rad_min)))
            p_min_2 = (int(xc_s - b_s * math.cos(rad_min)), int(yc_s - b_s * math.sin(rad_min)))
            cv2.line(vis_raw, p_min_1, p_min_2, (0, 165, 255), 2, cv2.LINE_AA)

        badge_text = f"RAW: Tilt {result.tilt_angle_deg:.1f} deg | Blur {result.blur_metric:.0f}"
        cv2.rectangle(vis_raw, (8, 8), (340, 38), (20, 20, 24), -1)
        cv2.putText(vis_raw, badge_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

        vis_canon = result.canonical_crop.copy()
        c_int = (int(round(self.canonical_center[0])), int(round(self.canonical_center[1])))
        r_int = int(round(self.canonical_radius_px))

        cv2.circle(vis_canon, c_int, r_int, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(vis_canon, c_int, (0, 255, 255), cv2.MARKER_CROSS, 16, 1, cv2.LINE_AA)

        canon_badge = f"CANONICAL: {self.pixels_per_mm:.1f} px/mm | R={self.nominal_radius_mm:.2f}mm"
        cv2.rectangle(vis_canon, (8, 8), (340, 38), (20, 20, 24), -1)
        cv2.putText(vis_canon, canon_badge, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1, cv2.LINE_AA)

        target_h = max(vis_h, self.canonical_dim)
        vis_raw_padded = cv2.copyMakeBorder(vis_raw, 0, target_h - vis_h, 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 18))
        vis_canon_padded = cv2.copyMakeBorder(vis_canon, 0, target_h - self.canonical_dim, 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 18))

        return np.hstack([vis_raw_padded, vis_canon_padded])
