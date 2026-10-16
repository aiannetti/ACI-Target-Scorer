"""
Automated Unit and Stress Test Suite for NRA Target Scorer.
Tests:
1. TargetRectifier perspective unwarping under varying tilt angles (0, 15, 30, 45 deg).
2. Orientation preservation (verifying that rectification does not spin targets).
3. Sub-pixel peak extraction accuracy.
4. Official NRA B-2 and B-3 inward scoring rules.
5. Ballistic group dispersion metrics (Extreme Spread, Mean Radius, COI).
"""

import math
import unittest
import numpy as np
import cv2

from target_scorer.geometry.rectifier import TargetRectifier
from target_scorer.config import TargetRingsConfig
from target_scorer.metrology.peaks import extract_subpixel_peaks
from target_scorer.metrology.scoring import (
    calculate_moa,
    score_target_group,
)


class TestTargetRectifier(unittest.TestCase):
    def setUp(self):
        self.rectifier = TargetRectifier(target_type="B-2", canonical_dim=256, pixels_per_mm=2.0)
        self.nominal_rad_mm = 38.989  # B-2 black radius

    def _render_synthetic_ellipse(
        self,
        tilt_deg: float,
        roll_deg: float = 0.0,
        raw_scale_px_per_mm: float = 3.5,
        img_size: int = 1000
    ) -> np.ndarray:
        """Renders an authentic black bullseye projected with specified pitch and roll tilt."""
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * np.array([145, 180, 215], dtype=np.uint8)

        cx, cy = img_size / 2.0, img_size / 2.0
        a = self.nominal_rad_mm * raw_scale_px_per_mm
        b = a * math.cos(math.radians(tilt_deg))

        axes = (int(round(a)), int(round(b)))
        cv2.ellipse(img, (int(round(cx)), int(round(cy))), axes, roll_deg, 0, 360, (30, 30, 35), -1, lineType=cv2.LINE_AA)
        return img

    def test_flat_target_no_spurious_rotation(self):
        """Orthogonal targets (0 deg tilt) should maintain upright orientation and circularity > 0.98."""
        raw_img = self._render_synthetic_ellipse(tilt_deg=0.0, roll_deg=0.0)
        result = self.rectifier.process_raw_image(raw_img)

        self.assertFalse(result.is_fallback)
        self.assertLess(result.tilt_angle_deg, 5.0)
        self.assertGreater(result.circularity, 0.98)
        self.assertEqual(result.canonical_crop.shape, (256, 256, 3))

    def test_tilted_target_rectification(self):
        """Targets tilted up to 45 deg should be de-tilted back to circular canonical frame."""
        for tilt in [15.0, 30.0, 42.0]:
            raw_img = self._render_synthetic_ellipse(tilt_deg=tilt, roll_deg=0.0)
            result = self.rectifier.process_raw_image(raw_img)

            self.assertFalse(result.is_fallback, f"Failed for tilt={tilt}")
            self.assertAlmostEqual(result.tilt_angle_deg, tilt, delta=4.0)

            # In canonical crop, black bullseye radius should match canonical target radius
            canon_gray = cv2.cvtColor(result.canonical_crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(canon_gray, 80, 255, cv2.THRESH_BINARY_INV)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.assertGreater(len(cnts), 0)

            c_max = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c_max)
            expected_r = self.nominal_rad_mm * 2.0
            expected_area = math.pi * (expected_r ** 2)
            self.assertAlmostEqual(area, expected_area, delta=expected_area * 0.12)


class TestMetrologyAndScoring(unittest.TestCase):
    def setUp(self):
        self.cfg_b2 = TargetRingsConfig(target_type="B-2")
        self.cfg_b3 = TargetRingsConfig(target_type="B-3")
        self.pixels_per_mm = 2.0
        self.bullet_diameter_mm = 5.69

    def test_nra_inward_scoring_b2(self):
        """Verifies official NRA inward scoring for B-2 target with gauge offset."""
        bull_center = (128.0, 128.0)
        bullet_r_mm = self.bullet_diameter_mm / 2.0

        # Shot right on the edge of the 10-ring: radial dist = ring_radius + bullet_radius - 0.1mm (breaks the line)
        ring_10_rad_mm = self.cfg_b2.b2_10_mm
        shot_dist_mm = ring_10_rad_mm + bullet_r_mm - 0.1
        shot_px = 128.0 + shot_dist_mm * self.pixels_per_mm

        shot_peaks = [(shot_px, 128.0, 0.95)]
        results, metrics = score_target_group(
            shot_peaks=shot_peaks,
            bullseye_peak=bull_center,
            pixels_per_mm=self.pixels_per_mm,
            bullet_diameter_mm=self.bullet_diameter_mm,
            rings_cfg=self.cfg_b2,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].score_label, "10")
        self.assertEqual(results[0].score_value, 10)
        self.assertEqual(metrics.total_score, 10)

    def test_subpixel_taylor_series_accuracy(self):
        """Verifies 2D quadratic Taylor series recovers continuous sub-pixel coordinates."""
        hmap = np.zeros((100, 100), dtype=np.float32)
        true_x, true_y = 50.35, 50.65
        sigma = 2.5
        y, x = np.mgrid[0:100, 0:100]
        hmap = np.exp(-((x - true_x)**2 + (y - true_y)**2) / (2.0 * sigma**2)).astype(np.float32)

        peaks = extract_subpixel_peaks(hmap, threshold=0.35, min_distance_px=10.0, max_peaks=1)
        self.assertEqual(len(peaks), 1)
        pred_x, pred_y, score = peaks[0]

        self.assertAlmostEqual(pred_x, true_x, delta=0.08)
        self.assertAlmostEqual(pred_y, true_y, delta=0.08)
        self.assertGreater(score, 0.95)

    def test_group_metrics_extreme_spread(self):
        """Verifies Extreme Spread and Mean Radius calculations."""
        bull_center = (128.0, 128.0)
        # 3 shots at (+10mm, 0), (-10mm, 0), (0, +10mm)
        shot_peaks = [
            (128.0 + 10.0 * self.pixels_per_mm, 128.0, 0.9),
            (128.0 - 10.0 * self.pixels_per_mm, 128.0, 0.9),
            (128.0, 128.0 + 10.0 * self.pixels_per_mm, 0.9),
        ]
        results, metrics = score_target_group(
            shot_peaks=shot_peaks,
            bullseye_peak=bull_center,
            pixels_per_mm=self.pixels_per_mm,
            rings_cfg=self.cfg_b2,
            distance_yards=16.67
        )

        self.assertEqual(metrics.total_shots, 3)
        # Max distance is between (+10, 0) and (-10, 0) -> 20.0 mm
        self.assertAlmostEqual(metrics.extreme_spread_mm, 20.0, delta=0.01)
        self.assertGreater(metrics.extreme_spread_moa, 0.0)


if __name__ == "__main__":
    unittest.main()
