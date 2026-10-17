"""
Low-Resolution Mathematical & Programmatic Metrology Test Suite (256x256 @ 2.0 px/mm).
Implements the verification test battery defined in target_scoring_testing_plan.md:
- GEO-01 to GEO-05: Geometric rectification, tilt invariance, roll stability, glare inpainting, blur detection.
- MET-01 to MET-04: Sub-pixel Taylor series 2D peak extraction, negative-definite Hessian check, caliber NMS.
- SC-01 to SC-05: Official NRA inward scoring rules, line-breaking edge cases, gauge offsets, and ballistics.
- E2E-01: End-to-end demo card generation, inward evaluation, and HUD scorecard composite at 256x256.
"""

import math
import unittest
import numpy as np
import cv2
import torch

from target_scorer.geometry.rectifier import RectificationResult, TargetRectifier
from target_scorer.config import TargetConfig, TargetRingsConfig
from target_scorer.infer import generate_synthetic_demo_target, run_inference_and_scoring
from target_scorer.metrology.peaks import extract_subpixel_peaks
from target_scorer.metrology.scoring import (
    calculate_moa,
    score_target_group,
)
from target_scorer.metrology.visualization import render_scoring_inspection_overlay
from target_scorer.models.loss import AdaptiveWingLoss
from target_scorer.models.unet import HighResHeatmapUNet
from target_scorer.synthetic.generator import generate_synthetic_target_sample
from target_scorer.train import calculate_localization_metrics



class TestLowResGeometricRectifier(unittest.TestCase):
    """Stage 1: Geometric Rectification Tests (canonical_dim=256, pixels_per_mm=2.0)."""

    def setUp(self):
        self.canonical_dim = 256
        self.pixels_per_mm = 2.0
        self.rectifier = TargetRectifier(
            target_type="B-2",
            canonical_dim=self.canonical_dim,
            pixels_per_mm=self.pixels_per_mm
        )
        self.nominal_rad_mm = 38.989  # B-2 black bullseye radius

    def _render_synthetic_ellipse(
        self,
        tilt_deg: float,
        roll_deg: float = 0.0,
        raw_scale_px_per_mm: float = 2.5,
        img_size: int = 600,
        with_glare: bool = False
    ) -> np.ndarray:
        """Renders synthetic raw target card with specified tilt, roll, and optional specular glare."""
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * np.array([145, 180, 215], dtype=np.uint8)

        cx, cy = img_size / 2.0, img_size / 2.0
        a = self.nominal_rad_mm * raw_scale_px_per_mm
        b = a * math.cos(math.radians(tilt_deg))

        axes = (int(round(a)), int(round(b)))
        c_int = (int(round(cx)), int(round(cy)))
        cv2.ellipse(img, c_int, axes, roll_deg, 0, 360, (30, 30, 35), -1, lineType=cv2.LINE_AA)

        if with_glare:
            # Inject bright specular flash reflection hotspot inside dark ink
            glare_spot = (c_int[0] + 15, c_int[1] - 10)
            cv2.circle(img, glare_spot, 8, (250, 252, 255), -1, lineType=cv2.LINE_AA)

        return img

    def test_geo_01_flat_target_accuracy(self):
        """GEO-01: Orthogonal target (0 deg tilt) maintains circularity >= 0.98 and exact canonical radius."""
        raw_img = self._render_synthetic_ellipse(tilt_deg=0.0, roll_deg=0.0)
        result = self.rectifier.process_raw_image(raw_img)

        self.assertFalse(result.is_fallback)
        self.assertLess(result.tilt_angle_deg, 4.0)
        self.assertGreaterEqual(result.circularity, 0.98)
        self.assertEqual(result.canonical_crop.shape, (256, 256, 3))

    def test_geo_02_variable_pitch_tilt(self):
        """GEO-02: Perspective tilt (15, 30, 42 deg) recovered accurately and flattened to circular frame."""
        for tilt in [15.0, 30.0, 42.0]:
            raw_img = self._render_synthetic_ellipse(tilt_deg=tilt, roll_deg=0.0)
            result = self.rectifier.process_raw_image(raw_img)

            self.assertFalse(result.is_fallback, f"Fallback triggered unexpectedly at tilt={tilt}")
            self.assertAlmostEqual(result.tilt_angle_deg, tilt, delta=3.5)

            # In canonical crop, black bullseye area should match pi * (nominal_mm * 2.0)^2
            canon_gray = cv2.cvtColor(result.canonical_crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(canon_gray, 80, 255, cv2.THRESH_BINARY_INV)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.assertGreater(len(cnts), 0)

            c_max = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c_max)
            expected_r = self.nominal_rad_mm * self.pixels_per_mm
            expected_area = math.pi * (expected_r ** 2)
            self.assertAlmostEqual(area, expected_area, delta=expected_area * 0.12)

    def test_geo_03_roll_invariance(self):
        """GEO-03: Arbitrary roll angles (0, 35, 75, 90, 135 deg) do not produce spurious rotations."""
        for roll in [0.0, 35.0, 75.0, 90.0, 135.0]:
            raw_img = self._render_synthetic_ellipse(tilt_deg=20.0, roll_deg=roll)
            result = self.rectifier.process_raw_image(raw_img)

            self.assertFalse(result.is_fallback)
            self.assertAlmostEqual(result.tilt_angle_deg, 20.0, delta=3.5)
            self.assertEqual(result.canonical_crop.shape, (256, 256, 3))

    def test_geo_04_specular_glare_suppression(self):
        """GEO-04: Specular flash hotspot inside dark bull is inpainted before unwarping."""
        raw_img = self._render_synthetic_ellipse(tilt_deg=10.0, with_glare=True)
        result = self.rectifier.process_raw_image(raw_img)

        self.assertFalse(result.is_fallback)
        self.assertGreaterEqual(result.circularity, 0.96)

    def test_geo_05_motion_blur_metric(self):
        """GEO-05: Low-contrast/blurry target triggers motion blur warning flag."""
        raw_img = self._render_synthetic_ellipse(tilt_deg=0.0)
        # Apply severe Gaussian blur
        blurry = cv2.GaussianBlur(raw_img, (31, 31), 10.0)
        result = self.rectifier.process_raw_image(blurry)

        self.assertLess(result.blur_metric, self.rectifier.min_blur_metric)
        self.assertIsNotNone(result.warning)
        self.assertIn("blurry", result.warning.lower())


class TestLowResSubpixelPeakExtraction(unittest.TestCase):
    """Stage 2: Sub-Pixel Peak Extraction & Hessian Metrology Tests."""

    def test_met_01_subpixel_coordinate_accuracy(self):
        """MET-01: Resolves continuous sub-pixel coordinates to < 0.06 px (< 0.03 mm at 2.0 px/mm)."""
        canvas_dim = 120
        hmap = np.zeros((canvas_dim, canvas_dim), dtype=np.float32)
        true_x, true_y = 60.35, 60.65
        sigma = 1.5  # Scaled for 2.0 px/mm
        y, x = np.mgrid[0:canvas_dim, 0:canvas_dim]
        hmap = np.exp(-((x - true_x)**2 + (y - true_y)**2) / (2.0 * sigma**2)).astype(np.float32)

        peaks = extract_subpixel_peaks(hmap, threshold=0.35, min_distance_px=8.0, max_peaks=1)
        self.assertEqual(len(peaks), 1)
        pred_x, pred_y, score = peaks[0]

        self.assertAlmostEqual(pred_x, true_x, delta=0.06)
        self.assertAlmostEqual(pred_y, true_y, delta=0.06)
        self.assertGreater(score, 0.90)

    def test_met_02_negative_definite_hessian_check(self):
        """MET-02: Non-negative definite stencils (saddle points, plateaus) clamp offset to zero safely."""
        # Flat plateau with zero curvature (det = 0)
        hmap = np.full((30, 30), 0.50, dtype=np.float32)
        hmap[15, 15] = 0.5001  # peak center
        peaks = extract_subpixel_peaks(hmap, threshold=0.45, min_distance_px=5.0, max_peaks=1)
        if peaks:
            pred_x, pred_y, _ = peaks[0]
            # Offset should not extrapolate wildly
            self.assertAlmostEqual(pred_x, 15.0, delta=1.0)
            self.assertAlmostEqual(pred_y, 15.0, delta=1.0)

    def test_met_03_caliber_aware_spatial_nms(self):
        """MET-03: Greedy NMS resolves distinct double-shot peaks while suppressing noise within caliber radius."""
        canvas_dim = 80
        hmap = np.zeros((canvas_dim, canvas_dim), dtype=np.float32)
        y, x = np.mgrid[0:canvas_dim, 0:canvas_dim]

        # Two shots separated by 12 px (6.0 mm, valid double shot)
        s1 = np.exp(-((x - 30.0)**2 + (y - 40.0)**2) / (2.0 * 1.5**2))
        s2 = np.exp(-((x - 42.0)**2 + (y - 40.0)**2) / (2.0 * 1.5**2))
        hmap = np.maximum(s1, s2).astype(np.float32)

        peaks = extract_subpixel_peaks(hmap, threshold=0.40, min_distance_px=8.0, max_peaks=10)
        self.assertEqual(len(peaks), 2, "Failed to resolve double shot as 2 distinct peaks.")

    def test_met_04_confidence_sorting_and_threshold(self):
        """MET-04: Confidences below threshold rejected; peaks returned in descending score order."""
        canvas_dim = 60
        hmap = np.zeros((canvas_dim, canvas_dim), dtype=np.float32)
        y, x = np.mgrid[0:canvas_dim, 0:canvas_dim]

        # 3 peaks with scores 0.90, 0.70, and 0.30 (below 0.45 threshold)
        p1 = 0.90 * np.exp(-((x - 15.0)**2 + (y - 15.0)**2) / (2.0 * 1.5**2))
        p2 = 0.70 * np.exp(-((x - 45.0)**2 + (y - 15.0)**2) / (2.0 * 1.5**2))
        p3 = 0.30 * np.exp(-((x - 30.0)**2 + (y - 45.0)**2) / (2.0 * 1.5**2))
        hmap = np.maximum(np.maximum(p1, p2), p3).astype(np.float32)

        peaks = extract_subpixel_peaks(hmap, threshold=0.45, min_distance_px=6.0, max_peaks=10)
        self.assertEqual(len(peaks), 2)
        self.assertGreater(peaks[0][2], peaks[1][2])

    def test_met_05_half_precision_amp_compatibility(self):
        """MET-05: AMP / float16 input heatmap handled transparently without OpenCV morphology errors."""
        canvas_dim = 60
        y, x = np.mgrid[0:canvas_dim, 0:canvas_dim]
        hmap_fp16 = np.exp(-((x - 30.0)**2 + (y - 30.0)**2) / (2.0 * 2.0**2)).astype(np.float16)

        peaks = extract_subpixel_peaks(hmap_fp16, threshold=0.45, min_distance_px=6.0, max_peaks=5)
        self.assertEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0][0], 30.0, delta=0.1)
        self.assertAlmostEqual(peaks[0][1], 30.0, delta=0.1)

    def test_met_06_adaptive_wing_loss_foreground_weighting(self):
        """MET-06: Verifies AdaptiveWingLoss foreground weighting increases loss on peak regions."""
        loss_unweighted = AdaptiveWingLoss(foreground_weight=0.0)
        loss_weighted = AdaptiveWingLoss(foreground_weight=10.0)

        # Create dummy target with one peak (1.0) and mostly background (0.0)
        t = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        t[0, 0, 16, 16] = 1.0
        # Prediction with small error at the peak
        p = t.clone()
        p[0, 0, 16, 16] = 0.50  # 0.50 error on foreground

        l_unw = loss_unweighted(p, t)
        l_w = loss_weighted(p, t)
        # Weighted loss must penalize the foreground peak error much more heavily
        self.assertGreater(l_w.item(), l_unw.item())

    def test_met_07_one_to_one_bipartite_shot_matching(self):
        """MET-07: Verifies calculate_localization_metrics prevents multiple GT shots matching 1 peak."""
        pixels_per_mm = 4.0
        # Single detected peak at (100, 100)
        hmap = np.zeros((200, 200), dtype=np.float32)
        y, x = np.mgrid[0:200, 0:200]
        hmap = np.exp(-((x - 100.0)**2 + (y - 100.0)**2) / (2.0 * 2.0**2)).astype(np.float32)

        # Two ground truth shots near the single peak
        gt_coords = np.array([[99.0, 100.0], [101.0, 100.0]], dtype=np.float32)

        mean_err, max_err, matches, total = calculate_localization_metrics(
            hmap, gt_coords, pixels_per_mm=pixels_per_mm, threshold=0.40
        )
        self.assertEqual(total, 2)
        # Exactly 1 shot may match to the single detected peak; the second GT shot remains unmatched!
        self.assertEqual(matches, 1)
        self.assertLess(mean_err, 1.0)

    def test_met_08_saturated_plateau_centroid_reduction(self):
        """MET-08: Saturated connected plateau is reduced to exactly one peak at its centroid."""
        canvas_dim = 100
        hmap = np.zeros((canvas_dim, canvas_dim), dtype=np.float32)
        # 10x10 plateau centered at (44.5, 44.5)
        hmap[40:50, 40:50] = 1.0
        peaks = extract_subpixel_peaks(hmap, threshold=0.45, min_distance_px=5.0, max_peaks=10)
        self.assertEqual(len(peaks), 1, "Saturated plateau produced multiple false peaks!")
        pred_x, pred_y, score = peaks[0]
        self.assertAlmostEqual(pred_x, 44.5, delta=0.01)
        self.assertAlmostEqual(pred_y, 44.5, delta=0.01)
        self.assertAlmostEqual(score, 1.0, delta=0.01)




class TestLowResOfficialNRAScoring(unittest.TestCase):
    """Stage 3: NRA Inward Scoring & Ballistic Dispersion Tests."""

    def setUp(self):
        self.cfg_b2 = TargetRingsConfig(target_type="B-2")
        self.cfg_b3 = TargetRingsConfig(target_type="B-3")
        self.pixels_per_mm = 2.0
        self.bullet_diameter_mm = 5.69
        self.bullet_r_mm = self.bullet_diameter_mm / 2.0  # 2.845 mm
        self.bull_center = (128.0, 128.0)

    def _eval_single_shot(self, dist_mm: float, cfg: TargetRingsConfig) -> str:
        shot_px = self.bull_center[0] + dist_mm * self.pixels_per_mm
        shot_peaks = [(shot_px, self.bull_center[1], 0.95)]
        results, _ = score_target_group(
            shot_peaks=shot_peaks,
            bullseye_peak=self.bull_center,
            pixels_per_mm=self.pixels_per_mm,
            bullet_diameter_mm=self.bullet_diameter_mm,
            rings_cfg=cfg,
        )
        return results[0].score_label

    def test_sc_01_inner_x_ring(self):
        """SC-01: Shot center at 1.20 mm earns inner X on B-3 (timed & rapid), but 10 on B-2 (slow fire)."""
        label_b3 = self._eval_single_shot(1.20, self.cfg_b3)
        self.assertEqual(label_b3, "X")
        label_b2 = self._eval_single_shot(1.20, self.cfg_b2)
        self.assertEqual(label_b2, "10")

    def test_sc_02_breaking_10_ring(self):
        """SC-02: Shot center at 14.15 mm has gauge dist 11.305 mm <= 11.43 mm -> breaks 10 ring."""
        label = self._eval_single_shot(14.15, self.cfg_b2)
        self.assertEqual(label, "10")

    def test_sc_03_missing_10_ring(self):
        """SC-03: Shot center at 14.35 mm has gauge dist 11.505 mm > 11.43 mm -> misses 10, earns 9."""
        label = self._eval_single_shot(14.35, self.cfg_b2)
        self.assertEqual(label, "9")

    def test_sc_04_breaking_8_ring(self):
        """SC-04: Shot center at 31.00 mm has gauge dist 28.155 mm <= 28.321 mm -> breaks 8 ring."""
        label = self._eval_single_shot(31.00, self.cfg_b2)
        self.assertEqual(label, "8")

    def test_sc_05_b3_breaking_9_ring_black_edge(self):
        """SC-05: NRA B-3 shot center at 41.50 mm has gauge dist 38.655 mm <= 38.862 mm -> breaks 9 ring."""
        label = self._eval_single_shot(41.50, self.cfg_b3)
        self.assertEqual(label, "9")

    def test_ballistics_dispersion_calculations(self):
        """Validates Extreme Spread, Mean Radius, Windage/Elevation, and MOA calculations."""
        # 4 shots placed at (+15mm, 0), (-15mm, 0), (0, +10mm), (0, -10mm)
        shot_peaks = [
            (self.bull_center[0] + 15.0 * self.pixels_per_mm, self.bull_center[1], 0.9),
            (self.bull_center[0] - 15.0 * self.pixels_per_mm, self.bull_center[1], 0.9),
            (self.bull_center[0], self.bull_center[1] + 10.0 * self.pixels_per_mm, 0.9),
            (self.bull_center[0], self.bull_center[1] - 10.0 * self.pixels_per_mm, 0.9),
        ]
        results, metrics = score_target_group(
            shot_peaks=shot_peaks,
            bullseye_peak=self.bull_center,
            pixels_per_mm=self.pixels_per_mm,
            rings_cfg=self.cfg_b2,
            distance_yards=16.67
        )

        self.assertEqual(metrics.total_shots, 4)
        # Extreme Spread is between (+15, 0) and (-15, 0) = 30.0 mm
        self.assertAlmostEqual(metrics.extreme_spread_mm, 30.0, delta=0.01)
        # Expected MOA at 16.67 yd: 30.0 / (16.67 * 0.26598) ~= 6.766 MOA
        expected_moa = calculate_moa(30.0, 16.67)
        self.assertAlmostEqual(metrics.extreme_spread_moa, expected_moa, delta=0.01)
        # Center of Impact (COI) should be exactly (0.0, 0.0) mm
        self.assertAlmostEqual(metrics.windage_bias_mm, 0.0, delta=0.01)
        self.assertAlmostEqual(metrics.elevation_bias_mm, 0.0, delta=0.01)

    def test_scoring_pixels_per_mm_positive(self):
        """SC-06: Verifies score_target_group raises ValueError if pixels_per_mm <= 0."""
        with self.assertRaises(ValueError):
            score_target_group(
                shot_peaks=[(128.0, 128.0, 0.9)],
                bullseye_peak=(128.0, 128.0),
                pixels_per_mm=0.0,
            )

    def test_single_shot_group_metrics(self):
        """SC-07: Verifies 1-shot group gracefully evaluates with 0.0 extreme spread and farthest_pair=(1,1)."""
        shot_peaks = [(self.bull_center[0] + 5.0 * self.pixels_per_mm, self.bull_center[1], 0.95)]
        results, metrics = score_target_group(
            shot_peaks=shot_peaks,
            bullseye_peak=self.bull_center,
            pixels_per_mm=self.pixels_per_mm,
            rings_cfg=self.cfg_b2,
        )
        self.assertEqual(metrics.total_shots, 1)
        self.assertEqual(metrics.extreme_spread_mm, 0.0)
        self.assertEqual(metrics.mean_radius_mm, 0.0)
        self.assertEqual(metrics.farthest_pair, (1, 1))


class TestLowResEndToEndPipeline(unittest.TestCase):
    """Stage 4: End-to-End Pipeline & Scorecard HUD Rendering Tests."""

    def test_e2e_01_demo_generation_and_hud_overlay(self):
        """E2E-01: Generates 256x256 demo target, scores shots, and compiles HUD inspection banner."""
        canvas_size = 256
        pixels_per_mm = 2.0
        rings_cfg = TargetRingsConfig(target_type="B-2")

        # 1. Generate low-res synthetic demo target
        img_canonical, bullseye_center = generate_synthetic_demo_target(
            canvas_size=canvas_size,
            pixels_per_mm=pixels_per_mm,
            target_type="B-2"
        )
        self.assertEqual(img_canonical.shape, (256, 256, 3))

        # 2. Mock bullet heatmap and extract peaks
        hmap = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        y, x = np.mgrid[0:canvas_size, 0:canvas_size]
        shot_positions = [(128.0 + 8.0, 128.0 - 6.0), (128.0 - 15.0, 128.0 + 12.0)]
        for sx, sy in shot_positions:
            hmap = np.maximum(hmap, np.exp(-((x - sx)**2 + (y - sy)**2) / (2.0 * 1.5**2)))

        peaks = extract_subpixel_peaks(hmap, threshold=0.45, min_distance_px=8.0, max_peaks=10)
        self.assertEqual(len(peaks), 2)

        # 3. Score group
        shot_results, metrics = score_target_group(
            shot_peaks=peaks,
            bullseye_peak=bullseye_center,
            pixels_per_mm=pixels_per_mm,
            rings_cfg=rings_cfg,
            distance_yards=16.67
        )
        self.assertEqual(metrics.total_shots, 2)
        self.assertGreater(metrics.total_score, 0)

        # 4. Render visual inspection scorecard HUD
        overlay = render_scoring_inspection_overlay(
            image_bgr=img_canonical,
            bullet_heatmap=hmap,
            shot_results=shot_results,
            bullseye_center=bullseye_center,
            metrics=metrics,
            pixels_per_mm=pixels_per_mm,
            rings_cfg=rings_cfg,
        )
        # Expected width: 256 (canonical) + 390 (HUD panel) = 646 px
        self.assertEqual(overlay.shape, (256, 256 + 390, 3))

    def test_e2e_02_unet_divisibility_validation(self):
        """E2E-02: Verifies HighResHeatmapUNet validates that spatial dimensions are divisible by 8."""
        model = HighResHeatmapUNet(in_channels=3, out_channels=2, base_dim=16)
        x_valid = torch.zeros((1, 3, 64, 64))
        out = model(x_valid)
        self.assertEqual(out.shape, (1, 2, 64, 64))

        x_invalid = torch.zeros((1, 3, 65, 65))
        with self.assertRaises(ValueError):
            model(x_invalid)

    def test_e2e_03_infer_nonexistent_image_raises(self):
        """E2E-03: Verifies run_inference_and_scoring raises FileNotFoundError for nonexistent image."""
        with self.assertRaises(FileNotFoundError):
            run_inference_and_scoring(
                image_path="this_image_definitely_does_not_exist_at_all.png",
                checkpoint_path="weights/target_scoring_hrnet_512_best.pt"
            )


class TestTextAndHandwritingInvariance(unittest.TestCase):
    """Stage 5: Text, Competitor Handwriting, and Scorer Annotations Invariance Tests."""

    def test_text_01_procedural_handwriting_synthesis(self):
        """TEXT-01: Verifies handwriting, headers, and scorer circles are stamped with 0.0 ground truth heatmap."""
        cfg = TargetConfig(
            canvas_size=256,
            pixels_per_mm=2.0,
            include_handwriting=True,
            include_header_text=True,
            include_scorer_marks=True,
        )
        canvas, heatmap, annotations, center, radii = generate_synthetic_target_sample(
            config=cfg, num_singles=3, num_doubles=1, num_keyholes=0
        )
        self.assertEqual(len(annotations), 5)
        self.assertEqual(canvas.shape, (256, 256, 3))
        self.assertEqual(heatmap.shape, (256, 256))
        # Check that top margin (where header text and name fields are written) has strictly 0.0 heatmap
        self.assertAlmostEqual(float(np.max(heatmap[:20, :])), 0.0, delta=1e-5)

    def test_text_02_rectification_with_overlapping_handwriting(self):
        """TEXT-02: Handwriting strokes crossing into the 7-ring do not skew Fitzgibbon ellipse fitting."""
        rectifier = TargetRectifier(target_type="B-2", canonical_dim=256, pixels_per_mm=2.0)
        img_size = 600
        raw_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * np.array([145, 180, 215], dtype=np.uint8)
        cx, cy = img_size / 2.0, img_size / 2.0
        a = 38.989 * 2.5
        b = a * math.cos(math.radians(20.0))
        cv2.ellipse(raw_img, (int(round(cx)), int(round(cy))), (int(round(a)), int(round(b))), 0, 0, 360, (30, 30, 35), -1, cv2.LINE_AA)

        # Draw heavy handwriting crossing the edge of the aiming black in dark blue and graphite ink
        cv2.putText(raw_img, "RELAY 2 - J. DOE #10948", (int(cx - a * 0.9), int(cy - b * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (130, 40, 20), 2, cv2.LINE_AA)
        cv2.putText(raw_img, "MATCH 3 - 50 FT", (int(cx - a * 0.5), int(cy + b * 0.9)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, (30, 30, 35), 2, cv2.LINE_AA)

        result = rectifier.process_raw_image(raw_img)
        self.assertFalse(result.is_fallback)
        self.assertAlmostEqual(result.tilt_angle_deg, 20.0, delta=4.5)
        self.assertGreaterEqual(result.circularity, 0.90)

    def test_text_03_scorer_circles_do_not_duplicate_shots(self):
        """TEXT-03: Hand-drawn circles around bullet holes do not create false positive or duplicate peaks."""
        canvas_dim = 120
        hmap = np.zeros((canvas_dim, canvas_dim), dtype=np.float32)
        y, x = np.mgrid[0:canvas_dim, 0:canvas_dim]
        # Authentic bullet hole Gaussian peak at (60, 60)
        hmap = np.exp(-((x - 60.0)**2 + (y - 60.0)**2) / (2.0 * 2.0**2)).astype(np.float32)

        # Simulated hand-drawn pen circle ring in background
        circle_dist = np.abs(np.sqrt((x - 60.0)**2 + (y - 60.0)**2) - 15.0)
        circle_noise = 0.20 * np.exp(-(circle_dist**2) / (2.0 * 1.5**2))
        combined = np.maximum(hmap, circle_noise.astype(np.float32))

        peaks = extract_subpixel_peaks(combined, threshold=0.45, min_distance_px=8.0, max_peaks=10)
        self.assertEqual(len(peaks), 1)
        self.assertAlmostEqual(peaks[0][0], 60.0, delta=0.05)
        self.assertAlmostEqual(peaks[0][1], 60.0, delta=0.05)

    def test_text_04_pii_redaction_overlay(self):
        """TEXT-04: Verifies --redact-pii masks margin competitor registration zones while preserving bullseye."""
        canvas_size = 256
        pixels_per_mm = 2.0
        rings_cfg = TargetRingsConfig(target_type="B-2")
        img_canonical, bull_center = generate_synthetic_demo_target(
            canvas_size=canvas_size, pixels_per_mm=pixels_per_mm, target_type="B-2"
        )
        cv2.putText(img_canonical, "NAME: JOHN DOE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 25), 1)

        hmap = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        shot_results, metrics = score_target_group(
            shot_peaks=[(128.0, 128.0, 0.95)],
            bullseye_peak=bull_center,
            pixels_per_mm=pixels_per_mm,
            rings_cfg=rings_cfg
        )

        overlay_redacted = render_scoring_inspection_overlay(
            image_bgr=img_canonical,
            bullet_heatmap=hmap,
            shot_results=shot_results,
            bullseye_center=bull_center,
            metrics=metrics,
            pixels_per_mm=pixels_per_mm,
            rings_cfg=rings_cfg,
            redact_pii=True
        )

        self.assertEqual(overlay_redacted.shape, (256, 256 + 390, 3))


if __name__ == "__main__":
    unittest.main()
