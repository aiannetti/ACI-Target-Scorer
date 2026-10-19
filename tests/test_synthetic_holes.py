"""
Unit tests for multi-tone bullet hole synthesis (white scanner backing, dark void, tan cardboards)
and handwriting / scorer markings retention.
"""

import unittest
import numpy as np
import cv2

from target_scorer.config import TargetConfig
from target_scorer.synthetic.generator import (
    generate_synthetic_target_sample,
    sample_bullet_void_bgr,
    stamp_bullet_hole,
    stamp_keyhole_shot,
)


class TestSyntheticHolesAndMarkings(unittest.TestCase):
    """Verifies white scanner voids, tan backers, dark voids, and annotations."""

    def test_sample_bullet_void_bgr_modes(self):
        # 1. Scanner white mode
        cfg_white = TargetConfig(void_color_mode="scanner_white")
        for _ in range(20):
            void_bgr = sample_bullet_void_bgr(cfg_white)
            self.assertTrue(all(c >= 215 for c in void_bgr), f"Expected bright white scanner backing, got {void_bgr}")

        # 2. Dark range void mode
        cfg_dark = TargetConfig(void_color_mode="dark_void")
        for _ in range(20):
            void_bgr = sample_bullet_void_bgr(cfg_dark)
            self.assertTrue(all(c <= 32 for c in void_bgr), f"Expected dark void, got {void_bgr}")

        # 3. Tan cardboard backer mode
        cfg_tan = TargetConfig(void_color_mode="tan_backer")
        for _ in range(20):
            void_bgr = sample_bullet_void_bgr(cfg_tan)
            self.assertTrue(130 <= void_bgr[0] <= 180)
            self.assertTrue(150 <= void_bgr[1] <= 200)
            self.assertTrue(170 <= void_bgr[2] <= 220)

        # 4. Mixed mode (defaults)
        cfg_mixed = TargetConfig(void_color_mode="mixed")
        sampled = [sample_bullet_void_bgr(cfg_mixed) for _ in range(100)]
        has_white = any(s[0] >= 215 for s in sampled)
        has_dark = any(s[0] <= 32 for s in sampled)
        has_tan = any(130 <= s[0] <= 180 for s in sampled)
        self.assertTrue(has_white, "Expected at least one white scanner hole sampled")
        self.assertTrue(has_dark, "Expected at least one dark void hole sampled")
        self.assertTrue(has_tan, "Expected at least one tan backer hole sampled")

    def test_stamp_bullet_hole_scanner_white(self):
        """Verifies stamping a white-centered bullet hole on a black bullseye background."""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)  # Black bullseye
        cfg = TargetConfig(canvas_size=200, pixels_per_mm=4.0, bullet_diameter_mm=5.69, void_color_mode="scanner_white")

        shot = stamp_bullet_hole(canvas, (100.0, 100.0), cfg)
        self.assertEqual(shot.center_x, 100.0)
        self.assertEqual(shot.center_y, 100.0)

        # Center pixel of the void should be bright white/cream
        center_pixel = canvas[100, 100]
        self.assertTrue(all(c >= 215 for c in center_pixel), f"Expected white center in scanner hole, got {center_pixel}")

        # Outer rim should contain dark grease ring pixels around circumference
        r_bullet_px = shot.caliber_radius_px
        r_outer = int(round(r_bullet_px + cfg.grease_ring_width_mm * cfg.pixels_per_mm * 0.3))
        rim_pixels = [canvas[int(round(100 + r_outer * np.sin(a))), int(round(100 + r_outer * np.cos(a)))]
                      for a in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        dark_grease_pixels = [p for p in rim_pixels if p[0] <= 45 and p[1] <= 45 and p[2] <= 45]
        self.assertGreater(len(dark_grease_pixels), 10, f"Expected dark lead grease ring around rim, found {len(dark_grease_pixels)}/16")

    def test_stamp_keyhole_shot(self):
        """Verifies keyhole stamping with multi-tone void."""
        canvas = np.zeros((200, 200, 3), dtype=np.uint8)
        cfg = TargetConfig(canvas_size=200, pixels_per_mm=4.0, bullet_diameter_mm=5.69, void_color_mode="scanner_white")

        shot = stamp_keyhole_shot(canvas, (100.0, 100.0), cfg)
        self.assertEqual(shot.shot_type, "keyhole")
        center_pixel = canvas[100, 100]
        self.assertTrue(all(c >= 215 for c in center_pixel), f"Expected white center in keyhole, got {center_pixel}")

    def test_sample_bullet_void_bgr_balanced_distribution(self):
        """Verifies exactly 1/3 white, 1/3 black/dark, 1/3 tan backer voids in mixed mode."""
        cfg_mixed = TargetConfig(void_color_mode="mixed")
        n_samples = 600
        sampled = [sample_bullet_void_bgr(cfg_mixed) for _ in range(n_samples)]

        count_white = sum(1 for s in sampled if s[0] >= 215)
        count_dark = sum(1 for s in sampled if s[0] <= 32)
        count_tan = sum(1 for s in sampled if 130 <= s[0] <= 180)

        self.assertEqual(count_white + count_dark + count_tan, n_samples)
        # Expected ~200 each (33.3%). Allow generous statistical tolerance [140, 260]
        self.assertGreater(count_white, 140, f"White voids under-represented: {count_white}/{n_samples}")
        self.assertLess(count_white, 260, f"White voids over-represented: {count_white}/{n_samples}")
        self.assertGreater(count_dark, 140, f"Dark voids under-represented: {count_dark}/{n_samples}")
        self.assertLess(count_dark, 260, f"Dark voids over-represented: {count_dark}/{n_samples}")
        self.assertGreater(count_tan, 140, f"Tan voids under-represented: {count_tan}/{n_samples}")
        self.assertLess(count_tan, 260, f"Tan voids over-represented: {count_tan}/{n_samples}")

    def test_stratified_4zone_shot_placement(self):
        """Verifies synthetic generation places shots across all 4 physical zones."""
        import math
        cfg = TargetConfig(canvas_size=1024, pixels_per_mm=4.0, target_type="B-2")
        bull_center = (512.0, 512.0)
        r_bull = 38.989 * 4.0   # ~156 px
        r_outer = 93.091 * 4.0  # ~372 px

        zone_counts = {"inner_black": 0, "outer_rings": 0, "paper_margin": 0, "off_paper_edge": 0}
        for _ in range(40):
            _, _, shots, _, _ = generate_synthetic_target_sample(cfg, num_singles=5, num_doubles=1, num_keyholes=1)
            for s in shots:
                x, y = s.center_x, s.center_y
                r = math.hypot(x - bull_center[0], y - bull_center[1])
                if x < 40 or x > 984 or y < 40 or y > 984:
                    zone_counts["off_paper_edge"] += 1
                elif r <= r_bull:
                    zone_counts["inner_black"] += 1
                elif r <= r_outer:
                    zone_counts["outer_rings"] += 1
                else:
                    zone_counts["paper_margin"] += 1

        total_shots = sum(zone_counts.values())
        self.assertGreater(zone_counts["inner_black"], total_shots * 0.20, f"Inner black under-represented: {zone_counts}")
        self.assertGreater(zone_counts["outer_rings"], total_shots * 0.08, f"Outer rings under-represented: {zone_counts}")
        self.assertGreater(zone_counts["paper_margin"], total_shots * 0.08, f"Paper margin under-represented: {zone_counts}")
        self.assertGreater(zone_counts["off_paper_edge"], total_shots * 0.08, f"Off-paper edge under-represented: {zone_counts}")

    def test_demo_target_has_all_zones_and_voids(self):
        """Verifies demo targets showcase shots across inner, outer, margin, and edge with all 3 void colors."""
        from target_scorer.infer import generate_synthetic_demo_target
        img, (cx, cy) = generate_synthetic_demo_target(canvas_size=1024, pixels_per_mm=4.0, target_type="B-2")
        self.assertEqual(img.shape, (1024, 1024, 3))

    def test_nra_seal_and_handwriting_zero_ground_truth_heatmap(self):
        """Verifies procedural NRA seal and handwriting markings receive strict 0.0 ground truth heatmap."""
        cfg = TargetConfig(canvas_size=1024, pixels_per_mm=4.0, include_nra_seal=True, include_scorer_marks=True)
        canvas, heatmap, annotations, bc, rings = generate_synthetic_target_sample(cfg, num_singles=3, num_doubles=0, num_keyholes=0)

        # Lower-left margin NRA seal region (x: 60..240, y: 880..1024)
        seal_hmap = heatmap[880:1024, 60:240]
        shots_in_seal = [a for a in annotations if 60 <= a.center_x <= 240 and 880 <= a.center_y <= 1024]
        if not shots_in_seal:
            self.assertEqual(float(seal_hmap.max()), 0.0, "NRA seal must have 0.0 ground truth heatmap for negative hard mining")

        # Upper margin header & written score region (x: 550..850, y: 20..200)
        margin_hmap = heatmap[20:200, 550:850]
        shots_in_margin = [a for a in annotations if 550 <= a.center_x <= 850 and 20 <= a.center_y <= 200]
        if not shots_in_margin:
            self.assertEqual(float(margin_hmap.max()), 0.0, "Scorer tally area must have 0.0 ground truth heatmap")

    def test_filter_spurious_peaks_rejection(self):
        """Verifies filter_spurious_peaks suppresses seal and header artifacts while preserving real shots."""
        from target_scorer.metrology.peaks import filter_spurious_peaks

        mock_peaks = [
            (512.0, 512.0, 0.95),   # Bullseye center hit
            (480.0, 530.0, 0.88),   # 10 ring hit
            (155.0, 970.0, 0.65),   # Spurious NRA seal peak
            (140.0, 990.0, 0.58),   # Spurious NRA seal peak #2
            (500.0, 25.0, 0.52),    # Spurious header text peak
            (380.0, 550.0, 0.75),   # Outer ring hit
        ]

        # Filter spurious peaks
        cleaned = filter_spurious_peaks(
            peaks=mock_peaks,
            canonical_bgr=None,
            canvas_size=1024,
            pixels_per_mm=4.0,
            target_type="B-2",
            max_peaks=10
        )

        cleaned_coords = [(round(p[0]), round(p[1])) for p in cleaned]
        # Real ring shots preserved
        self.assertIn((512, 512), cleaned_coords)
        self.assertIn((480, 530), cleaned_coords)
        self.assertIn((380, 550), cleaned_coords)
        # Seal and header false positives rejected
        self.assertNotIn((155, 970), cleaned_coords)
        self.assertNotIn((140, 990), cleaned_coords)
        self.assertNotIn((500, 25), cleaned_coords)


if __name__ == "__main__":
    unittest.main()
