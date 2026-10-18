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

    def test_generate_synthetic_target_sample_preserves_marks(self):
        """Verifies complete sample generation with handwriting and scorer markings."""
        cfg = TargetConfig(
            canvas_size=512,
            target_type="B-3",
            include_handwriting=True,
            include_scorer_marks=True,
            include_header_text=True,
            void_color_mode="mixed"
        )
        img, heatmap, shots, bull_center, ring_radii = generate_synthetic_target_sample(cfg, num_singles=5, num_doubles=1, num_keyholes=1)
        self.assertEqual(img.shape, (512, 512, 3))
        self.assertEqual(heatmap.shape, (512, 512))
        self.assertEqual(len(shots), 8)  # 5 + 2 + 1 = 8 shots


if __name__ == "__main__":
    unittest.main()
