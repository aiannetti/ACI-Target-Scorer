"""
Unit tests for Progressive Resizing training curriculum and multi-resolution execution.
Verifies fully convolutional dynamic spatial resizing across 256, 512, 768, and 1024 px,
optimizer buffer compatibility, and curriculum orchestration.
"""

import os
import shutil
import tempfile
import unittest
import torch
import numpy as np

from target_scorer.config import TargetConfig, TrainConfig
from target_scorer.models.unet import HighResHeatmapUNet
from target_scorer.synthetic.generator import (
    generate_cardboard_texture,
    draw_standard_bullseye,
    stamp_bullet_hole,
)
from target_scorer.geometry.rectifier import TargetRectifier
from target_scorer.train import (
    DEFAULT_PROGRESSIVE_STAGES,
    parse_progressive_stages,
    run_progressive_training,
)


class TestProgressiveTraining(unittest.TestCase):
    """Test suite for progressive resizing curriculum and resolution transitions."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_unet_dynamic_spatial_resolutions(self):
        """Verifies HighResHeatmapUNet processes 256, 512, 768, and 1024 without weight changes."""
        model = HighResHeatmapUNet(in_channels=3, out_channels=2, base_dim=16)
        model.eval()

        resolutions = [256, 512, 768, 1024]
        for res in resolutions:
            dummy_input = torch.zeros(1, 3, res, res, dtype=torch.float32)
            with torch.no_grad():
                out = model(dummy_input)
            self.assertEqual(out.shape, (1, 2, res, res), f"Output shape mismatch for input resolution {res}")

    def test_parse_progressive_stages_default(self):
        """Verifies default stages schedule parsing."""
        stages = parse_progressive_stages(None)
        self.assertEqual(len(stages), 4)
        # Check resolutions
        res_list = [s[0] for s in stages]
        self.assertEqual(res_list, [256, 512, 768, 1024])
        # Check uniform 256-step scale progression
        scales = [s[1] for s in stages]
        self.assertEqual(scales, [1.0, 2.0, 3.0, 4.0])

    def test_parse_progressive_stages_custom(self):
        """Verifies custom stage string parsing across formats."""
        # 3-token format: canvas:epochs:lr
        s3 = parse_progressive_stages("256:2:1e-3,512:1:5e-4")
        self.assertEqual(len(s3), 2)
        self.assertEqual(s3[0][0], 256)
        self.assertEqual(s3[0][4], 2)
        self.assertEqual(s3[0][5], 1e-3)
        self.assertEqual(s3[1][0], 512)
        self.assertEqual(s3[1][4], 1)

        # 6-token format: canvas:scale:sigma:batch:epochs:lr
        s6 = parse_progressive_stages("256:1.0:1.6:8:3:2e-4")
        self.assertEqual(len(s6), 1)
        self.assertEqual(s6[0][0], 256)
        self.assertEqual(s6[0][1], 1.0)
        self.assertEqual(s6[0][2], 1.6)
        self.assertEqual(s6[0][3], 8)
        self.assertEqual(s6[0][4], 3)
        self.assertEqual(s6[0][5], 2e-4)

        # Invalid token
        with self.assertRaises(ValueError):
            parse_progressive_stages("invalid:tokens:a:b:c:d:e:f:g:h")

    def test_synthetic_generation_at_all_resolutions(self):
        """Verifies synthetic canvas, bullseye, and stamp functions across all 4 resolutions."""
        for res, scale in [(256, 1.0), (512, 2.0), (768, 3.0), (1024, 4.0)]:
            cfg = TargetConfig(canvas_size=res, pixels_per_mm=scale, target_type="B-2")
            canvas = generate_cardboard_texture(cfg)
            self.assertEqual(canvas.shape, (res, res, 3))

            canvas, center, ring_radii = draw_standard_bullseye(canvas, cfg)
            self.assertEqual(canvas.shape, (res, res, 3))
            self.assertAlmostEqual(center[0], res / 2.0, delta=1.0)
            self.assertAlmostEqual(center[1], res / 2.0, delta=1.0)

            # Test bullet hole stamping
            shot_ann = stamp_bullet_hole(canvas, (float(center[0]), float(center[1])), cfg)
            self.assertEqual(canvas.shape, (res, res, 3))
            self.assertIsNotNone(shot_ann)

    def test_rectifier_at_all_resolutions(self):
        """Verifies TargetRectifier supports arbitrary canonical dimensions."""
        for res, scale in [(256, 1.0), (512, 2.0), (768, 3.0), (1024, 4.0)]:
            rectifier = TargetRectifier(target_type="B-2", canonical_dim=res, pixels_per_mm=scale)
            self.assertEqual(rectifier.canonical_dim, res)
            self.assertEqual(rectifier.pixels_per_mm, scale)

    def test_run_progressive_training_micro(self):
        """Verifies run_progressive_training completes an end-to-end multi-stage micro run on CPU."""
        micro_stages = [
            # (canvas_size, scale, sigma, batch_size, epochs, lr, train_samples, val_samples)
            (256, 1.0, 1.6, 2, 1, 1e-3, 2, 2),
            (512, 2.0, 2.2, 2, 1, 1e-3, 2, 2),
        ]
        final_ckpt = os.path.join(self.temp_dir, "weights", "target_scoring_hrnet_1024_best.pt")
        ckpt_dir = os.path.join(self.temp_dir, "weights")

        result_path = run_progressive_training(
            stages=micro_stages,
            checkpoint_path=final_ckpt,
            num_workers=0,
            use_amp=False,
            checkpoint_dir=ckpt_dir,
        )

        # Checkpoint for stage 1 (256) should exist
        stage1_ckpt = os.path.join(ckpt_dir, "target_scoring_hrnet_256_best.pt")
        self.assertTrue(os.path.exists(stage1_ckpt), f"Stage 1 checkpoint not saved at {stage1_ckpt}")

        # Checkpoint for stage 2 (512) should exist
        stage2_ckpt = os.path.join(ckpt_dir, "target_scoring_hrnet_512_best.pt")
        self.assertTrue(os.path.exists(stage2_ckpt), f"Stage 2 checkpoint not saved at {stage2_ckpt}")

        # Final checkpoint should exist
        self.assertTrue(os.path.exists(final_ckpt), f"Final checkpoint not saved at {final_ckpt}")

        # Verify loaded checkpoint payload
        data = torch.load(final_ckpt, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", data)
        self.assertIn("config", data)
        self.assertEqual(data["config"].canvas_size, 512)


if __name__ == "__main__":
    unittest.main()
