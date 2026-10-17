"""
Unit tests for real-world target dataset ingestion, TargetAnnotation schema,
HybridTargetDataset blending, and assisted auto-labeling tooling.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from target_scorer.config import TrainConfig
from target_scorer.data.autolabel import autolabel_target_image
from target_scorer.data.hybrid_dataset import HybridTargetDataset
from target_scorer.data.real_dataset import RealTargetDataset
from target_scorer.data.schema import AnnotatedShot, TargetAnnotation
from target_scorer.synthetic.dataset import custom_collate_fn


class TestTargetSchema(unittest.TestCase):
    """Tests for AnnotatedShot and TargetAnnotation dataclasses and serialization."""

    def test_annotated_shot_roundtrip(self):
        shot = AnnotatedShot(
            id=1,
            center_x=256.5,
            center_y=260.2,
            score=10,
            is_x=True,
            caliber_mm=5.69,
            shot_type="single",
            confidence=0.98,
        )
        d = shot.to_dict()
        self.assertEqual(d["center_x"], 256.5)
        self.assertEqual(d["score"], 10)
        self.assertTrue(d["is_x"])

        restored = AnnotatedShot.from_dict(d)
        self.assertEqual(restored.center_x, 256.5)
        self.assertEqual(restored.center_y, 260.2)
        self.assertEqual(restored.score, 10)
        self.assertTrue(restored.is_x)
        self.assertEqual(restored.confidence, 0.98)

    def test_target_annotation_json_io(self):
        shots = [
            AnnotatedShot(id=1, center_x=512.0, center_y=512.0, score=10, is_x=True),
            AnnotatedShot(id=2, center_x=525.0, center_y=518.0, score=9, is_x=False),
        ]
        ann = TargetAnnotation(
            target_id="target_001",
            image_filename="target_001.png",
            target_type="B-2",
            bullseye_center=(512.0, 512.0),
            shots=shots,
            canvas_size=1024,
            pixels_per_mm=4.0,
            metadata={"notes": "test annotation"},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "test_ann.json")
            ann.to_json(json_path)
            self.assertTrue(os.path.exists(json_path))

            restored = TargetAnnotation.from_json(json_path)
            self.assertEqual(restored.target_id, "target_001")
            self.assertEqual(restored.image_filename, "target_001.png")
            self.assertEqual(restored.target_type, "B-2")
            self.assertEqual(len(restored.shots), 2)
            self.assertEqual(restored.total_score, 19)
            self.assertEqual(restored.x_count, 1)

            coords = restored.to_numpy_coords()
            self.assertIsInstance(coords, np.ndarray)
            self.assertEqual(coords.shape, (2, 2))
            self.assertAlmostEqual(coords[0, 0], 512.0)
            self.assertAlmostEqual(coords[1, 0], 525.0)

    def test_target_annotation_empty_shots(self):
        ann = TargetAnnotation(
            target_id="empty",
            image_filename="empty.png",
            target_type="B-3",
            bullseye_center=(256.0, 256.0),
            canvas_size=512,
            pixels_per_mm=2.0,
        )
        coords = ann.to_numpy_coords()
        self.assertEqual(coords.shape, (0, 2))
        self.assertEqual(ann.total_score, 0)
        self.assertEqual(ann.x_count, 0)

    def test_target_annotation_from_dict_legacy_xy_keys(self):
        d = {
            "target_id": "legacy_01",
            "image_filename": "legacy.png",
            "target_type": "B-2",
            "bullseye_center": [512.0, 512.0],
            "shots": [
                {"x": 510.5, "y": 511.2, "score": 10, "is_x": True},
                {"x": 490.0, "y": 520.0, "score": 9},
            ],
            "canvas_size": 1024,
            "pixels_per_mm": 4.0,
        }
        ann = TargetAnnotation.from_dict(d)
        self.assertEqual(len(ann.shots), 2)
        self.assertEqual(ann.shots[0].center_x, 510.5)
        self.assertEqual(ann.shots[0].center_y, 511.2)
        self.assertEqual(ann.shots[0].score, 10)
        self.assertTrue(ann.shots[0].is_x)
        self.assertEqual(ann.shots[1].center_x, 490.0)

    def test_target_annotation_from_json_raw_string(self):
        raw_json = json.dumps({
            "target_id": "raw_str_01",
            "image_filename": "test.png",
            "target_type": "B-2",
            "bullseye_center": [512.0, 512.0],
            "shots": [{"center_x": 512.0, "center_y": 512.0, "score": 10}],
            "canvas_size": 1024,
            "pixels_per_mm": 4.0,
        })
        ann = TargetAnnotation.from_json(raw_json)
        self.assertEqual(ann.target_id, "raw_str_01")
        self.assertEqual(len(ann.shots), 1)
        self.assertEqual(ann.shots[0].center_x, 512.0)


class TestRealTargetDataset(unittest.TestCase):
    """Tests for RealTargetDataset with calibration targets."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.calib_dir = cls.project_root / "datasets" / "calibration_targets"
        cls.config = TrainConfig(
            canvas_size=256,
            pixels_per_mm=2.0,
            heatmap_sigma_px=2.0,
            batch_size=2,
            num_workers=0,
        )

    def test_load_calibration_dataset(self):
        if not self.calib_dir.exists():
            self.skipTest(f"Calibration directory '{self.calib_dir}' not found.")

        dataset = RealTargetDataset(
            data_dir=self.calib_dir,
            config=self.config,
            is_train=False,
            virtual_size=6,
        )
        self.assertEqual(len(dataset), 6)

        # Inspect first sample
        sample = dataset[0]
        self.assertIn("image", sample)
        self.assertIn("heatmap", sample)
        self.assertIn("shot_coords", sample)
        self.assertIn("bullseye_coord", sample)
        self.assertIn("target_type", sample)

        # Check tensor dimensions
        self.assertEqual(sample["image"].shape, (3, 256, 256))
        self.assertEqual(sample["heatmap"].shape, (2, 256, 256))
        self.assertEqual(sample["bullseye_coord"].shape, (2,))
        self.assertIsInstance(sample["shot_coords"], torch.Tensor)

    def test_virtual_size_expansion(self):
        if not self.calib_dir.exists():
            self.skipTest(f"Calibration directory '{self.calib_dir}' not found.")

        # Test repeating small dataset over a larger virtual epoch
        dataset = RealTargetDataset(
            data_dir=self.calib_dir,
            config=self.config,
            is_train=True,
            virtual_size=18,
        )
        self.assertEqual(len(dataset), 18)
        # Sample indexing beyond original count (18 > 6)
        sample = dataset[17]
        self.assertEqual(sample["image"].shape, (3, 256, 256))

    def test_dataloader_batching(self):
        if not self.calib_dir.exists():
            self.skipTest(f"Calibration directory '{self.calib_dir}' not found.")

        dataset = RealTargetDataset(
            data_dir=self.calib_dir,
            config=self.config,
            is_train=False,
            virtual_size=4,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn)

        batches = list(loader)
        self.assertEqual(len(batches), 2)
        first_batch = batches[0]
        self.assertEqual(first_batch["image"].shape, (2, 3, 256, 256))
        self.assertEqual(first_batch["heatmap"].shape, (2, 2, 256, 256))
        self.assertEqual(first_batch["bullseye_coord"].shape, (2, 2))
        self.assertEqual(len(first_batch["shot_coords"]), 2)

    def test_train_augmentation_coordinate_transform(self):
        """Verifies that affine transformations in training mode (is_train=True) do not corrupt coordinates."""
        if not self.calib_dir.exists():
            self.skipTest(f"Calibration directory '{self.calib_dir}' not found.")

        dataset = RealTargetDataset(
            data_dir=self.calib_dir,
            config=self.config,
            is_train=True,
            virtual_size=10,
        )
        for i in range(10):
            sample = dataset[i]
            bull = sample["bullseye_coord"].numpy()
            # Canvas size is 256. Bullseye should be near center within reasonable margin
            self.assertTrue(50.0 <= bull[0] <= 206.0, f"Bullseye X out of bounds: {bull[0]}")
            self.assertTrue(50.0 <= bull[1] <= 206.0, f"Bullseye Y out of bounds: {bull[1]}")

            # Heatmap channel 1 peak should align with bullseye coordinate
            hmap_bull = sample["heatmap"][1].numpy()
            y_peak, x_peak = np.unravel_index(np.argmax(hmap_bull), hmap_bull.shape)
            dist_to_hmap = np.hypot(x_peak - bull[0], y_peak - bull[1])
            self.assertLess(dist_to_hmap, 3.0, f"Heatmap peak ({x_peak}, {y_peak}) diverged from bullseye ({bull[0]}, {bull[1]})")


class TestHybridTargetDataset(unittest.TestCase):
    """Tests for HybridTargetDataset blending synthetic and real datasets."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.calib_dir = cls.project_root / "datasets" / "calibration_targets"
        cls.config = TrainConfig(
            canvas_size=256,
            pixels_per_mm=2.0,
            heatmap_sigma_px=2.0,
            batch_size=2,
            num_workers=0,
        )

    def test_hybrid_pure_real(self):
        """Option B: real_ratio = 1.0 (pure real training)."""
        if not self.calib_dir.exists():
            self.skipTest(f"Calibration directory '{self.calib_dir}' not found.")

        dataset = HybridTargetDataset(
            real_data_dir=self.calib_dir,
            config=self.config,
            size=4,
            real_ratio=1.0,
            is_train=False,
        )
        self.assertEqual(len(dataset), 4)
        sample = dataset[0]
        self.assertEqual(sample["image"].shape, (3, 256, 256))

    def test_hybrid_blended(self):
        """Option A: real_ratio = 0.5 (blended training)."""
        if not self.calib_dir.exists():
            self.skipTest(f"Calibration directory '{self.calib_dir}' not found.")

        dataset = HybridTargetDataset(
            real_data_dir=self.calib_dir,
            config=self.config,
            size=4,
            real_ratio=0.5,
            is_train=False,
        )
        self.assertEqual(len(dataset), 4)
        for i in range(4):
            sample = dataset[i]
            self.assertEqual(sample["image"].shape, (3, 256, 256))

    def test_hybrid_pure_synthetic(self):
        dataset = HybridTargetDataset(
            real_data_dir=None,
            config=self.config,
            size=4,
            real_ratio=0.0,
            is_train=False,
        )
        self.assertEqual(len(dataset), 4)
        self.assertEqual(dataset.real_ratio, 0.0)
        sample = dataset[0]
        self.assertEqual(sample["image"].shape, (3, 256, 256))

    def test_hybrid_fallback_on_invalid_dir(self):
        dataset = HybridTargetDataset(
            real_data_dir="/non/existent/path/for/target/scorer/data",
            config=self.config,
            size=4,
            real_ratio=0.5,
            is_train=False,
        )
        self.assertEqual(dataset.real_ratio, 0.0)
        self.assertIsNone(dataset.real_dataset)
        sample = dataset[0]
        self.assertEqual(sample["image"].shape, (3, 256, 256))

    def test_hybrid_validation_determinism(self):
        """Verifies that is_train=False samples deterministically across multiple evaluations."""
        if not self.calib_dir.exists():
            self.skipTest(f"Calibration directory '{self.calib_dir}' not found.")

        dataset = HybridTargetDataset(
            real_data_dir=self.calib_dir,
            config=self.config,
            size=10,
            real_ratio=0.5,
            is_train=False,
        )
        # Verify that accessing indices repeatedly returns identical target types and bullseye coords
        for idx in range(10):
            sample_1 = dataset[idx]
            sample_2 = dataset[idx]
            self.assertEqual(sample_1["target_type"], sample_2["target_type"])
            self.assertTrue(torch.allclose(sample_1["bullseye_coord"], sample_2["bullseye_coord"]))


class TestAutoLabeling(unittest.TestCase):
    """Tests for assisted auto-labeling pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parent.parent
        cls.calib_dir = cls.project_root / "datasets" / "calibration_targets"

    def test_autolabel_single_target(self):
        if not self.calib_dir.exists():
            self.skipTest(f"Calibration directory '{self.calib_dir}' not found.")

        raw_photo = self.calib_dir / "raw" / "calib_b2_001_raw_photo.jpg"
        if not raw_photo.exists():
            self.skipTest(f"Calibration raw photo '{raw_photo}' not found.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            ann, json_path, review_path = autolabel_target_image(
                image_source=raw_photo,
                output_dir=tmp_dir,
                checkpoint_path=None,  # Use fresh model if weights not present
                target_type="B-2",
                canvas_size=256,
                pixels_per_mm=2.0,
                peak_threshold=0.30,
                max_shots=10,
            )

            self.assertTrue(json_path.exists())
            self.assertTrue(review_path.exists())
            self.assertIsInstance(ann, TargetAnnotation)
            self.assertEqual(ann.target_type, "B-2")
            self.assertGreater(ann.canvas_size, 0)

            # Check that the saved json can be deserialized
            restored = TargetAnnotation.from_json(json_path)
            self.assertEqual(restored.target_type, "B-2")
            self.assertEqual(len(restored.shots), len(ann.shots))

    def test_autolabel_with_redact_pii(self):
        if not self.calib_dir.exists():
            self.skipTest(f"Calibration directory '{self.calib_dir}' not found.")

        raw_photo = self.calib_dir / "raw" / "calib_b2_001_raw_photo.jpg"
        if not raw_photo.exists():
            self.skipTest(f"Calibration raw photo '{raw_photo}' not found.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            ann, json_path, review_path = autolabel_target_image(
                image_source=raw_photo,
                output_dir=tmp_dir,
                checkpoint_path=None,
                target_type="B-2",
                canvas_size=256,
                pixels_per_mm=2.0,
                peak_threshold=0.30,
                max_shots=10,
                redact_pii=True,
            )

            self.assertTrue(json_path.exists())
            self.assertTrue(review_path.exists())
            self.assertTrue(ann.metadata.get("redact_pii"))

            restored = TargetAnnotation.from_json(json_path)
            self.assertTrue(restored.metadata.get("redact_pii"))


if __name__ == "__main__":
    unittest.main()
