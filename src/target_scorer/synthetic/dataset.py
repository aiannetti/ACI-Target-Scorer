"""
PyTorch procedural dataset for on-the-fly target generation and training augmentations.
"""

import random
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from target_scorer.config import ShotAnnotation, TargetConfig, TrainConfig
from target_scorer.synthetic.generator import (
    generate_ground_truth_heatmap,
    generate_synthetic_target_sample,
)


class SyntheticTargetDataset(Dataset):
    """
    Infinite procedural PyTorch Dataset alternating between official NRA B-2 and B-3 targets.
    Integrates affine geometric jitter, scale jitter, tilt perturbation, and photometric noise.
    """
    def __init__(self, size: int, config: TrainConfig, is_train: bool = True):
        self.size = size
        self.config = config
        self.is_train = is_train

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        self.mean_tensor = torch.from_numpy(self.mean)
        self.std_tensor = torch.from_numpy(self.std)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        target_type = random.choice(["B-2", "B-3"]) if self.is_train else ("B-2" if idx % 2 == 0 else "B-3")

        cardboard_tone = (
            int(np.clip(random.gauss(145, 12), 115, 175)),
            int(np.clip(random.gauss(180, 12), 150, 210)),
            int(np.clip(random.gauss(215, 12), 185, 245))
        ) if self.is_train else (145, 180, 215)

        gen_cfg = TargetConfig(
            canvas_size=self.config.canvas_size,
            pixels_per_mm=self.config.pixels_per_mm,
            bullet_diameter_mm=self.config.bullet_diameter_mm,
            heatmap_sigma_px=self.config.heatmap_sigma_px,
            cardboard_base_bgr=cardboard_tone,
            target_type=target_type
        )

        num_singles = random.randint(3, 6) if self.is_train else 4
        num_doubles = random.randint(1, 2) if self.is_train else 1
        num_keyholes = random.randint(0, 1) if self.is_train else 1

        img_bgr, _, shot_annotations, bull_center, _ = generate_synthetic_target_sample(
            config=gen_cfg,
            num_singles=num_singles,
            num_doubles=num_doubles,
            num_keyholes=num_keyholes,
        )

        shot_coords_list = [[s.center_x, s.center_y] for s in shot_annotations]
        bull_x, bull_y = bull_center

        if self.is_train:
            c_x = self.config.canvas_size / 2.0
            c_y = self.config.canvas_size / 2.0
            scale_jitter = random.uniform(0.94, 1.06)
            angle_jitter = random.uniform(-4.0, 4.0)
            max_shift = self.config.canvas_size * 0.045
            dx = random.uniform(-max_shift, max_shift)
            dy = random.uniform(-max_shift, max_shift)

            M = cv2.getRotationMatrix2D((c_x, c_y), angle_jitter, scale_jitter)
            M[0, 2] += dx
            M[1, 2] += dy

            img_bgr = cv2.warpAffine(
                img_bgr, M, (self.config.canvas_size, self.config.canvas_size),
                borderMode=cv2.BORDER_REFLECT
            )

            bull_x = float(M[0, 0] * bull_center[0] + M[0, 1] * bull_center[1] + M[0, 2])
            bull_y = float(M[1, 0] * bull_center[0] + M[1, 1] * bull_center[1] + M[1, 2])

            transformed_shots = []
            margin = max(6.0, 2.0 * self.config.pixels_per_mm)
            for sx, sy in shot_coords_list:
                tx = float(M[0, 0] * sx + M[0, 1] * sy + M[0, 2])
                ty = float(M[1, 0] * sx + M[1, 1] * sy + M[1, 2])
                if margin <= tx < self.config.canvas_size - margin and margin <= ty < self.config.canvas_size - margin:
                    transformed_shots.append([tx, ty])
            shot_coords_list = transformed_shots

            alpha = random.uniform(0.82, 1.18)
            beta = random.uniform(-14.0, 14.0)
            noise = np.random.normal(0.0, 3.0, img_bgr.shape).astype(np.float32)
            img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta + noise, 0, 255).astype(np.uint8)

        # Render exact unattenuated sub-pixel Gaussians directly at final continuous coordinates
        final_shot_annotations = [
            ShotAnnotation(center_x=sx, center_y=sy, shot_type="shot", caliber_radius_px=0.0)
            for sx, sy in shot_coords_list
        ]
        bullet_hmap = generate_ground_truth_heatmap(final_shot_annotations, gen_cfg)

        bullseye_annotation = [
            ShotAnnotation(center_x=bull_x, center_y=bull_y, shot_type="bullseye", caliber_radius_px=0.0)
        ]
        bullseye_sigma = max(3.5, gen_cfg.heatmap_sigma_px * 1.5)
        bullseye_hmap = generate_ground_truth_heatmap(
            bullseye_annotation, gen_cfg, sigma_override=bullseye_sigma
        )

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
        img_tensor = (img_tensor - self.mean_tensor) / self.std_tensor

        hmap_combined = np.stack([bullet_hmap, bullseye_hmap], axis=0).astype(np.float32)
        hmap_tensor = torch.from_numpy(hmap_combined)
        shot_coords = np.array(shot_coords_list, dtype=np.float32) if shot_coords_list else np.zeros((0, 2), dtype=np.float32)

        return {
            "image": img_tensor,
            "heatmap": hmap_tensor,
            "shot_coords": torch.from_numpy(shot_coords),
            "bullseye_coord": torch.tensor([bull_x, bull_y], dtype=torch.float32),
            "target_type": target_type,
        }


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Batches images and heatmaps while preserving variable-length shot annotations."""
    images = torch.stack([item["image"] for item in batch], dim=0)
    heatmaps = torch.stack([item["heatmap"] for item in batch], dim=0)
    bullseyes = torch.stack([item["bullseye_coord"] for item in batch], dim=0)
    shot_coords = [item["shot_coords"] for item in batch]

    return {
        "image": images,
        "heatmap": heatmaps,
        "bullseye_coord": bullseyes,
        "shot_coords": shot_coords,
    }
