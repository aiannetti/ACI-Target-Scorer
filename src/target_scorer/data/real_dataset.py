"""
PyTorch dataset for training and validating on real-world target images and JSON annotations.
"""

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from target_scorer.config import ShotAnnotation, TargetConfig, TrainConfig
from target_scorer.data.schema import TargetAnnotation
from target_scorer.geometry.canonical_loader import canonicalize_image
from target_scorer.synthetic.generator import generate_ground_truth_heatmap


class RealTargetDataset(Dataset):
    """
    PyTorch Dataset ingesting real-world target photographs and ground-truth JSON annotations.
    Generates exact 2-channel continuous sub-pixel Gaussian regression heatmaps and applies
    photometric and geometric domain-invariance augmentations.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        config: TrainConfig,
        is_train: bool = True,
        manifest_file: Optional[str] = None,
        virtual_size: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.is_train = is_train

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        self.mean_tensor = torch.from_numpy(self.mean)
        self.std_tensor = torch.from_numpy(self.std)

        # Locate image and annotation directories
        self.images_dir = self._find_dir(["canonical_1024", "images", "canonical", ""])
        self.annotations_dir = self._find_dir(["annotations", "labels", ""])

        # Load samples
        self.samples: List[Dict[str, Path]] = []
        self._load_dataset_index(manifest_file)

        if len(self.samples) == 0:
            raise FileNotFoundError(f"No valid target image/annotation pairs found in '{self.data_dir}'.")

        # In low-data regime, virtual_size repeats the real dataset over virtual epochs
        self.virtual_size = virtual_size if virtual_size is not None else len(self.samples)

    def _find_dir(self, candidates: List[str]) -> Path:
        for candidate in candidates:
            if not candidate:
                return self.data_dir
            p = self.data_dir / candidate
            if p.exists() and p.is_dir():
                return p
        return self.data_dir

    def _load_dataset_index(self, manifest_file: Optional[str]):
        """Indexes matching pairs of target images and annotation JSON files."""
        # 1. Check for explicit manifest.json
        manifest_path = self.data_dir / manifest_file if manifest_file else self.data_dir / "manifest.json"
        if manifest_path.exists() and manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            targets = data.get("targets", [])
            for item in targets:
                img_name = item.get("canonical_image") or item.get("image") or item.get("image_filename")
                ann_name = item.get("annotation") or item.get("annotation_filename")
                if img_name and ann_name:
                    img_p = self.images_dir / img_name
                    ann_p = self.annotations_dir / ann_name
                    if img_p.exists() and ann_p.exists() and not img_p.name.endswith("_review.png"):
                        self.samples.append({"image": img_p, "annotation": ann_p})

        # 2. If no manifest or empty, scan annotations directory and data directory
        if not self.samples:
            search_dirs = [self.annotations_dir]
            if self.data_dir != self.annotations_dir:
                search_dirs.append(self.data_dir)

            seen_anns = set()
            for s_dir in search_dirs:
                for ann_p in sorted(s_dir.rglob("*.json")):
                    if ann_p in seen_anns or ann_p.name == "manifest.json":
                        continue
                    seen_anns.add(ann_p)
                    try:
                        ann = TargetAnnotation.from_json(ann_p)
                        rel_sub = ""
                        try:
                            rel_sub = ann_p.relative_to(s_dir).parent
                        except ValueError:
                            pass

                        # Check possible image locations across flat, structured, and nested subdirectories
                        # (Note: *_review.png cards with visual overlays and HUD graphics are strictly excluded)
                        possible_img_paths = [
                            ann_p.parent / ann.image_filename,
                            ann_p.parent / f"{ann.target_id}_canonical.png",
                            ann_p.parent / f"{ann.target_id}.png",
                            self.images_dir / rel_sub / ann.image_filename,
                            self.images_dir / rel_sub / f"{ann.target_id}_canonical.png",
                            self.images_dir / ann.image_filename,
                            self.images_dir / f"{ann.target_id}_canonical.png",
                            self.data_dir / "canonical_1024" / rel_sub / ann.image_filename,
                            self.data_dir / "images" / rel_sub / ann.image_filename,
                            self.data_dir / "canonical_1024" / ann.image_filename,
                            self.data_dir / "images" / ann.image_filename,
                            self.annotations_dir.parent / "canonical_1024" / ann.image_filename,
                            self.annotations_dir.parent / "images" / ann.image_filename,
                            ann_p.parent.parent / "canonical_1024" / ann.image_filename,
                            ann_p.parent.parent / "canonical_1024" / f"{ann.target_id}_canonical.png",
                            ann_p.parent.parent / "images" / ann.image_filename,
                            ann_p.parent.parent / ann.image_filename,
                        ]
                        for img_p in possible_img_paths:
                            if img_p.exists() and img_p.is_file() and not img_p.name.endswith("_review.png"):
                                self.samples.append({"image": img_p, "annotation": ann_p})
                                break
                    except Exception:
                        continue

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx % len(self.samples)]
        img_path = sample["image"]
        ann_path = sample["annotation"]

        # 1. Load canonical image using ImageCanonicalizer (handles HEIC, TIFF, PNG, etc.)
        img_bgr = canonicalize_image(img_path, target_format="BGR")
        ann = TargetAnnotation.from_json(ann_path)

        # Ensure canvas matches expected config dimensions
        if img_bgr.shape[0] != self.config.canvas_size or img_bgr.shape[1] != self.config.canvas_size:
            scale_factor = self.config.canvas_size / float(img_bgr.shape[0])
            img_bgr = cv2.resize(
                img_bgr,
                (self.config.canvas_size, self.config.canvas_size),
                interpolation=cv2.INTER_AREA
            )
            bull_x = ann.bullseye_center[0] * scale_factor
            bull_y = ann.bullseye_center[1] * scale_factor
            shot_coords = [[s.center_x * scale_factor, s.center_y * scale_factor] for s in ann.shots]
        else:
            bull_x, bull_y = ann.bullseye_center
            shot_coords = [[s.center_x, s.center_y] for s in ann.shots]

        # 2. Apply training augmentations (rotations, shadows, lighting, JPEG compression)
        if self.is_train:
            c_x = self.config.canvas_size / 2.0
            c_y = self.config.canvas_size / 2.0
            angle_jitter = random.uniform(-180.0, 180.0)  # Targets are rotationally symmetric
            scale_jitter = random.uniform(0.95, 1.05)
            max_shift = self.config.canvas_size * 0.03
            dx = random.uniform(-max_shift, max_shift)
            dy = random.uniform(-max_shift, max_shift)

            M = cv2.getRotationMatrix2D((c_x, c_y), angle_jitter, scale_jitter)
            M[0, 2] += dx
            M[1, 2] += dy

            img_bgr = cv2.warpAffine(
                img_bgr, M, (self.config.canvas_size, self.config.canvas_size),
                borderMode=cv2.BORDER_REFLECT
            )

            # Transform coordinates
            new_bull_x = float(M[0, 0] * bull_x + M[0, 1] * bull_y + M[0, 2])
            new_bull_y = float(M[1, 0] * bull_x + M[1, 1] * bull_y + M[1, 2])
            bull_x, bull_y = new_bull_x, new_bull_y

            transformed_shots = []
            margin = max(4.0, 1.5 * self.config.pixels_per_mm)
            for sx, sy in shot_coords:
                tx = float(M[0, 0] * sx + M[0, 1] * sy + M[0, 2])
                ty = float(M[1, 0] * sx + M[1, 1] * sy + M[1, 2])
                if margin <= tx < self.config.canvas_size - margin and margin <= ty < self.config.canvas_size - margin:
                    transformed_shots.append([tx, ty])
            shot_coords = transformed_shots

            # Photometric lighting & shadow gradient
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-12.0, 12.0)
            noise = np.random.normal(0.0, 2.5, img_bgr.shape).astype(np.float32)
            img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta + noise, 0, 255).astype(np.uint8)

            # Directional stall shadow gradient
            if random.random() < 0.40:
                shadow_ang = random.uniform(0, 2 * math.pi)
                x_g = np.linspace(-0.15, 0.15, self.config.canvas_size, dtype=np.float32)
                y_g = np.linspace(-0.15, 0.15, self.config.canvas_size, dtype=np.float32)
                xx, yy = np.meshgrid(x_g, y_g)
                grad = 1.0 + (xx * math.cos(shadow_ang) + yy * math.sin(shadow_ang))
                img_bgr = np.clip(img_bgr.astype(np.float32) * grad[:, :, np.newaxis], 0, 255).astype(np.uint8)

            # JPEG DCT quantization jitter
            if random.random() < 0.60:
                quality = random.randint(35, 90)
                success, enc_buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                if success:
                    decoded = cv2.imdecode(enc_buf, cv2.IMREAD_COLOR)
                    if decoded is not None:
                        img_bgr = decoded

        # 3. Render continuous 2D sub-pixel Gaussian regression heatmaps
        gen_cfg = TargetConfig(
            canvas_size=self.config.canvas_size,
            pixels_per_mm=self.config.pixels_per_mm,
            heatmap_sigma_px=self.config.heatmap_sigma_px,
            target_type=ann.target_type,
        )

        final_shot_annotations = [
            ShotAnnotation(center_x=sx, center_y=sy, shot_type="shot", caliber_radius_px=0.0)
            for sx, sy in shot_coords
        ]
        bullet_hmap = generate_ground_truth_heatmap(final_shot_annotations, gen_cfg)

        bullseye_annotation = [
            ShotAnnotation(center_x=bull_x, center_y=bull_y, shot_type="bullseye", caliber_radius_px=0.0)
        ]
        bullseye_sigma = max(3.5, gen_cfg.heatmap_sigma_px * 1.5)
        bullseye_hmap = generate_ground_truth_heatmap(
            bullseye_annotation, gen_cfg, sigma_override=bullseye_sigma
        )

        # 4. Standard PyTorch tensor normalization
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
        img_tensor = (img_tensor - self.mean_tensor) / self.std_tensor

        hmap_combined = np.stack([bullet_hmap, bullseye_hmap], axis=0).astype(np.float32)
        hmap_tensor = torch.from_numpy(hmap_combined)
        coords_tensor = torch.tensor(shot_coords, dtype=torch.float32) if shot_coords else torch.zeros((0, 2), dtype=torch.float32)

        return {
            "image": img_tensor,
            "heatmap": hmap_tensor,
            "shot_coords": coords_tensor,
            "bullseye_coord": torch.tensor([bull_x, bull_y], dtype=torch.float32),
            "target_type": ann.target_type,
        }
