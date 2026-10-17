"""
Generates realistic sample calibration targets for training, active learning, and pipeline validation.
Produces:
1. Canonical 1024x1024 rectified target images
2. Realistic perspective-tilted smartphone range photographs
3. Ground-truth JSON annotations adhering to TargetAnnotation schema
4. Dataset manifest.json index
"""

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from target_scorer.config import TargetConfig, TargetRingsConfig
from target_scorer.data.schema import AnnotatedShot, TargetAnnotation
from target_scorer.metrology.scoring import score_target_group
from target_scorer.synthetic.generator import generate_synthetic_target_sample


def project_to_perspective_photo(
    canonical_bgr: np.ndarray,
    out_w: int = 1600,
    out_h: int = 1200,
    pitch_deg: float = 18.0,
    yaw_deg: float = -12.0,
    roll_deg: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Renders a realistic perspective photograph of a target card hanging on a range backer.
    Returns the synthetic photograph and the 3x3 homography matrix (raw -> canonical).
    """
    can_h, can_w = canonical_bgr.shape[:2]

    # Create larger wooden / acoustic range stall background
    bg = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    # Range stall color (dark gray/slate with subtle wood grain)
    bg[:, :] = [45, 48, 52]
    grain = np.random.normal(0, 4, (out_h, out_w, 3)).astype(np.float32)
    bg = np.clip(bg.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    # Calculate 3D rotation matrix
    focal = 1400.0  # Simulated smartphone lens focal length
    rad_pitch = math.radians(pitch_deg)
    rad_yaw = math.radians(yaw_deg)
    rad_roll = math.radians(roll_deg)

    # Rotation matrices around X, Y, Z
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rad_pitch), -math.sin(rad_pitch)],
        [0, math.sin(rad_pitch), math.cos(rad_pitch)]
    ], dtype=np.float64)

    Ry = np.array([
        [math.cos(rad_yaw), 0, math.sin(rad_yaw)],
        [0, 1, 0],
        [-math.sin(rad_yaw), 0, math.cos(rad_yaw)]
    ], dtype=np.float64)

    Rz = np.array([
        [math.cos(rad_roll), -math.sin(rad_roll), 0],
        [math.sin(rad_roll), math.cos(rad_roll), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    R = Rz @ Ry @ Rx

    # Camera intrinsic matrix
    K = np.array([
        [focal, 0, out_w / 2.0],
        [0, focal, out_h / 2.0],
        [0, 0, 1]
    ], dtype=np.float64)

    # 4 corners in canonical card space centered at origin
    half_w = can_w / 2.0
    half_h = can_h / 2.0
    dist_z = 1350.0  # Camera distance in mm/pixels

    corners_canonical = np.array([
        [0.0, 0.0],
        [can_w, 0.0],
        [can_w, can_h],
        [0.0, can_h]
    ], dtype=np.float32)

    corners_3d = np.array([
        [-half_w, -half_h, 0.0],
        [half_w, -half_h, 0.0],
        [half_w, half_h, 0.0],
        [-half_w, half_h, 0.0]
    ], dtype=np.float64)

    # Rotate in 3D and project
    corners_proj: List[Tuple[float, float]] = []
    for pt in corners_3d:
        p_rot = R @ pt
        p_cam = p_rot + np.array([0, 0, dist_z])
        u = K[0, 0] * (p_cam[0] / p_cam[2]) + K[0, 2]
        v = K[1, 1] * (p_cam[1] / p_cam[2]) + K[1, 2]
        corners_proj.append((u, v))

    dst_pts = np.array(corners_proj, dtype=np.float32)

    # Homography from canonical card to raw phone frame
    H_can_to_raw = cv2.getPerspectiveTransform(corners_canonical, dst_pts)
    H_raw_to_can = np.linalg.inv(H_can_to_raw)

    # Warp card onto background
    warped_card = cv2.warpPerspective(canonical_bgr, H_can_to_raw, (out_w, out_h))
    mask = cv2.warpPerspective(np.full((can_h, can_w), 255, dtype=np.uint8), H_can_to_raw, (out_w, out_h))

    # Composite card into stall background
    inv_mask = cv2.bitwise_not(mask)
    bg_masked = cv2.bitwise_and(bg, bg, mask=inv_mask)
    card_masked = cv2.bitwise_and(warped_card, warped_card, mask=mask)
    composite = cv2.add(bg_masked, card_masked)

    # Apply directional range lighting gradient
    x_lin = np.linspace(-0.25, 0.25, out_w, dtype=np.float32)
    y_lin = np.linspace(-0.20, 0.20, out_h, dtype=np.float32)
    xx, yy = np.meshgrid(x_lin, y_lin)
    lighting_grad = 1.0 + (xx * math.cos(rad_yaw) + yy * math.sin(rad_pitch))
    composite = np.clip(composite.astype(np.float32) * lighting_grad[:, :, np.newaxis], 0, 255).astype(np.uint8)

    # Sensor noise
    sensor_noise = np.random.normal(0, 3.0, (out_h, out_w, 3)).astype(np.float32)
    final_photo = np.clip(composite.astype(np.float32) + sensor_noise, 0, 255).astype(np.uint8)

    return final_photo, H_raw_to_can


def generate_calibration_dataset(output_dir: str = "datasets/calibration_targets"):
    """Generates complete sample calibration dataset with 6 diverse NRA target cards."""
    out_path = Path(output_dir)
    raw_dir = out_path / "raw"
    canon_dir = out_path / "canonical_1024"
    ann_dir = out_path / "annotations"

    raw_dir.mkdir(parents=True, exist_ok=True)
    canon_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    targets_to_generate = [
        {"id": "calib_b2_001", "type": "B-2", "pitch": 18.0, "yaw": -14.0, "roll": 4.0, "shots": (6, 2, 0)},
        {"id": "calib_b2_002", "type": "B-2", "pitch": -22.0, "yaw": 10.0, "roll": -6.0, "shots": (8, 1, 0)},
        {"id": "calib_b2_003", "type": "B-2", "pitch": 12.0, "yaw": 15.0, "roll": 2.0, "shots": (7, 1, 1)},
        {"id": "calib_b3_001", "type": "B-3", "pitch": 15.0, "yaw": -10.0, "roll": -5.0, "shots": (6, 2, 0)},
        {"id": "calib_b3_002", "type": "B-3", "pitch": -18.0, "yaw": -12.0, "roll": 3.0, "shots": (8, 1, 0)},
        {"id": "calib_b3_003", "type": "B-3", "pitch": 20.0, "yaw": 8.0, "roll": -4.0, "shots": (7, 1, 1)},
    ]

    manifest = {
        "version": "1.0",
        "description": "Sample calibration targets for real-world target scoring training & active learning.",
        "targets": []
    }

    print(f"[+] Generating {len(targets_to_generate)} calibration targets in '{output_dir}'...")

    for spec in targets_to_generate:
        tid = spec["id"]
        ttype = spec["type"]
        n_s, n_d, n_k = spec["shots"]

        # 1. Generate canonical 1024x1024 target with realistic handwriting, grain, and cardboard
        cfg = TargetConfig(
            canvas_size=1024,
            pixels_per_mm=4.0,
            target_type=ttype,
            include_handwriting=True,
            include_scorer_marks=True,
            include_header_text=True,
        )

        canonical_img, _, shot_annotations, bull_center, ring_radii = generate_synthetic_target_sample(
            config=cfg,
            num_singles=n_s,
            num_doubles=n_d,
            num_keyholes=n_k
        )

        # Score the shots
        peaks = [(s.center_x, s.center_y, 1.0) for s in shot_annotations]
        shot_results, group_metrics = score_target_group(
            shot_peaks=peaks,
            bullseye_peak=bull_center,
            pixels_per_mm=4.0,
            bullet_diameter_mm=5.69,
            rings_cfg=TargetRingsConfig(target_type=ttype),
            distance_yards=16.67
        )

        # Build AnnotatedShot objects
        annotated_shots: List[AnnotatedShot] = []
        for i, (orig_ann, s_res) in enumerate(zip(shot_annotations, shot_results)):
            annotated_shots.append(AnnotatedShot(
                id=i + 1,
                center_x=float(orig_ann.center_x),
                center_y=float(orig_ann.center_y),
                score=s_res.score_value,
                is_x=s_res.is_inner_x,
                caliber_mm=5.69,
                shot_type=orig_ann.shot_type,
                confidence=1.0
            ))

        # 2. Save Canonical 1024x1024 Image
        canon_filename = f"{tid}_canonical_1024.png"
        canon_file_path = canon_dir / canon_filename
        cv2.imwrite(str(canon_file_path), canonical_img)

        # 3. Save JSON Annotation
        ann_filename = f"{tid}.json"
        ann_file_path = ann_dir / ann_filename
        target_ann = TargetAnnotation(
            target_id=tid,
            image_filename=canon_filename,
            target_type=ttype,
            bullseye_center=bull_center,
            shots=annotated_shots,
            canvas_size=1024,
            pixels_per_mm=4.0,
            metadata={
                "total_score": group_metrics.total_score,
                "x_count": group_metrics.x_count,
                "extreme_spread_mm": round(group_metrics.extreme_spread_mm, 2),
                "mean_radius_mm": round(group_metrics.mean_radius_mm, 2),
                "pitch_deg": spec["pitch"],
                "yaw_deg": spec["yaw"],
                "roll_deg": spec["roll"],
            }
        )
        target_ann.to_json(ann_file_path)

        # 4. Generate perspective-skewed range photograph
        photo_bgr, _ = project_to_perspective_photo(
            canonical_img,
            out_w=1600,
            out_h=1200,
            pitch_deg=spec["pitch"],
            yaw_deg=spec["yaw"],
            roll_deg=spec["roll"],
        )
        raw_photo_filename = f"{tid}_raw_photo.jpg"
        raw_photo_path = raw_dir / raw_photo_filename
        cv2.imwrite(str(raw_photo_path), photo_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        manifest["targets"].append({
            "target_id": tid,
            "target_type": ttype,
            "canonical_image": str(canon_filename),
            "raw_photo": str(raw_photo_filename),
            "annotation": str(ann_filename),
            "num_shots": len(annotated_shots),
            "total_score": group_metrics.total_score,
            "x_count": group_metrics.x_count,
        })
        print(f"  [✓] Generated {tid} ({ttype}) -> Score: {group_metrics.total_score} ({len(annotated_shots)} shots)")

    # Save manifest.json
    manifest_path = out_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[✓] Calibration dataset generation complete! Manifest: {manifest_path}")


if __name__ == "__main__":
    generate_calibration_dataset()
