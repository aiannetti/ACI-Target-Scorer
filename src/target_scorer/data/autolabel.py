"""
Assisted auto-labeling and active learning tooling.
Predicts shot coordinates and bullseye center on raw or canonical targets using pre-trained weights,
generating standard TargetAnnotation JSON files and visual verification review cards.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from target_scorer.config import TargetRingsConfig
from target_scorer.data.schema import AnnotatedShot, TargetAnnotation
from target_scorer.geometry.canonical_loader import canonicalize_image
from target_scorer.geometry.rectifier import TargetRectifier
from target_scorer.metrology.peaks import extract_subpixel_peaks, filter_spurious_peaks
from target_scorer.metrology.scoring import score_target_group
from target_scorer.models.unet import HighResHeatmapUNet


def autolabel_target_image(
    image_source: Union[str, Path, np.ndarray],
    output_dir: Union[str, Path],
    checkpoint_path: str = "weights/target_scoring_hrnet_1024_best.pt",
    target_type: str = "B-2",
    canvas_size: int = 1024,
    pixels_per_mm: float = 4.0,
    peak_threshold: float = 0.25,
    max_shots: int = 10,
    bullet_diameter_mm: float = 5.69,
    device_str: Optional[str] = None,
    target_id: Optional[str] = None,
    redact_pii: bool = False,
) -> Tuple[TargetAnnotation, Path, Path]:
    """
    Runs model-assisted auto-labeling on an input target image.
    Outputs:
    1. Canonical 1024x1024 image
    2. Ground-truth candidate JSON annotation
    3. Visual verification review card with annotated crosshairs
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_str)

    # Resolve checkpoint
    model = HighResHeatmapUNet(in_channels=3, out_channels=2, base_dim=32).to(device)
    resolved_ckpt = checkpoint_path
    if resolved_ckpt is None:
        default_candidate = os.path.join(os.path.dirname(__file__), "..", "..", "..", "weights", "target_scoring_hrnet_1024_best.pt")
        if os.path.exists(default_candidate):
            resolved_ckpt = default_candidate

    if resolved_ckpt is not None:
        if not os.path.exists(resolved_ckpt):
            alt = os.path.join(os.path.dirname(__file__), "..", "..", "..", "weights", os.path.basename(resolved_ckpt))
            if os.path.exists(alt):
                resolved_ckpt = alt
            else:
                raise FileNotFoundError(f"Model checkpoint '{checkpoint_path}' not found.")

        try:
            ckpt = torch.load(resolved_ckpt, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(resolved_ckpt, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)

    model.eval()

    # 1. Ingest & Rectify image
    rectifier = TargetRectifier(target_type=target_type, canonical_dim=canvas_size, pixels_per_mm=pixels_per_mm)
    rect_result = rectifier.rectify(image_source)
    canonical_bgr = rect_result.canonical_crop

    # Derive target ID
    if target_id is None:
        if isinstance(image_source, (str, Path)) and str(image_source) != "-":
            target_id = Path(image_source).stem
        else:
            import uuid
            target_id = f"autolabel_{uuid.uuid4().hex[:8]}"

    # 2. Run HighResHeatmapUNet inference

    # Preprocess image tensor
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    norm_tensor = ((tensor - torch.from_numpy(mean)) / torch.from_numpy(std)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_hmap = model(norm_tensor).squeeze(0).cpu().numpy()

    bullet_hmap = pred_hmap[0]
    bull_hmap = pred_hmap[1]

    # Extract candidate bullet coordinates (using generous candidate pool)
    min_dist_px = max(6.0, 2.5 * pixels_per_mm)
    candidate_peaks = extract_subpixel_peaks(
        bullet_hmap,
        threshold=peak_threshold,
        min_distance_px=min_dist_px,
        max_peaks=max(30, max_shots * 3) if max_shots is not None else None
    )

    # Filter spurious printed target artifacts (official NRA seal, header text)
    detected_peaks = filter_spurious_peaks(
        peaks=candidate_peaks,
        canonical_bgr=canonical_bgr,
        canvas_size=canvas_size,
        pixels_per_mm=pixels_per_mm,
        target_type=target_type,
        max_peaks=max_shots
    )

    # Extract bullseye center (TargetRectifier centers the bullseye at canvas_size / 2)
    c_mid = canvas_size / 2.0
    search_radius = max(20.0, 8.0 * pixels_per_mm)
    yy, xx = np.ogrid[:canvas_size, :canvas_size]
    central_mask = ((xx - c_mid) ** 2 + (yy - c_mid) ** 2) <= (search_radius ** 2)
    bull_hmap_central = bull_hmap.copy()
    bull_hmap_central[~central_mask] = 0.0

    bull_peaks = extract_subpixel_peaks(bull_hmap_central, threshold=0.15, min_distance_px=10.0, max_peaks=1)
    if bull_peaks:
        bull_center = (float(bull_peaks[0][0]), float(bull_peaks[0][1]))
    else:
        bull_center = (c_mid, c_mid)

    # Optional Competitor Registration PII Redaction
    rings_cfg = TargetRingsConfig(target_type=target_type)
    header_h = 0
    if redact_pii:
        margin_mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        rings_table = rings_cfg.get_rings_table()
        outer_margin_rad = int(round(rings_table[-1][1] * pixels_per_mm + 12.0 * (pixels_per_mm / 4.0)))
        bc_int_x, bc_int_y = int(round(bull_center[0])), int(round(bull_center[1]))
        cv2.circle(margin_mask, (bc_int_x, bc_int_y), outer_margin_rad, 255, -1)
        header_h = max(28, int(round(bc_int_y - outer_margin_rad + 8)))
        if header_h > 0:
            margin_mask[:header_h, :] = 0

        blurred_crop = cv2.GaussianBlur(canonical_bgr, (45, 45), 25.0)
        pii_zone = (margin_mask == 0)[:, :, np.newaxis]
        canonical_bgr = np.where(pii_zone, blurred_crop, canonical_bgr)

    # Save canonical image (redacted if redact_pii=True)
    canonical_filename = f"{target_id}_canonical.png"
    canonical_path = out_dir / canonical_filename
    cv2.imwrite(str(canonical_path), canonical_bgr)

    # Score detected shots
    peaks_tuples = [(p[0], p[1], p[2]) for p in detected_peaks]
    shot_results, group_metrics = score_target_group(
        shot_peaks=peaks_tuples,
        bullseye_peak=bull_center,
        pixels_per_mm=pixels_per_mm,
        bullet_diameter_mm=bullet_diameter_mm,
        rings_cfg=rings_cfg,
        distance_yards=16.67
    )

    # Build AnnotatedShot list
    annotated_shots: List[AnnotatedShot] = []
    for s_res, peak in zip(shot_results, detected_peaks):
        annotated_shots.append(AnnotatedShot(
            id=s_res.shot_id,
            center_x=float(peak[0]),
            center_y=float(peak[1]),
            score=s_res.score_value,
            is_x=s_res.is_inner_x,
            caliber_mm=bullet_diameter_mm,
            shot_type="single",
            confidence=float(peak[2])
        ))

    # 3. Save JSON annotation
    ann = TargetAnnotation(
        target_id=target_id,
        image_filename=canonical_filename,
        target_type=target_type,
        bullseye_center=bull_center,
        shots=annotated_shots,
        canvas_size=canvas_size,
        pixels_per_mm=pixels_per_mm,
        metadata={
            "autolabeled": True,
            "redact_pii": redact_pii,
            "total_score": group_metrics.total_score,
            "x_count": group_metrics.x_count,
            "extreme_spread_mm": round(group_metrics.extreme_spread_mm, 2),
            "mean_radius_mm": round(group_metrics.mean_radius_mm, 2),
            "source_blur": round(rect_result.blur_metric, 1),
            "source_tilt": round(rect_result.tilt_angle_deg, 1),
        }
    )
    json_path = out_dir / f"{target_id}.json"
    ann.to_json(json_path)

    # 4. Generate Visual Verification Review Card
    review_path = out_dir / f"{target_id}_review.png"
    review_bgr = render_review_card(
        canonical_bgr=canonical_bgr,
        annotation=ann,
        output_path=review_path,
        redact_pii=redact_pii,
    )

    return ann, json_path, review_path


def render_review_card(
    canonical_bgr: np.ndarray,
    annotation: TargetAnnotation,
    output_path: Optional[Union[str, Path]] = None,
    redact_pii: bool = False,
) -> np.ndarray:
    """
    Renders visual verification review card with annotated crosshairs,
    caliber rings, shot badges, and score HUD banner.
    """
    review_bgr = canonical_bgr.copy()
    canvas_size = annotation.canvas_size
    pixels_per_mm = annotation.pixels_per_mm
    bull_center = annotation.bullseye_center
    annotated_shots = annotation.shots
    target_id = annotation.target_id
    target_type = annotation.target_type
    total_score = annotation.total_score
    x_count = annotation.x_count
    sf = canvas_size / 512.0

    if redact_pii:
        rings_cfg = TargetRingsConfig(target_type=target_type)
        rings_table = rings_cfg.get_rings_table()
        outer_margin_rad = int(round(rings_table[-1][1] * pixels_per_mm + 12.0 * (pixels_per_mm / 4.0)))
        bc_int_x, bc_int_y = int(round(bull_center[0])), int(round(bull_center[1]))
        header_h = max(28, int(round(bc_int_y - outer_margin_rad + 8)))
        badge_y = max(18, int(round(header_h * 0.6)))
        cv2.putText(
            review_bgr, "[COMPETITOR REGISTRATION REDACTED]", (18, badge_y),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.32, 0.38 * sf), (130, 130, 140), 1, cv2.LINE_AA
        )

    # Draw calibrated NRA scoring ring guide contours
    rings_cfg = TargetRingsConfig(target_type=target_type)
    rings_table = rings_cfg.get_rings_table()
    bc_x, bc_y = int(round(bull_center[0])), int(round(bull_center[1]))
    black_rad_mm = rings_cfg.get_black_radius_mm()
    for label, radius_mm, _ in rings_table:
        r_px = int(round(radius_mm * pixels_per_mm))
        if radius_mm <= black_rad_mm:
            ring_color = (0, 215, 255) if label == "X" else (160, 160, 175)
        else:
            ring_color = (75, 75, 85)
        cv2.circle(review_bgr, (bc_x, bc_y), r_px, ring_color, 1, lineType=cv2.LINE_AA)

    # Draw bullseye center crosshair (gold)
    cv2.drawMarker(review_bgr, (bc_x, bc_y), (0, 215, 255), cv2.MARKER_CROSS, int(18 * sf), 2, cv2.LINE_AA)

    # Draw detected shots
    for s in annotated_shots:
        sx, sy = int(round(s.center_x)), int(round(s.center_y))
        radius_px = int(round((s.caliber_mm / 2.0) * pixels_per_mm))
        gauge_color = (0, 215, 255) if s.is_x else ((0, 255, 120) if s.score >= 9 else (0, 165, 255))
        # Bullet caliber circle
        cv2.circle(review_bgr, (sx, sy), radius_px, gauge_color, 2, cv2.LINE_AA)
        cv2.drawMarker(review_bgr, (sx, sy), (0, 0, 255), cv2.MARKER_CROSS, int(12 * sf), 1, cv2.LINE_AA)
        # Shot badge
        tag = f"#{s.id}:{s.score}{'X' if s.is_x else ''}"
        cv2.putText(
            review_bgr, tag, (sx + radius_px + 2, sy - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40 * sf, (0, 255, 255), 1, cv2.LINE_AA
        )

    # Top HUD badge
    x_str = f" ({x_count}X)" if x_count > 0 else ""
    header = f"REVIEW: {target_id} ({target_type}) | Score: {total_score}{x_str} ({len(annotated_shots)} shots)"
    cv2.rectangle(review_bgr, (8, 8), (int(520 * sf), int(36 * sf)), (20, 20, 24), -1)
    cv2.putText(review_bgr, header, (16, int(26 * sf)), cv2.FONT_HERSHEY_SIMPLEX, 0.46 * sf, (0, 255, 200), 1, cv2.LINE_AA)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), review_bgr)

    return review_bgr
