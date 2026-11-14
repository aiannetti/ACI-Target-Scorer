"""
Assisted auto-labeling and active learning tooling.
Predicts shot coordinates and bullseye center on raw or canonical targets using pre-trained weights,
generating standard TargetAnnotation JSON files and visual verification review cards.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from target_scorer.config import TargetRingsConfig
from target_scorer.data.schema import AnnotatedShot, TargetAnnotation
from target_scorer.geometry.rectifier import TargetRectifier
from target_scorer.metrology.peaks import extract_subpixel_peaks, filter_spurious_peaks
from target_scorer.metrology.scoring import score_target_group
from target_scorer.models.unet import HighResHeatmapUNet

_DEFAULT_SENTINEL = object()


def resolve_checkpoint_path(checkpoint_path: Optional[Union[str, Path]]) -> Optional[str]:
    """
    Resolves model checkpoint path with fallback to repository weights directory.

    If checkpoint_path is None:
        Checks for 'weights/target_scoring_hrnet_1024_best.pt' relative to repository root.
        Returns the resolved path if found, otherwise returns None (fresh model fallback).
    If checkpoint_path is provided:
        Checks if candidate path exists. If not, checks 'weights/<basename>'.
        Returns the resolved path string if found, otherwise raises FileNotFoundError.
    """
    if checkpoint_path is None:
        default_candidate = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "weights", "target_scoring_hrnet_1024_best.pt"
        )
        if os.path.exists(default_candidate):
            return os.path.abspath(default_candidate)
        return None

    resolved_ckpt = str(checkpoint_path)
    if os.path.exists(resolved_ckpt):
        return os.path.abspath(resolved_ckpt)

    alt = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "weights", os.path.basename(resolved_ckpt)
    )
    if os.path.exists(alt):
        return os.path.abspath(alt)

    raise FileNotFoundError(f"Model checkpoint '{checkpoint_path}' not found.")


class AutoLabelPipeline:
    """
    Reusable, stateful auto-labeling pipeline for NRA target scoring.
    Caches the HighResHeatmapUNet model and device across repeated predict calls.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = "weights/target_scoring_hrnet_1024_best.pt",
        device_str: Optional[str] = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.device_str = device_str

        if device_str is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        else:
            self.device = torch.device(device_str)

        self._model: Optional[HighResHeatmapUNet] = None
        self._loaded_checkpoint: Optional[str] = None
        self._is_fresh_model: bool = False

    @property
    def loaded_checkpoint(self) -> Optional[str]:
        """Returns the absolute path of the currently loaded model weights checkpoint, or None."""
        return self._loaded_checkpoint

    @property
    def model(self) -> Optional[HighResHeatmapUNet]:
        """Returns the underlying HighResHeatmapUNet model instance, or None if not yet loaded."""
        return self._model

    def ensure_model_loaded(
        self, checkpoint_path: Union[Optional[str], Optional[Path], object] = _DEFAULT_SENTINEL
    ) -> HighResHeatmapUNet:
        """
        Ensures HighResHeatmapUNet is instantiated and weights are loaded onto target device.
        Caches model across repeated calls for the same resolved checkpoint path.
        """
        target_ckpt = self.checkpoint_path if checkpoint_path is _DEFAULT_SENTINEL else checkpoint_path
        resolved_ckpt = resolve_checkpoint_path(target_ckpt)

        # Reuse existing model if resolved checkpoint matches
        if self._model is not None:
            if resolved_ckpt is not None and self._loaded_checkpoint == resolved_ckpt:
                return self._model
            if resolved_ckpt is None and self._is_fresh_model:
                return self._model

        # Instantiate model on device
        model = HighResHeatmapUNet(in_channels=3, out_channels=2, base_dim=32).to(self.device)

        if resolved_ckpt is not None:
            try:
                ckpt = torch.load(resolved_ckpt, map_location=self.device, weights_only=False)
            except TypeError:
                ckpt = torch.load(resolved_ckpt, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            self._loaded_checkpoint = resolved_ckpt
            self._is_fresh_model = False
        else:
            self._loaded_checkpoint = None
            self._is_fresh_model = True

        model.eval()
        self._model = model
        self.checkpoint_path = target_ckpt
        return self._model

    def predict(
        self,
        image_source: Union[str, Path, np.ndarray],
        target_type: str = "B-2",
        canvas_size: int = 1024,
        pixels_per_mm: float = 4.0,
        peak_threshold: float = 0.40,
        max_shots: int = 10,
        bullet_diameter_mm: float = 5.69,
        target_id: Optional[str] = None,
        redact_pii: bool = False,
    ) -> Tuple[TargetAnnotation, np.ndarray]:
        """
        Runs TargetRectifier, cached HighResHeatmapUNet inference, subpixel peak extraction,
        spurious artifact filtering, bullseye localization, optional PII redaction, and
        NRA inward scoring entirely in memory.

        Returns:
            Tuple of (annotation: TargetAnnotation, canonical_bgr: np.ndarray)
        """
        model = self.ensure_model_loaded()

        # Derive target ID
        if target_id is None:
            if isinstance(image_source, (str, Path)) and str(image_source) != "-":
                resolved_target_id = Path(image_source).stem
            else:
                import uuid
                resolved_target_id = f"autolabel_{uuid.uuid4().hex[:8]}"
        else:
            resolved_target_id = str(target_id)

        # 1. Ingest & Rectify image
        rectifier = TargetRectifier(target_type=target_type, canonical_dim=canvas_size, pixels_per_mm=pixels_per_mm)
        rect_result = rectifier.rectify(image_source)
        canonical_bgr = rect_result.canonical_crop.copy()

        # 2. Preprocess & execute HighResHeatmapUNet inference
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        rgb = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        norm_tensor = ((tensor - torch.from_numpy(mean)) / torch.from_numpy(std)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_hmap = model(norm_tensor).squeeze(0).cpu().numpy()

        bullet_hmap = pred_hmap[0]
        bull_hmap = pred_hmap[1]

        # 3. Extract candidate bullet coordinates
        min_dist_px = max(6.0, 2.5 * pixels_per_mm)
        candidate_peaks = extract_subpixel_peaks(
            bullet_hmap,
            threshold=peak_threshold,
            min_distance_px=min_dist_px,
            max_peaks=max(30, max_shots * 3) if max_shots is not None else None,
        )

        # 4. Filter spurious printed target artifacts (official NRA seal, header text)
        detected_peaks = filter_spurious_peaks(
            peaks=candidate_peaks,
            canonical_bgr=canonical_bgr,
            canvas_size=canvas_size,
            pixels_per_mm=pixels_per_mm,
            target_type=target_type,
            max_peaks=max_shots,
        )

        # 5. Extract bullseye center (TargetRectifier centers the bullseye at canvas_size / 2)
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

        # 6. Optional Competitor Registration PII Redaction
        rings_cfg = TargetRingsConfig(target_type=target_type)
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

        # 7. Score detected shots
        peaks_tuples = [(p[0], p[1], p[2]) for p in detected_peaks]
        shot_results, group_metrics = score_target_group(
            shot_peaks=peaks_tuples,
            bullseye_peak=bull_center,
            pixels_per_mm=pixels_per_mm,
            bullet_diameter_mm=bullet_diameter_mm,
            rings_cfg=rings_cfg,
            distance_yards=16.67,
        )

        # 8. Build AnnotatedShot list
        annotated_shots: List[AnnotatedShot] = []
        for s_res, peak in zip(shot_results, detected_peaks):
            annotated_shots.append(
                AnnotatedShot(
                    id=s_res.shot_id,
                    center_x=float(peak[0]),
                    center_y=float(peak[1]),
                    score=s_res.score_value,
                    is_x=s_res.is_inner_x,
                    caliber_mm=bullet_diameter_mm,
                    shot_type="single",
                    confidence=float(peak[2]),
                )
            )

        metadata = {
            "autolabeled": True,
            "redact_pii": redact_pii,
            "total_score": group_metrics.total_score,
            "x_count": group_metrics.x_count,
            "extreme_spread_mm": round(group_metrics.extreme_spread_mm, 2),
            "mean_radius_mm": round(group_metrics.mean_radius_mm, 2),
            "source_blur": round(rect_result.blur_metric, 1),
            "source_tilt": round(rect_result.tilt_angle_deg, 1),
        }
        if isinstance(image_source, (str, Path)) and str(image_source) != "-":
            metadata["source_image"] = str(Path(image_source).resolve())

        canonical_filename = f"{resolved_target_id}_canonical.png"
        ann = TargetAnnotation(
            target_id=resolved_target_id,
            image_filename=canonical_filename,
            target_type=target_type,
            bullseye_center=bull_center,
            shots=annotated_shots,
            canvas_size=canvas_size,
            pixels_per_mm=pixels_per_mm,
            metadata=metadata,
        )

        return ann, canonical_bgr


def autolabel_target_image(
    image_source: Union[str, Path, np.ndarray],
    output_dir: Union[str, Path],
    checkpoint_path: Optional[str] = "weights/target_scoring_hrnet_1024_best.pt",
    target_type: str = "B-2",
    canvas_size: int = 1024,
    pixels_per_mm: float = 4.0,
    peak_threshold: float = 0.25,
    max_shots: int = 10,
    bullet_diameter_mm: float = 5.69,
    device_str: Optional[str] = None,
    target_id: Optional[str] = None,
    redact_pii: bool = False,
    pipeline: Optional[AutoLabelPipeline] = None,
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

    if pipeline is None:
        pipeline = AutoLabelPipeline(checkpoint_path=checkpoint_path, device_str=device_str)

    ann, canonical_bgr = pipeline.predict(
        image_source=image_source,
        target_type=target_type,
        canvas_size=canvas_size,
        pixels_per_mm=pixels_per_mm,
        peak_threshold=peak_threshold,
        max_shots=max_shots,
        bullet_diameter_mm=bullet_diameter_mm,
        target_id=target_id,
        redact_pii=redact_pii,
    )

    # 1. Save canonical image (redacted if redact_pii=True)
    canonical_filename = ann.image_filename
    canonical_path = out_dir / canonical_filename
    cv2.imwrite(str(canonical_path), canonical_bgr)

    # 2. Save JSON annotation
    json_path = out_dir / f"{ann.target_id}.json"
    ann.to_json(json_path)

    # 3. Generate Visual Verification Review Card
    review_path = out_dir / f"{ann.target_id}_review.png"
    render_review_card(
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
