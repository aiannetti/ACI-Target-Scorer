"""
Inference and scoring pipeline entry point.
"""

import argparse
import math
import os
import random
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from target_scorer.config import TargetConfig, TargetRingsConfig, TrainConfig
from target_scorer.geometry.rectifier import RectificationResult, TargetRectifier
from target_scorer.metrology.peaks import extract_subpixel_peaks
from target_scorer.metrology.scoring import GroupMetrics, ShotResult, score_target_group
from target_scorer.metrology.visualization import render_scoring_inspection_overlay
from target_scorer.models.unet import HighResHeatmapUNet
from target_scorer.synthetic.generator import (
    draw_standard_bullseye,
    generate_cardboard_texture,
    stamp_bullet_hole,
)


def generate_synthetic_demo_target(
    canvas_size: int = 512,
    pixels_per_mm: float = 4.0,
    target_type: str = "B-2"
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Generates a deterministic, photorealistic NRA B-2 or B-3 demo target card."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        random.seed(42)
        np.random.seed(42)

        cfg = TargetConfig(
            canvas_size=canvas_size,
            pixels_per_mm=pixels_per_mm,
            target_type=target_type,
        )
        img = generate_cardboard_texture(cfg)
        img, center, _ = draw_standard_bullseye(img, cfg)

        cx, cy = center
        scale_ratio = pixels_per_mm / 4.0
        demo_shots = [
            (cx + 6.0 * scale_ratio, cy - 8.0 * scale_ratio),
            (cx - 18.0 * scale_ratio, cy + 16.0 * scale_ratio),
            (cx - 28.0 * scale_ratio, cy + 24.0 * scale_ratio),
            (cx + 38.0 * scale_ratio, cy + 30.0 * scale_ratio),
            (cx - 55.0 * scale_ratio, cy - 35.0 * scale_ratio),
        ]

        for sx, sy in demo_shots:
            stamp_bullet_hole(img, (sx, sy), cfg, raggedness=1.0)
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)

    return img, center


def run_inference_and_scoring(
    checkpoint_path: str = "target_scoring_hrnet_best.pt",
    image_path: Optional[str] = None,
    output_path: str = "scored_inspection_result.png",
    debug_rect_path: Optional[str] = "rectification_debug.png",
    target_type: str = "B-2",
    distance_yards: float = 16.67,
    peak_threshold: float = 0.45,
    max_shots: Optional[int] = 10,
    device_str: Optional[str] = None,
    canvas_size: Optional[int] = None,
    pixels_per_mm: Optional[float] = None
) -> Tuple[List[ShotResult], GroupMetrics]:
    """Runs end-to-end target rectification, U-Net inference, inward scoring, and HUD generation."""
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(device_str)

    target_type = target_type.upper()
    rings_cfg = TargetRingsConfig(target_type=target_type)

    print("=" * 72)
    print("  NRA PRECISION PISTOL TARGET SCORING & GAUGE METROLOGY")
    print(f"  Target Type:        NRA {target_type} (50ft {'Slow' if target_type == 'B-2' else 'Timed/Rapid'})")
    print(f"  Device:             {device}")
    print(f"  Checkpoint:         {checkpoint_path}")
    print(f"  Distance:           50.0 ft ({distance_yards:.2f} yards)")
    print(f"  Target Shot Count:  {max_shots if max_shots is not None else 'All'} (Threshold: {peak_threshold})")
    print("=" * 72)

    # Search fallback paths for checkpoint if not found directly
    resolved_ckpt = checkpoint_path
    if not os.path.exists(resolved_ckpt):
        alt_path = os.path.join(os.path.dirname(__file__), "..", "..", "weights", os.path.basename(checkpoint_path))
        if os.path.exists(alt_path):
            resolved_ckpt = alt_path
        else:
            print(f"[-] Checkpoint '{checkpoint_path}' not found! Run training first or specify path.")
            sys.exit(1)

    try:
        checkpoint = torch.load(resolved_ckpt, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(resolved_ckpt, map_location=device)

    config = checkpoint.get("config", None)
    if isinstance(config, dict):
        ckpt_canvas = config.get("canvas_size", 512)
        ckpt_scale = config.get("pixels_per_mm", 4.0)
        bullet_diameter_mm = config.get("bullet_diameter_mm", 5.69)
    elif config is not None:
        ckpt_canvas = getattr(config, "canvas_size", 512)
        ckpt_scale = getattr(config, "pixels_per_mm", 4.0)
        bullet_diameter_mm = getattr(config, "bullet_diameter_mm", 5.69)
    else:
        ckpt_canvas = 512
        ckpt_scale = 4.0
        bullet_diameter_mm = 5.69

    eff_canvas_size = canvas_size if canvas_size is not None else ckpt_canvas
    eff_pixels_per_mm = pixels_per_mm if pixels_per_mm is not None else ckpt_scale

    model = HighResHeatmapUNet(in_channels=3, out_channels=2, base_dim=32).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rectifier = TargetRectifier(
        target_type=target_type,
        canonical_dim=eff_canvas_size,
        pixels_per_mm=eff_pixels_per_mm
    )

    rect_meta: Optional[RectificationResult] = None

    if image_path is not None:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Specified target image '{image_path}' does not exist.")
        print(f"[+] Loading raw image: '{image_path}'")
        raw_bgr = rectifier.load_image_with_exif(image_path)
        rect_meta = rectifier.process_raw_image(raw_bgr, image_path=image_path)
        img_canonical = rect_meta.canonical_crop

        print(f"    --> Raw Dimensions:   {raw_bgr.shape[1]}x{raw_bgr.shape[0]} px")
        print(f"    --> Motion Blur Var:  {rect_meta.blur_metric:.1f}")
        print(f"    --> Optical Tilt:     {rect_meta.tilt_angle_deg:.1f}° off-normal")
        print(f"    --> Detected Scale:   {rect_meta.raw_pixels_per_mm:.2f} raw px/mm -> {eff_pixels_per_mm:.1f} canonical px/mm")
        if rect_meta.warning:
            print(f"    [!] Warning:          {rect_meta.warning}")

        if debug_rect_path:
            debug_dir = os.path.dirname(os.path.abspath(debug_rect_path))
            if debug_dir:
                os.makedirs(debug_dir, exist_ok=True)
            debug_img = rectifier.render_diagnostic_panel(raw_bgr, rect_meta)
            cv2.imwrite(debug_rect_path, debug_img)
            print(f"[✓] Saved camera rectification diagnostic to: '{debug_rect_path}'")
    else:
        print(f"[*] No input image specified. Generating synthetic {target_type} demo target...")
        img_canonical, _ = generate_synthetic_demo_target(
            canvas_size=eff_canvas_size,
            pixels_per_mm=eff_pixels_per_mm,
            target_type=target_type
        )

    img_rgb = cv2.cvtColor(img_canonical, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    norm_img = (img_rgb - mean) / std
    tensor_img = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_heatmaps = model(tensor_img).squeeze(0).float().cpu().numpy()

    bullet_heatmap = pred_heatmaps[0]
    bullseye_heatmap = pred_heatmaps[1].copy()

    # Constrain bullseye search to central region of canonical rectified frame
    c_mid = eff_canvas_size / 2.0
    search_radius = eff_canvas_size * 0.15
    yy, xx = np.ogrid[:eff_canvas_size, :eff_canvas_size]
    central_mask = ((xx - c_mid) ** 2 + (yy - c_mid) ** 2) <= (search_radius ** 2)
    bullseye_heatmap[~central_mask] = 0.0

    min_dist_px = max(6.0, (bullet_diameter_mm * 0.45) * eff_pixels_per_mm)
    shot_peaks = extract_subpixel_peaks(
        bullet_heatmap,
        threshold=peak_threshold,
        min_distance_px=min_dist_px,
        max_peaks=max_shots
    )
    bull_peaks = extract_subpixel_peaks(
        bullseye_heatmap,
        threshold=0.25,
        min_distance_px=20.0 * (eff_pixels_per_mm / 4.0),
        max_peaks=1
    )
    bullseye_center = (bull_peaks[0][0], bull_peaks[0][1]) if bull_peaks else (c_mid, c_mid)

    print(f"[+] Detected {len(shot_peaks)} bullet impacts (Max requested: {max_shots}).")
    print(f"[+] Bullseye Center: ({bullseye_center[0]:.2f}, {bullseye_center[1]:.2f}) px")

    shot_results, metrics = score_target_group(
        shot_peaks=shot_peaks,
        bullseye_peak=bullseye_center,
        pixels_per_mm=eff_pixels_per_mm,
        bullet_diameter_mm=bullet_diameter_mm,
        rings_cfg=rings_cfg,
        distance_yards=distance_yards
    )

    print("-" * 72)
    print(f"  TARGET:              NRA {target_type} (50 ft)")
    print(f"  TOTAL SCORE:         {metrics.total_score} - {metrics.x_count}X / {metrics.max_possible_score}")
    print(f"  EXTREME SPREAD:      {metrics.extreme_spread_mm:.2f} mm  ({metrics.extreme_spread_moa:.2f} MOA)")
    print(f"  MEAN RADIUS:         {metrics.mean_radius_mm:.2f} mm  ({metrics.mean_radius_moa:.2f} MOA)")
    print(f"  WINDAGE BIAS:        {metrics.windage_bias_mm:+.2f} mm ({'Right' if metrics.windage_bias_mm >= 0 else 'Left'})")
    print(f"  ELEVATION BIAS:      {metrics.elevation_bias_mm:+.2f} mm ({'High' if metrics.elevation_bias_mm >= 0 else 'Low'})")
    print("-" * 72)
    for s in shot_results:
        print(f"  Shot #{s.shot_id:02d} | Ring: {s.score_label:>2} | Center Dist: {s.radial_dist_mm:5.2f} mm | Gauge Dist: {s.effective_gauge_dist_mm:5.2f} mm | Conf: {s.confidence:.2f}")
    print("-" * 72)

    overlay_img = render_scoring_inspection_overlay(
        image_bgr=img_canonical,
        bullet_heatmap=bullet_heatmap,
        shot_results=shot_results,
        bullseye_center=bullseye_center,
        metrics=metrics,
        pixels_per_mm=eff_pixels_per_mm,
        bullet_diameter_mm=bullet_diameter_mm,
        rings_cfg=rings_cfg,
        rect_meta=rect_meta
    )

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, overlay_img)
    print(f"[✓] Visual inspection scorecard saved to: '{output_path}'")
    return shot_results, metrics


def main():
    parser = argparse.ArgumentParser(description="Camera-Independent Target Scoring & Metrology Engine (NRA B-2 & B-3)")
    parser.add_argument("--checkpoint", type=str, default="target_scoring_hrnet_best.pt", help="Path to trained PyTorch weights")
    parser.add_argument("--image", type=str, default=None, help="Path to raw smartphone target photo")
    parser.add_argument("--target", type=str, default="B-2", choices=["B-2", "B-3", "b-2", "b-3"], help="NRA target face ('B-2' or 'B-3')")
    parser.add_argument("--output", type=str, default="scored_inspection_result.png", help="Path to save output scorecard image")
    parser.add_argument("--debug-rect", type=str, default="rectification_debug.png", help="Path to save rectification side-by-side visualization")
    parser.add_argument("--distance", type=float, default=16.67, help="Shooting distance in yards (50 ft = 16.67 yd)")
    parser.add_argument("--shots", type=int, default=10, help="Number of shots to evaluate (default: 10 for standard NRA string)")
    parser.add_argument("--threshold", type=float, default=0.45, help="Heatmap peak detection confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Inference device: 'cuda', 'cpu', or 'mps'")
    parser.add_argument("--canvas-size", type=int, default=None, help="Override canonical canvas size (e.g. 256 for fast verification)")
    parser.add_argument("--scale", type=float, default=None, help="Override pixels_per_mm scale (e.g. 2.0 for 256x256)")

    args = parser.parse_args()

    run_inference_and_scoring(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        output_path=args.output,
        debug_rect_path=args.debug_rect,
        target_type=args.target,
        distance_yards=args.distance,
        peak_threshold=args.threshold,
        max_shots=args.shots,
        device_str=args.device,
        canvas_size=args.canvas_size,
        pixels_per_mm=args.scale,
    )


if __name__ == "__main__":
    main()
