#!/usr/bin/env python3
"""
Automated Multi-Resolution Google Colab Cloud Training Pipeline.
Sequentially trains and updates target scoring weights for:
  - 512 x 512 (NVIDIA T4, sigma=2.5)
  - 800 x 800 (NVIDIA A100, sigma=2.8)
  - 1024 x 1024 (NVIDIA A100, sigma=3.2)
Using the multi-tone bullet void procedural synthetic generator (scanner white, dark void, tan backer).
"""

import os
import sys
import time
from pathlib import Path

# Ensure package root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from target_scorer.colab_runner import (
    check_colab_usage,
    get_colab_bin,
    list_colab_sessions,
    run_colab_training,
)


def run_pipeline():
    start_time = time.time()
    colab_bin = get_colab_bin()

    print("=" * 76)
    print("  MULTI-RESOLUTION CLOUD GPU TRAINING PIPELINE (GOOGLE COLAB CLI)")
    print("  Target Resolutions: 512x512 (T4) | 800x800 (A100) | 1024x1024 (A100)")
    print("  Dataset: Multi-Tone Bullet Void Synthetic Generator")
    print("=" * 76)

    print("\n[*] Initial Colab Compute Account Status:")
    check_colab_usage(colab_bin)

    runs = [
        {
            "name": "512x512 High-Speed Model",
            "canvas_size": 512,
            "scale": 4.0,
            "epochs": 12,
            "batch_size": 8,
            "train_samples": 350,
            "val_samples": 50,
            "lr": 3.0e-4,
            "sigma": 2.5,
            "gpu": "T4",
            "initial_weights": str(PROJECT_ROOT / "weights" / "target_scoring_hrnet_512_best.pt"),
            "checkpoint": str(PROJECT_ROOT / "weights" / "target_scoring_hrnet_512_best.pt"),
            "scorecard_b2": str(PROJECT_ROOT / "visual_inspection_reports" / "colab_scorecard_512_b2.png"),
            "scorecard_b3": str(PROJECT_ROOT / "visual_inspection_reports" / "colab_scorecard_512_b3.png"),
        },
        {
            "name": "800x800 Balanced Precision Model",
            "canvas_size": 800,
            "scale": 4.0,
            "epochs": 12,
            "batch_size": 8,
            "train_samples": 350,
            "val_samples": 50,
            "lr": 2.5e-4,
            "sigma": 2.8,
            "gpu": "A100",
            "initial_weights": str(PROJECT_ROOT / "weights" / "target_scoring_hrnet_512_best.pt"),
            "checkpoint": str(PROJECT_ROOT / "weights" / "target_scoring_hrnet_800_best.pt"),
            "scorecard_b2": str(PROJECT_ROOT / "visual_inspection_reports" / "colab_scorecard_800_b2.png"),
            "scorecard_b3": str(PROJECT_ROOT / "visual_inspection_reports" / "colab_scorecard_800_b3.png"),
        },
        {
            "name": "1024x1024 Ultra-Resolution Subpixel Model",
            "canvas_size": 1024,
            "scale": 4.0,
            "epochs": 15,
            "batch_size": 8,
            "train_samples": 400,
            "val_samples": 50,
            "lr": 2.0e-4,
            "sigma": 3.2,
            "gpu": "A100",
            "initial_weights": str(PROJECT_ROOT / "weights" / "target_scoring_hrnet_1024_best.pt"),
            "checkpoint": str(PROJECT_ROOT / "weights" / "target_scoring_hrnet_1024_best.pt"),
            "scorecard_b2": str(PROJECT_ROOT / "visual_inspection_reports" / "colab_scorecard_1024_b2.png"),
            "scorecard_b3": str(PROJECT_ROOT / "visual_inspection_reports" / "colab_scorecard_1024_b3.png"),
        },
    ]

    results = []

    for idx, run_cfg in enumerate(runs, 1):
        print("\n" + "#" * 76)
        print(f"  RUN [{idx}/{len(runs)}]: {run_cfg['name']} ({run_cfg['canvas_size']}x{run_cfg['canvas_size']} on {run_cfg['gpu']})")
        print("#" * 76)

        t0 = time.time()
        try:
            init_w = run_cfg["initial_weights"] if Path(run_cfg["initial_weights"]).exists() else None
            ckpt_path = run_colab_training(
                canvas_size=run_cfg["canvas_size"],
                pixels_per_mm=run_cfg["scale"],
                epochs=run_cfg["epochs"],
                batch_size=run_cfg["batch_size"],
                train_samples_per_epoch=run_cfg["train_samples"],
                val_samples_per_epoch=run_cfg["val_samples"],
                learning_rate=run_cfg["lr"],
                gpu=run_cfg["gpu"],
                checkpoint_dest=run_cfg["checkpoint"],
                scorecard_b2_dest=run_cfg["scorecard_b2"],
                scorecard_b3_dest=run_cfg["scorecard_b3"],
                sigma=run_cfg["sigma"],
                initial_weights=init_w,
            )
            elapsed = time.time() - t0
            ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
            print(f"[✓] Successfully completed {run_cfg['name']} in {elapsed / 60.0:.1f} min ({ckpt_size_mb:.2f} MB)")
            results.append({
                "name": run_cfg["name"],
                "canvas_size": run_cfg["canvas_size"],
                "gpu": run_cfg["gpu"],
                "status": "SUCCESS",
                "time_min": elapsed / 60.0,
                "checkpoint": ckpt_path,
                "size_mb": ckpt_size_mb,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[-] FAILED run {run_cfg['name']}: {e}")
            results.append({
                "name": run_cfg["name"],
                "canvas_size": run_cfg["canvas_size"],
                "gpu": run_cfg["gpu"],
                "status": f"FAILED ({e})",
                "time_min": elapsed / 60.0,
                "checkpoint": None,
                "size_mb": 0.0,
            })

    total_time = time.time() - start_time
    print("\n" + "=" * 76)
    print("  MULTI-RESOLUTION CLOUD TRAINING PIPELINE SUMMARY")
    print("=" * 76)
    for r in results:
        status_sym = "✓" if r["status"] == "SUCCESS" else "✗"
        print(f"  [{status_sym}] {r['name']:<42} | Status: {r['status']:<10} | Time: {r['time_min']:.1f}m | Size: {r['size_mb']:.1f} MB")
    print("-" * 76)
    print(f"  Total Elapsed Time: {total_time / 60.0:.1f} minutes")
    print("\n[*] Final Colab Compute Account Status:")
    check_colab_usage(colab_bin)
    print("=" * 76)

    # Return success if at least one passed and none failed fatally
    failures = [r for r in results if r["status"] != "SUCCESS"]
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
