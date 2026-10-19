#!/usr/bin/env python3
"""
Unified Progressive Resizing Cloud GPU Training Pipeline (Google Colab CLI).
Executes the full 4-stage Harmonic 256-Grid curriculum (256 -> 512 -> 768 -> 1024)
inside a single NVIDIA A100 or T4 cloud VM session.
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
    run_colab_progressive_training,
)


def run_pipeline(gpu: str = "A100"):
    start_time = time.time()
    colab_bin = get_colab_bin()

    print("=" * 76)
    print("  PROGRESSIVE RESIZING CLOUD TRAINING PIPELINE (GOOGLE COLAB CLI)")
    print("  Curriculum: 256x256 (1.0 px/mm) -> 512x512 (2.0 px/mm)")
    print("              -> 768x768 (3.0 px/mm) -> 1024x1024 (4.0 px/mm)")
    print(f"  Target Accelerator: NVIDIA {gpu}")
    print("=" * 76)

    print("\n[*] Initial Colab Compute Account Status:")
    check_colab_usage(colab_bin)

    ckpt_dir = str(PROJECT_ROOT / "weights")
    scorecards_dir = str(PROJECT_ROOT / "visual_inspection_reports")
    real_targets_dir = str(PROJECT_ROOT / "datasets" / "real_targets")
    use_real = os.path.exists(real_targets_dir)

    # Run the unified progressive cloud curriculum
    primary_ckpt = run_colab_progressive_training(
        gpu=gpu,
        high_mem=False,
        real_data_dir=real_targets_dir if use_real else None,
        real_ratio=0.35 if use_real else 0.0,
        checkpoint_dir="weights",
        scorecards_dir="visual_inspection_reports",
        timeout_seconds=3600.0,
    )

    elapsed_total = time.time() - start_time

    print("\n" + "=" * 76)
    print("  PROGRESSIVE RESIZING PIPELINE EXECUTION SUMMARY")
    print("=" * 76)
    print(f"  Total Duration:      {elapsed_total / 60.0:.2f} minutes")
    print(f"  Primary Weights:     {primary_ckpt}")
    print(f"  Stage Checkpoints:")

    resolutions = [256, 512, 768, 1024]
    for sz in resolutions:
        cp = os.path.join(ckpt_dir, f"target_scoring_hrnet_{sz}_best.pt")
        if os.path.exists(cp):
            size_mb = os.path.getsize(cp) / (1024 ** 2)
            print(f"    - Stage {sz}x{sz}: {cp} ({size_mb:.2f} MB)")
        else:
            print(f"    - Stage {sz}x{sz}: [MISSING]")

    print("\n  Diagnostic Inspection Scorecards:")
    for sz in resolutions:
        for tgt in ["b2", "b3"]:
            sc = os.path.join(scorecards_dir, f"colab_scorecard_{sz}_{tgt}.png")
            if os.path.exists(sc):
                print(f"    - Stage {sz}x{sz} {tgt.upper()}: {sc}")

    print("\n[*] Final Colab Compute Account Status:")
    check_colab_usage(colab_bin)
    print("=" * 76)


if __name__ == "__main__":
    gpu_choice = sys.argv[1] if len(sys.argv) > 1 else "A100"
    run_pipeline(gpu=gpu_choice)
