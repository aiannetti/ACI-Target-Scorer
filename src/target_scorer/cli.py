"""
Unified command-line interface for the Target Scorer package.
"""

import argparse
import sys

from target_scorer.infer import main as infer_main
from target_scorer.train import main as train_main


def rectify_cli():
    from target_scorer.geometry.rectifier import TargetRectifier
    import cv2
    import os
    parser = argparse.ArgumentParser(prog="target-scorer rectify", description="Target Rectification & Perspective Unwarping")
    parser.add_argument("--image", type=str, required=True, help="Path to input smartphone target photograph")
    parser.add_argument("--target", type=str, default="B-2", choices=["B-2", "B-3", "b-2", "b-3"], help="NRA target type")
    parser.add_argument("--output", type=str, default="rectified_target.png", help="Path to save canonical crop")
    parser.add_argument("--debug", type=str, default="rectification_debug.png", help="Path to save diagnostic panel")
    parser.add_argument("--canvas-size", type=int, default=512, help="Canonical canvas dimensions")
    parser.add_argument("--scale", type=float, default=4.0, help="Pixels per mm metric scale")
    args = parser.parse_args(sys.argv[2:])

    rectifier = TargetRectifier(target_type=args.target, canonical_dim=args.canvas_size, pixels_per_mm=args.scale)
    img_bgr = rectifier.load_image_with_exif(args.image)
    result = rectifier.process_raw_image(img_bgr, image_path=args.image)

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(args.output, result.canonical_crop)
    print(f"[✓] Saved canonical rectified crop ({result.canonical_crop.shape[1]}x{result.canonical_crop.shape[0]} @ {result.canonical_pixels_per_mm:.1f} px/mm) to: {args.output}")

    if args.debug:
        dbg_dir = os.path.dirname(os.path.abspath(args.debug))
        if dbg_dir:
            os.makedirs(dbg_dir, exist_ok=True)
        diag = rectifier.render_diagnostic_panel(img_bgr, result)
        cv2.imwrite(args.debug, diag)
        print(f"[✓] Saved rectification diagnostic visualization to: {args.debug}")

    print(f"    Tilt: {result.tilt_angle_deg:.1f}° | Scale: {result.raw_pixels_per_mm:.2f} raw px/mm | Blur Var: {result.blur_metric:.1f}")
    if result.warning:
        print(f"    [!] Warning: {result.warning}")


def main():
    parser = argparse.ArgumentParser(
        prog="target-scorer",
        description="Camera-Independent NRA Pistol Target Metrology & Scoring Engine"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")
    subparsers.add_parser("infer", help="Run shot detection, inward scoring, and scorecard rendering")
    subparsers.add_parser("train", help="Run procedural synthetic training pipeline")
    subparsers.add_parser("rectify", help="Run standalone geometric rectification")

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        parser.print_help()
        sys.exit(0 if (len(sys.argv) >= 2 and sys.argv[1] in ("-h", "--help")) else 1)

    subcommand = sys.argv[1]
    if subcommand == "infer":
        # Forward sys.argv
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        infer_main()
    elif subcommand == "train":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        train_main()
    elif subcommand == "rectify":
        rectify_cli()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
