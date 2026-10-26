"""
Unified command-line interface for the Target Scorer package.
"""

import argparse
import sys

from target_scorer.infer import main as infer_main
from target_scorer.train import main as train_main
from target_scorer.colab_runner import main as colab_main


def rectify_cli():
    from target_scorer.geometry.rectifier import TargetRectifier
    import cv2
    import os
    parser = argparse.ArgumentParser(prog="target-scorer rectify", description="Target Rectification & Perspective Unwarping")
    parser.add_argument("--image", type=str, required=True, help="Path to input smartphone target photograph")
    parser.add_argument("--target", type=str, default="B-2", choices=["B-2", "B-3", "b-2", "b-3"], help="NRA target type")
    parser.add_argument("--output", type=str, default="rectified_target.png", help="Path to save canonical crop")
    parser.add_argument("--debug", type=str, default="rectification_debug.png", help="Path to save diagnostic panel")
    parser.add_argument("--canvas-size", type=int, default=1024, help="Canonical canvas dimensions (default: 1024)")
    parser.add_argument("--scale", type=float, default=4.0, help="Pixels per mm metric scale")
    args = parser.parse_args(sys.argv[2:])

    rectifier = TargetRectifier(target_type=args.target, canonical_dim=args.canvas_size, pixels_per_mm=args.scale)
    img_bgr = rectifier.load_image_with_exif(args.image)
    result = rectifier.process_raw_image(img_bgr)

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


def autolabel_cli():
    import argparse
    from pathlib import Path
    from target_scorer.data.autolabel import autolabel_target_image

    parser = argparse.ArgumentParser(prog="target-scorer autolabel", description="Assisted Auto-Labeling for Real Target Datasets")
    parser.add_argument("--image", type=str, default=None, help="Path to single input target photograph")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory of target photographs to batch auto-label")
    parser.add_argument("--output-dir", type=str, default="private/data/real_targets", help="Directory to save annotations and review cards (defaults to private submodule path 'private/data/real_targets')")
    parser.add_argument("--target", type=str, default="B-2", choices=["B-2", "B-3", "b-2", "b-3"], help="NRA target type")
    parser.add_argument("--checkpoint", type=str, default="weights/target_scoring_hrnet_1024_best.pt", help="Path to model weights")
    parser.add_argument("--canvas-size", type=int, default=1024, help="Canonical canvas dimension")
    parser.add_argument("--scale", type=float, default=4.0, help="Pixels per mm metric scale")
    parser.add_argument("--threshold", type=float, default=0.40, help="Peak confidence threshold")
    parser.add_argument("--shots", type=int, default=10, help="Maximum number of shots to detect")
    parser.add_argument("--redact-pii", action="store_true", default=False, help="Redact competitor margin registration info (Name, NRA#, etc.)")
    args = parser.parse_args(sys.argv[2:])

    if not args.image and not args.input_dir:
        print("[-] Error: Either --image or --input-dir must be specified.")
        sys.exit(1)

    targets_to_process = []
    if args.image:
        targets_to_process.append(Path(args.image))
    if args.input_dir:
        in_p = Path(args.input_dir)
        for ext in ("*.heic", "*.HEIC", "*.jpg", "*.jpeg", "*.JPG", "*.png", "*.PNG", "*.tiff", "*.tif", "*.dng", "*.DNG", "*.webp"):
            targets_to_process.extend(in_p.rglob(ext))

    targets_to_process = sorted(list(set(targets_to_process)))
    print(f"[+] Found {len(targets_to_process)} target(s) to auto-label.")

    success_count = 0
    failure_count = 0
    for i, target_file in enumerate(targets_to_process, 1):
        print(f"[{i}/{len(targets_to_process)}] Processing '{target_file.name}'...")
        try:
            ann, json_p, rev_p = autolabel_target_image(
                image_source=target_file,
                output_dir=args.output_dir,
                checkpoint_path=args.checkpoint,
                target_type=args.target.upper(),
                canvas_size=args.canvas_size,
                pixels_per_mm=args.scale,
                peak_threshold=args.threshold,
                max_shots=args.shots,
                redact_pii=args.redact_pii,
            )
            print(f"    [✓] Auto-labeled {len(ann.shots)} shots -> JSON: {json_p.name} | Review: {rev_p.name}")
            success_count += 1
        except Exception as e:
            print(f"    [-] Failed to process '{target_file.name}': {e}")
            failure_count += 1

    print(f"\n[+] Auto-labeling complete: {success_count} succeeded, {failure_count} failed.")
    if failure_count > 0 and success_count == 0:
        sys.exit(1)


def colab_cli():
    try:
        colab_main()
    except FileNotFoundError as e:
        print(f"[-] Error: {e}")
        print("    To use Google Colab cloud orchestration, install the CLI:")
        print("        pip install google-colab-cli")
        print("    Then authenticate your Google account:")
        print("        colab login")
        sys.exit(1)


def colab_autolabel_cli():
    from target_scorer.colab_runner import run_colab_autolabel
    parser = argparse.ArgumentParser(prog="target-scorer colab-autolabel", description="Cloud-Accelerated Assisted Auto-Labeling on Google Colab CLI")
    parser.add_argument("--input-dir", type=str, required=True, help="Local directory of target photographs to upload and auto-label")
    parser.add_argument("--output-dir", type=str, default="datasets/autolabeled_targets", help="Local directory to download annotations and review cards")
    parser.add_argument("--target", type=str, default="B-2", choices=["B-2", "B-3", "b-2", "b-3"], help="NRA target type")
    parser.add_argument("--checkpoint", type=str, default="weights/target_scoring_hrnet_1024_best.pt", help="Path to local model weights to upload")
    parser.add_argument("--canvas-size", type=int, default=1024, help="Canonical canvas dimension")
    parser.add_argument("--scale", type=float, default=4.0, help="Pixels per mm metric scale")
    parser.add_argument("--threshold", type=float, default=0.40, help="Peak confidence threshold")
    parser.add_argument("--gpu", type=str, default="T4", choices=["A100", "H100", "T4", "L4", "G4"], help="Cloud GPU type (default: T4)")
    parser.add_argument("--session", type=str, default=None, help="Custom Colab session name")
    parser.add_argument("--timeout", type=float, default=1800.0, help="Maximum execution timeout in seconds")
    parser.add_argument("--redact-pii", action="store_true", default=False, help="Redact competitor margin registration info (Name, NRA#, etc.)")
    args = parser.parse_args(sys.argv[2:])

    try:
        run_colab_autolabel(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_type=args.target.upper(),
            checkpoint=args.checkpoint,
            canvas_size=args.canvas_size,
            pixels_per_mm=args.scale,
            threshold=args.threshold,
            gpu=args.gpu,
            session_name=args.session,
            timeout_seconds=args.timeout,
            redact_pii=args.redact_pii,
        )
    except FileNotFoundError as e:
        print(f"[-] Error: {e}")
        print("    To use Google Colab cloud orchestration, install the CLI:")
        print("        pip install google-colab-cli")
        print("    Then authenticate your Google account:")
        print("        colab login")
        sys.exit(1)


def annotate_cli():
    import argparse
    from target_scorer.gui.annotation_editor import launch_annotation_editor

    parser = argparse.ArgumentParser(
        prog="target-scorer annotate",
        description="Interactive GUI for Reviewing and Correcting Shot Markers and Target Scores"
    )
    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default=None,
        help="Path to directory containing JSON annotations (e.g. autolabeled_photos/B-3 or private/data/real_targets) or single JSON file"
    )
    parser.add_argument(
        "--dir",
        type=str,
        dest="dir_opt",
        default=None,
        help="Directory containing JSON annotations and target images"
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        choices=["B-2", "B-3", "b-2", "b-3"],
        help="Override NRA target type (B-2 or B-3)"
    )
    parser.add_argument(
        "--copy-to-dir",
        type=str,
        default=None,
        help="Optional destination directory to copy raw photo, canonical image, and annotations to on Save"
    )
    args = parser.parse_args(sys.argv[2:])

    target_path = args.dir_opt or args.path
    launch_annotation_editor(
        target_path=target_path,
        target_type=args.target.upper() if args.target else None,
        copy_to_dir=args.copy_to_dir
    )


def main():
    parser = argparse.ArgumentParser(
        prog="target-scorer",
        description="Camera-Independent NRA Pistol Target Metrology & Scoring Engine"
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")
    subparsers.add_parser("infer", help="Run shot detection, inward scoring, and scorecard rendering")
    subparsers.add_parser("train", help="Run procedural synthetic training pipeline")
    subparsers.add_parser("colab-train", help="Run cloud GPU training via Google Colab CLI (A100/H100/T4)")
    subparsers.add_parser("colab-progressive-train", help="Run multi-stage Progressive Resizing training via Google Colab CLI (A100/T4)")
    subparsers.add_parser("colab-autolabel", help="Run cloud GPU assisted auto-labeling via Google Colab CLI")
    subparsers.add_parser("rectify", help="Run standalone geometric rectification")
    subparsers.add_parser("autolabel", help="Run model-assisted active learning and auto-labeling")
    subparsers.add_parser("annotate", help="Launch interactive GUI to manipulate shot markers and re-score targets")

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
    elif subcommand == "colab-train":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        colab_cli()
    elif subcommand == "colab-progressive-train":
        sys.argv = [sys.argv[0], "--progressive"] + sys.argv[2:]
        colab_cli()
    elif subcommand == "colab-autolabel":
        colab_autolabel_cli()
    elif subcommand == "rectify":
        rectify_cli()
    elif subcommand == "autolabel":
        autolabel_cli()
    elif subcommand == "annotate":
        annotate_cli()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
