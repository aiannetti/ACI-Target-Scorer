"""
Automated Google Colab CLI training runner for high-resolution models (800x800 and 1024x1024).
Provisions cloud GPU (NVIDIA A100 / T4) via Colab CLI, syncs codebase, executes training,
retrieves weights and visual scorecard, and terminates session safely.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from typing import Optional


def get_colab_bin() -> str:
    """Finds the path to the colab executable."""
    colab_path = shutil.which("colab")
    if colab_path:
        return colab_path
    home_bin = os.path.expanduser("~/.local/bin/colab")
    if os.path.exists(home_bin):
        return home_bin
    raise FileNotFoundError("Google Colab CLI ('colab') executable not found in PATH or ~/.local/bin/colab.")


def run_colab_training(
    canvas_size: int = 800,
    pixels_per_mm: float = 4.0,
    epochs: int = 15,
    batch_size: int = 8,
    train_samples_per_epoch: int = 400,
    val_samples_per_epoch: int = 50,
    learning_rate: float = 3.5e-4,
    gpu: str = "A100",
    session_name: Optional[str] = None,
    checkpoint_dest: Optional[str] = None,
    scorecard_dest: Optional[str] = None,
    timeout_seconds: float = 3600.0,
    sigma: Optional[float] = None,
) -> str:
    """
    End-to-end cloud training orchestration using Google Colab CLI:
      1. Repackages latest codebase into zip bundle.
      2. Creates a cloud GPU session (default: NVIDIA A100).
      3. Uploads bundle and sets up remote Python environment.
      4. Executes training with mixed-precision (AMP) acceleration.
      5. Generates visual inspection scorecard.
      6. Downloads optimal weights and scorecard to local machine.
      7. Terminates cloud session automatically to conserve compute units.
    """
    colab_bin = get_colab_bin()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if session_name is None:
        session_name = f"target-train-{canvas_size}-{int(time.time()) % 100000:05d}"

    if checkpoint_dest is None:
        checkpoint_dest = os.path.join(project_root, "weights", f"target_scoring_hrnet_{canvas_size}_best.pt")

    if scorecard_dest is None:
        scorecard_dest = os.path.join(project_root, "visual_inspection_reports", f"colab_scorecard_{canvas_size}.png")

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_dest)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(scorecard_dest)), exist_ok=True)

    print("=" * 72)
    print("  GOOGLE COLAB CLI CLOUD TRAINING ORCHESTRATION")
    print(f"  Canvas Dimensions:   {canvas_size} x {canvas_size} px")
    print(f"  Metrology Scale:     {pixels_per_mm} pixels/mm")
    print(f"  Target Accelerator:  NVIDIA {gpu}")
    print(f"  Colab Session Name:  {session_name}")
    print(f"  Local Output Ckpt:   {checkpoint_dest}")
    print("=" * 72)

    # 1. Repackage latest codebase bundle
    print("[1/6] Packaging latest codebase bundle...")
    bundle_script = os.path.join(project_root, "scripts", "create_colab_bundle.py")
    bundle_zip = os.path.join(project_root, "target_scorer_colab_bundle.zip")
    if os.path.exists(bundle_script):
        subprocess.run([sys.executable, bundle_script], cwd=project_root, check=True)
    elif not os.path.exists(bundle_zip):
        raise FileNotFoundError(f"Colab bundle '{bundle_zip}' not found and packaging script missing.")

    session_created = False
    try:
        # 2. Provision cloud GPU VM
        print(f"[2/6] Provisioning Colab VM session '{session_name}' with {gpu} accelerator...")
        cmd_new = [colab_bin, "new", "-s", session_name, "--gpu", gpu]
        res_new = subprocess.run(cmd_new)
        if res_new.returncode != 0:
            raise RuntimeError(f"Failed to allocate Colab session '{session_name}' with GPU {gpu}.")
        session_created = True

        # 3. Upload codebase bundle
        print(f"[3/6] Uploading codebase archive ({os.path.getsize(bundle_zip) / (1024**2):.1f} MB) to Colab VM...")
        cmd_upload = [colab_bin, "upload", bundle_zip, "/content/target_scorer_colab_bundle.zip", "-s", session_name]
        subprocess.run(cmd_upload, check=True)

        # 4. Unpack and install on remote VM
        print("[4/6] Initializing remote environment on Colab VM...")
        setup_script = (
            "import subprocess, sys\n"
            "subprocess.run(['unzip', '-q', '-o', '/content/target_scorer_colab_bundle.zip', '-d', '/content/'], check=True)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '/content/target_scorer'], check=True)\n"
            "print('[Colab VM] Target Scorer installed successfully.')\n"
        )
        proc_setup = subprocess.Popen(
            [colab_bin, "exec", "-s", session_name, "--timeout", "120"],
            stdin=subprocess.PIPE,
            text=True
        )
        proc_setup.communicate(input=setup_script)
        if proc_setup.returncode != 0:
            raise RuntimeError("Remote environment setup failed on Colab VM.")

        # 5. Execute training with mixed-precision (AMP)
        print(f"[5/6] Executing {canvas_size}x{canvas_size} training on NVIDIA {gpu}...")
        sigma_arg = f"    '--sigma', '{sigma}',\n" if sigma is not None else ""
        train_script = (
            "import subprocess, sys, os\n"
            "os.makedirs('/content/target_scorer/weights', exist_ok=True)\n"
            f"cmd = [\n"
            f"    sys.executable, '-u', '-m', 'target_scorer.train',\n"
            f"    '--canvas-size', '{canvas_size}',\n"
            f"    '--scale', '{pixels_per_mm}',\n"
            f"    '--epochs', '{epochs}',\n"
            f"    '--batch-size', '{batch_size}',\n"
            f"    '--train-samples', '{train_samples_per_epoch}',\n"
            f"    '--val-samples', '{val_samples_per_epoch}',\n"
            f"    '--lr', '{learning_rate}',\n"
            f"{sigma_arg}"
            f"    '--checkpoint', '/content/target_scorer/weights/target_scoring_hrnet_{canvas_size}_best.pt',\n"
            f"    '--amp',\n"
            f"]\n"
            "proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
            "for line in proc.stdout:\n"
            "    print(line, end='', flush=True)\n"
            "proc.wait()\n"
            "if proc.returncode != 0:\n"
            "    sys.exit(proc.returncode)\n"
            "# Generate scorecard\n"
            f"infer_cmd = [\n"
            f"    sys.executable, '-u', '-m', 'target_scorer.infer',\n"
            f"    '--checkpoint', '/content/target_scorer/weights/target_scoring_hrnet_{canvas_size}_best.pt',\n"
            f"    '--canvas-size', '{canvas_size}',\n"
            f"    '--scale', '{pixels_per_mm}',\n"
            f"    '--threshold', '0.35',\n"
            f"    '--output', '/content/target_scorer/colab_inspection_{canvas_size}.png',\n"
            f"]\n"
            "infer_proc = subprocess.Popen(infer_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
            "for line in infer_proc.stdout:\n"
            "    print(line, end='', flush=True)\n"
            "infer_proc.wait()\n"
        )
        proc_train = subprocess.Popen(
            [colab_bin, "exec", "-s", session_name, "--timeout", str(timeout_seconds)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        proc_train.stdin.write(train_script)
        proc_train.stdin.close()
        for line in proc_train.stdout:
            print(line, end='', flush=True)
        proc_train.wait()
        if proc_train.returncode != 0:
            raise RuntimeError(f"Training failed on Colab VM with code {proc_train.returncode}.")

        # 6. Retrieve weights and visual scorecard
        print(f"[6/6] Downloading optimal model weights to '{checkpoint_dest}'...")
        remote_ckpt = f"/content/target_scorer/weights/target_scoring_hrnet_{canvas_size}_best.pt"
        cmd_dl_ckpt = [colab_bin, "download", remote_ckpt, checkpoint_dest, "-s", session_name]
        subprocess.run(cmd_dl_ckpt, check=True)
        print(f"[✓] Model checkpoint saved locally ({os.path.getsize(checkpoint_dest) / (1024**2):.2f} MB).")

        remote_png = f"/content/target_scorer/colab_inspection_{canvas_size}.png"
        try:
            cmd_dl_png = [colab_bin, "download", remote_png, scorecard_dest, "-s", session_name]
            subprocess.run(cmd_dl_png, check=True)
            print(f"[✓] Visual inspection scorecard saved locally to '{scorecard_dest}'.")
        except Exception:
            print("[!] Note: Scorecard image download skipped or not found.")

    finally:
        # Always terminate the Colab session to avoid burning compute units
        if session_created:
            print(f"[*] Releasing Colab VM session '{session_name}'...")
            subprocess.run([colab_bin, "stop", "-s", session_name])
            print("[✓] Colab session terminated successfully.")

    print("=" * 72)
    print(f"[✓] Cloud training complete! Checkpoint: {checkpoint_dest}")
    print("=" * 72)
    return checkpoint_dest


def main():
    parser = argparse.ArgumentParser(description="Orchestrate High-Resolution Training on Google Colab CLI (A100 / T4)")
    parser.add_argument("--canvas-size", type=int, default=800, choices=[512, 800, 1024], help="Target canvas resolution (800 or 1024 recommended)")
    parser.add_argument("--scale", type=float, default=4.0, help="Pixels per mm metric scale (default: 4.0)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for A100 (default: 8)")
    parser.add_argument("--train-samples", type=int, default=400, help="Procedural training samples per epoch")
    parser.add_argument("--val-samples", type=int, default=50, help="Validation samples per epoch")
    parser.add_argument("--lr", type=float, default=3.5e-4, help="Learning rate (default: 3.5e-4)")
    parser.add_argument("--gpu", type=str, default="A100", choices=["A100", "T4", "L4"], help="Cloud GPU type (default: A100)")
    parser.add_argument("--session", type=str, default=None, help="Custom Colab session name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Local destination path for trained checkpoint")
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian heatmap sigma in pixels (e.g. 3.2)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Maximum execution timeout in seconds")

    args = parser.parse_args()

    run_colab_training(
        canvas_size=args.canvas_size,
        pixels_per_mm=args.scale,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_samples_per_epoch=args.train_samples,
        val_samples_per_epoch=args.val_samples,
        learning_rate=args.lr,
        gpu=args.gpu,
        session_name=args.session,
        checkpoint_dest=args.checkpoint,
        timeout_seconds=args.timeout,
        sigma=args.sigma,
    )


if __name__ == "__main__":
    main()
