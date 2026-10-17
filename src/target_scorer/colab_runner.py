"""
Automated Google Colab CLI cloud orchestrator for high-resolution models and active learning.
Provisions cloud GPUs (NVIDIA A100 / H100 / T4 / L4) via Colab CLI, syncs codebase, executes
mixed-precision training and assisted auto-labeling, retrieves weights and dual scorecards,
and terminates cloud sessions safely to conserve compute units.
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


def check_colab_usage(colab_bin: str) -> None:
    """Queries and displays current Colab compute unit balance and usage rate."""
    try:
        out = subprocess.check_output([colab_bin, "usage"], text=True, stderr=subprocess.STDOUT)
        lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
        for line in lines:
            print(f"  [Colab Account] {line}")
    except Exception as e:
        print(f"  [!] Note: Could not query compute unit usage: {e}")


def get_session_url(colab_bin: str, session_name: str) -> Optional[str]:
    """Retrieves direct browser link to connect to the running Colab VM."""
    try:
        out = subprocess.check_output([colab_bin, "url", "-s", session_name], text=True, stderr=subprocess.STDOUT)
        for line in out.splitlines():
            line_s = line.strip()
            if line_s.startswith("https://colab.research.google.com"):
                return line_s
    except Exception:
        pass
    return None


def list_colab_sessions(colab_bin: str) -> None:
    """Lists all active Colab sessions."""
    try:
        out = subprocess.check_output([colab_bin, "sessions"], text=True, stderr=subprocess.STDOUT)
        print("=" * 60)
        print("  ACTIVE GOOGLE COLAB SESSIONS")
        print("=" * 60)
        print(out.strip())
        print("=" * 60)
    except Exception as e:
        print(f"[-] Error retrieving Colab sessions: {e}")


def stop_colab_session(colab_bin: str, session_name: str) -> None:
    """Terminates a specific Colab session."""
    print(f"[*] Releasing Colab VM session '{session_name}'...")
    res = subprocess.run([colab_bin, "stop", "-s", session_name])
    if res.returncode == 0:
        print(f"[✓] Session '{session_name}' terminated successfully.")
    else:
        print(f"[-] Failed to terminate session '{session_name}'.")


def run_colab_training(
    canvas_size: int = 800,
    pixels_per_mm: float = 4.0,
    epochs: int = 15,
    batch_size: int = 8,
    train_samples_per_epoch: int = 400,
    val_samples_per_epoch: int = 50,
    learning_rate: float = 3.5e-4,
    gpu: str = "A100",
    high_mem: bool = False,
    session_name: Optional[str] = None,
    checkpoint_dest: Optional[str] = None,
    scorecard_b2_dest: Optional[str] = None,
    scorecard_b3_dest: Optional[str] = None,
    timeout_seconds: float = 3600.0,
    sigma: Optional[float] = None,
    initial_weights: Optional[str] = None,
    real_data_dir: Optional[str] = None,
    real_ratio: float = 0.35,
) -> str:
    """
    End-to-end cloud training orchestration using Google Colab CLI:
      1. Displays compute unit balance and usage rate.
      2. Repackages latest codebase into zip bundle.
      3. Provisions cloud GPU session (NVIDIA A100 / H100 / T4 / L4).
      4. Prints interactive browser URL for live monitoring.
      5. Uploads bundle, initial weights, and real target dataset.
      6. Sets up remote environment and executes mixed-precision (AMP) training.
      7. Evaluates dual NRA competition scorecards (B-2 Slow Fire & B-3 Timed/Rapid).
      8. Downloads optimal checkpoint and both scorecards to local machine.
      9. Automatically terminates session to conserve compute units.
    """
    colab_bin = get_colab_bin()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if session_name is None:
        session_name = f"target-train-{canvas_size}-{int(time.time()) % 100000:05d}"

    if checkpoint_dest is None:
        checkpoint_dest = os.path.join(project_root, "weights", f"target_scoring_hrnet_{canvas_size}_best.pt")

    if scorecard_b2_dest is None:
        scorecard_b2_dest = os.path.join(project_root, "visual_inspection_reports", f"colab_scorecard_{canvas_size}_b2.png")

    if scorecard_b3_dest is None:
        scorecard_b3_dest = os.path.join(project_root, "visual_inspection_reports", f"colab_scorecard_{canvas_size}_b3.png")

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_dest)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(scorecard_b2_dest)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(scorecard_b3_dest)), exist_ok=True)

    print("=" * 72)
    print("  GOOGLE COLAB CLI CLOUD TRAINING ORCHESTRATION")
    print(f"  Canvas Dimensions:   {canvas_size} x {canvas_size} px")
    print(f"  Metrology Scale:     {pixels_per_mm} pixels/mm")
    print(f"  Target Accelerator:  NVIDIA {gpu} {'(High-RAM)' if high_mem else ''}")
    print(f"  Colab Session Name:  {session_name}")
    print(f"  Local Output Ckpt:   {checkpoint_dest}")
    if real_data_dir:
        print(f"  Real Target Dataset: {real_data_dir} (Ratio: {real_ratio:.2f})")
    check_colab_usage(colab_bin)
    print("=" * 72)

    # 1. Repackage latest codebase bundle
    print("[1/7] Packaging latest codebase bundle...")
    bundle_script = os.path.join(project_root, "scripts", "create_colab_bundle.py")
    bundle_zip = os.path.join(project_root, "target_scorer_colab_bundle.zip")
    if os.path.exists(bundle_script):
        subprocess.run([sys.executable, bundle_script], cwd=project_root, check=True)
    elif not os.path.exists(bundle_zip):
        raise FileNotFoundError(f"Colab bundle '{bundle_zip}' not found and packaging script missing.")

    session_created = False
    try:
        # 2. Provision cloud GPU VM
        print(f"[2/7] Provisioning Colab VM session '{session_name}' with {gpu} accelerator...")
        cmd_new = [colab_bin, "new", "-s", session_name, "--gpu", gpu]
        if high_mem:
            cmd_new.append("--high-mem")
        res_new = subprocess.run(cmd_new)
        if res_new.returncode != 0:
            raise RuntimeError(f"Failed to allocate Colab session '{session_name}' with GPU {gpu}.")
        session_created = True

        # Print browser URL for live monitoring
        session_url = get_session_url(colab_bin, session_name)
        if session_url:
            print(f"  [✓] Interactive Browser URL: {session_url}")

        # 3. Upload codebase bundle & assets
        print(f"[3/7] Uploading codebase archive ({os.path.getsize(bundle_zip) / (1024**2):.1f} MB) to Colab VM...")
        cmd_upload = [colab_bin, "upload", bundle_zip, "/content/target_scorer_colab_bundle.zip", "-s", session_name]
        subprocess.run(cmd_upload, check=True)

        if initial_weights is not None and os.path.exists(initial_weights):
            print(f"[*] Uploading initial weights ({os.path.getsize(initial_weights) / (1024**2):.1f} MB) to Colab VM...")
            cmd_upload_w = [colab_bin, "upload", initial_weights, "/content/initial_weights.pt", "-s", session_name]
            subprocess.run(cmd_upload_w, check=True)

        if real_data_dir is not None and os.path.exists(real_data_dir):
            real_data_zip = os.path.join(project_root, "real_data_bundle.zip")
            if os.path.exists(real_data_zip):
                os.remove(real_data_zip)
            print(f"[*] Packaging real target dataset from '{real_data_dir}'...")
            shutil.make_archive(os.path.splitext(real_data_zip)[0], "zip", real_data_dir)
            try:
                print(f"[*] Uploading real dataset archive ({os.path.getsize(real_data_zip) / (1024**2):.1f} MB) to Colab VM...")
                cmd_upload_data = [colab_bin, "upload", real_data_zip, "/content/real_data.zip", "-s", session_name]
                subprocess.run(cmd_upload_data, check=True)
            finally:
                if os.path.exists(real_data_zip):
                    os.remove(real_data_zip)

        # 4. Unpack and install on remote VM
        print("[4/7] Initializing remote environment on Colab VM...")
        setup_script = (
            "import subprocess, sys, os\n"
            "subprocess.run(['unzip', '-q', '-o', '/content/target_scorer_colab_bundle.zip', '-d', '/content/'], check=True)\n"
            "if os.path.exists('/content/real_data.zip'):\n"
            "    subprocess.run(['unzip', '-q', '-o', '/content/real_data.zip', '-d', '/content/real_data'], check=True)\n"
            "    print('[Colab VM] Real target dataset extracted.')\n"
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
        print(f"[5/7] Executing {canvas_size}x{canvas_size} training on NVIDIA {gpu}...")
        sigma_arg = f"    '--sigma', '{sigma}',\n" if sigma is not None else ""
        init_w_arg = "    '--initial-weights', '/content/initial_weights.pt',\n" if (initial_weights and os.path.exists(initial_weights)) else ""
        real_arg = f"    '--real-data-dir', '/content/real_data',\n    '--real-ratio', '{real_ratio}',\n" if (real_data_dir and os.path.exists(real_data_dir)) else ""
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
            f"{init_w_arg}"
            f"{real_arg}"
            f"    '--checkpoint', '/content/target_scorer/weights/target_scoring_hrnet_{canvas_size}_best.pt',\n"
            f"    '--amp',\n"
            f"]\n"
            "proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
            "for line in proc.stdout:\n"
            "    print(line, end='', flush=True)\n"
            "proc.wait()\n"
            "if proc.returncode != 0:\n"
            "    sys.exit(proc.returncode)\n"
            "# Generate B-2 scorecard (Slow Fire)\n"
            f"infer_b2 = [\n"
            f"    sys.executable, '-u', '-m', 'target_scorer.infer',\n"
            f"    '--checkpoint', '/content/target_scorer/weights/target_scoring_hrnet_{canvas_size}_best.pt',\n"
            f"    '--canvas-size', '{canvas_size}',\n"
            f"    '--scale', '{pixels_per_mm}',\n"
            f"    '--threshold', '0.35',\n"
            f"    '--target', 'B-2',\n"
            f"    '--output', '/content/target_scorer/colab_inspection_{canvas_size}_b2.png',\n"
            f"]\n"
            "p_b2 = subprocess.Popen(infer_b2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
            "for line in p_b2.stdout: print(line, end='', flush=True)\n"
            "p_b2.wait()\n"
            "# Generate B-3 scorecard (Timed & Rapid Fire)\n"
            f"infer_b3 = [\n"
            f"    sys.executable, '-u', '-m', 'target_scorer.infer',\n"
            f"    '--checkpoint', '/content/target_scorer/weights/target_scoring_hrnet_{canvas_size}_best.pt',\n"
            f"    '--canvas-size', '{canvas_size}',\n"
            f"    '--scale', '{pixels_per_mm}',\n"
            f"    '--threshold', '0.35',\n"
            f"    '--target', 'B-3',\n"
            f"    '--output', '/content/target_scorer/colab_inspection_{canvas_size}_b3.png',\n"
            f"]\n"
            "p_b3 = subprocess.Popen(infer_b3, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
            "for line in p_b3.stdout: print(line, end='', flush=True)\n"
            "p_b3.wait()\n"
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

        # 6. Retrieve weights and visual scorecards
        print(f"[6/7] Downloading optimal model weights to '{checkpoint_dest}'...")
        remote_ckpt = f"/content/target_scorer/weights/target_scoring_hrnet_{canvas_size}_best.pt"
        cmd_dl_ckpt = [colab_bin, "download", remote_ckpt, checkpoint_dest, "-s", session_name]
        subprocess.run(cmd_dl_ckpt, check=True)
        print(f"[✓] Model checkpoint saved locally ({os.path.getsize(checkpoint_dest) / (1024**2):.2f} MB).")

        # Download B-2 scorecard
        remote_b2 = f"/content/target_scorer/colab_inspection_{canvas_size}_b2.png"
        try:
            cmd_dl_b2 = [colab_bin, "download", remote_b2, scorecard_b2_dest, "-s", session_name]
            subprocess.run(cmd_dl_b2, check=True)
            print(f"[✓] NRA B-2 visual scorecard saved locally to '{scorecard_b2_dest}'.")
        except Exception:
            print("[!] Note: NRA B-2 scorecard image download skipped or not found.")

        # Download B-3 scorecard
        remote_b3 = f"/content/target_scorer/colab_inspection_{canvas_size}_b3.png"
        try:
            cmd_dl_b3 = [colab_bin, "download", remote_b3, scorecard_b3_dest, "-s", session_name]
            subprocess.run(cmd_dl_b3, check=True)
            print(f"[✓] NRA B-3 visual scorecard saved locally to '{scorecard_b3_dest}'.")
        except Exception:
            print("[!] Note: NRA B-3 scorecard image download skipped or not found.")

    finally:
        # 7. Always terminate the Colab session to avoid burning compute units
        if session_created:
            print(f"[7/7] Releasing Colab VM session '{session_name}'...")
            subprocess.run([colab_bin, "stop", "-s", session_name])
            print("[✓] Colab session terminated successfully.")

    print("=" * 72)
    print(f"[✓] Cloud training complete! Checkpoint: {checkpoint_dest}")
    print("=" * 72)
    return checkpoint_dest


def run_colab_autolabel(
    input_dir: str,
    output_dir: str = "datasets/autolabeled_targets",
    target_type: str = "B-2",
    checkpoint: Optional[str] = "weights/target_scoring_hrnet_1024_best.pt",
    canvas_size: int = 1024,
    pixels_per_mm: float = 4.0,
    threshold: float = 0.40,
    gpu: str = "T4",
    session_name: Optional[str] = None,
    timeout_seconds: float = 1800.0,
    redact_pii: bool = False,
) -> str:
    """
    Executes assisted auto-labeling on an entire directory of smartphone photographs
    using an accelerated Google Colab cloud GPU session (default: T4).
    """
    colab_bin = get_colab_bin()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' not found.")

    if session_name is None:
        session_name = f"target-autolabel-{int(time.time()) % 100000:05d}"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 72)
    print("  GOOGLE COLAB CLI CLOUD AUTO-LABELING ORCHESTRATION")
    print(f"  Input Directory:     {input_dir}")
    print(f"  Target Type:         NRA {target_type.upper()}")
    print(f"  Target Accelerator:  NVIDIA {gpu}")
    print(f"  Colab Session Name:  {session_name}")
    print(f"  Local Output Dir:    {output_dir}")
    check_colab_usage(colab_bin)
    print("=" * 72)

    # 1. Package codebase and input images
    bundle_script = os.path.join(project_root, "scripts", "create_colab_bundle.py")
    bundle_zip = os.path.join(project_root, "target_scorer_colab_bundle.zip")
    if os.path.exists(bundle_script):
        subprocess.run([sys.executable, bundle_script], cwd=project_root, check=True)

    input_zip = os.path.join(project_root, "autolabel_inputs.zip")
    if os.path.exists(input_zip):
        os.remove(input_zip)
    print(f"[1/5] Packaging input images from '{input_dir}'...")
    shutil.make_archive(os.path.splitext(input_zip)[0], "zip", input_dir)

    session_created = False
    try:
        # 2. Provision VM
        print(f"[2/5] Provisioning Colab VM '{session_name}' with {gpu} accelerator...")
        cmd_new = [colab_bin, "new", "-s", session_name, "--gpu", gpu]
        res_new = subprocess.run(cmd_new)
        if res_new.returncode != 0:
            raise RuntimeError(f"Failed to allocate Colab session '{session_name}'.")
        session_created = True

        session_url = get_session_url(colab_bin, session_name)
        if session_url:
            print(f"  [✓] Interactive Browser URL: {session_url}")

        # 3. Upload codebase, model weights, and inputs
        print("[3/5] Uploading codebase archive and input images...")
        subprocess.run([colab_bin, "upload", bundle_zip, "/content/target_scorer_colab_bundle.zip", "-s", session_name], check=True)
        try:
            subprocess.run([colab_bin, "upload", input_zip, "/content/autolabel_inputs.zip", "-s", session_name], check=True)
        finally:
            if os.path.exists(input_zip):
                os.remove(input_zip)

        if checkpoint and os.path.exists(checkpoint):
            print(f"[*] Uploading model checkpoint '{checkpoint}'...")
            subprocess.run([colab_bin, "upload", checkpoint, "/content/model_weights.pt", "-s", session_name], check=True)
            ckpt_remote_arg = "'--checkpoint', '/content/model_weights.pt',"
        else:
            ckpt_remote_arg = ""

        redact_remote_arg = "    '--redact-pii',\n" if redact_pii else ""

        # 4. Setup and run auto-labeling
        print("[4/5] Running assisted auto-labeling on cloud GPU...")
        autolabel_script = (
            "import subprocess, sys, shutil, os\n"
            "subprocess.run(['unzip', '-q', '-o', '/content/target_scorer_colab_bundle.zip', '-d', '/content/'], check=True)\n"
            "os.makedirs('/content/input_photos', exist_ok=True)\n"
            "os.makedirs('/content/autolabeled_output', exist_ok=True)\n"
            "subprocess.run(['unzip', '-q', '-o', '/content/autolabel_inputs.zip', '-d', '/content/input_photos'], check=True)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '/content/target_scorer'], check=True)\n"
            f"cmd = [\n"
            f"    sys.executable, '-u', '-m', 'target_scorer.cli', 'autolabel',\n"
            f"    '--input-dir', '/content/input_photos',\n"
            f"    '--output-dir', '/content/autolabeled_output',\n"
            f"    '--target', '{target_type}',\n"
            f"    '--canvas-size', '{canvas_size}',\n"
            f"    '--scale', '{pixels_per_mm}',\n"
            f"    '--threshold', '{threshold}',\n"
            f"{redact_remote_arg}"
            f"    {ckpt_remote_arg}\n"
            f"]\n"
            "proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
            "for line in proc.stdout: print(line, end='', flush=True)\n"
            "proc.wait()\n"
            "if proc.returncode != 0:\n"
            "    sys.exit(proc.returncode)\n"
            "shutil.make_archive('/content/autolabeled_results', 'zip', '/content/autolabeled_output')\n"
        )
        proc_exec = subprocess.Popen(
            [colab_bin, "exec", "-s", session_name, "--timeout", str(timeout_seconds)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        proc_exec.stdin.write(autolabel_script)
        proc_exec.stdin.close()
        for line in proc_exec.stdout:
            print(line, end='', flush=True)
        proc_exec.wait()
        if proc_exec.returncode != 0:
            raise RuntimeError(f"Cloud auto-labeling failed with code {proc_exec.returncode}.")

        # 5. Download results
        print(f"[5/5] Downloading auto-labeled annotations and review cards to '{output_dir}'...")
        local_results_zip = os.path.join(project_root, "autolabeled_results.zip")
        subprocess.run([colab_bin, "download", "/content/autolabeled_results.zip", local_results_zip, "-s", session_name], check=True)
        shutil.unpack_archive(local_results_zip, output_dir)
        if os.path.exists(local_results_zip):
            os.remove(local_results_zip)
        print(f"[✓] Extracted results into '{output_dir}'.")

    finally:
        if session_created:
            print(f"[*] Releasing Colab VM session '{session_name}'...")
            subprocess.run([colab_bin, "stop", "-s", session_name])
            print("[✓] Colab session terminated successfully.")

    print("=" * 72)
    print(f"[✓] Cloud auto-labeling complete! Output directory: {output_dir}")
    print("=" * 72)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Orchestrate High-Resolution Training & Auto-Labeling on Google Colab CLI")
    parser.add_argument("--canvas-size", type=int, default=800, choices=[256, 512, 800, 1024], help="Target canvas resolution (800 or 1024 recommended)")
    parser.add_argument("--scale", type=float, default=4.0, help="Pixels per mm metric scale (default: 4.0)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for cloud GPU (default: 8)")
    parser.add_argument("--train-samples", type=int, default=400, help="Procedural training samples per epoch")
    parser.add_argument("--val-samples", type=int, default=50, help="Validation samples per epoch")
    parser.add_argument("--lr", type=float, default=3.5e-4, help="Learning rate (default: 3.5e-4)")
    parser.add_argument("--gpu", type=str, default="A100", choices=["A100", "H100", "T4", "L4", "G4"], help="Cloud GPU type (default: A100)")
    parser.add_argument("--high-mem", action="store_true", default=False, help="Request high-memory instance shape")
    parser.add_argument("--session", type=str, default=None, help="Custom Colab session name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Local destination path for trained checkpoint")
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian heatmap sigma in pixels (e.g. 3.2)")
    parser.add_argument("--timeout", type=float, default=3600.0, help="Maximum execution timeout in seconds")
    parser.add_argument("--initial-weights", type=str, default=None, help="Path to local weights checkpoint to warm-start/fine-tune from in Colab")
    parser.add_argument("--real-data-dir", type=str, default=None, help="Path to local real target dataset directory to upload for training")
    parser.add_argument("--real-ratio", type=float, default=0.35, help="Sampling probability for real target images (1.0 = pure real fine-tuning, 0.35 = hybrid)")
    parser.add_argument("--list-sessions", action="store_true", default=False, help="List active Colab sessions and exit")
    parser.add_argument("--stop-session", type=str, default=None, help="Stop an active Colab session and exit")

    args = parser.parse_args()
    colab_bin = get_colab_bin()

    if args.list_sessions:
        list_colab_sessions(colab_bin)
        return

    if args.stop_session:
        stop_colab_session(colab_bin, args.stop_session)
        return

    run_colab_training(
        canvas_size=args.canvas_size,
        pixels_per_mm=args.scale,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_samples_per_epoch=args.train_samples,
        val_samples_per_epoch=args.val_samples,
        learning_rate=args.lr,
        gpu=args.gpu,
        high_mem=args.high_mem,
        session_name=args.session,
        checkpoint_dest=args.checkpoint,
        timeout_seconds=args.timeout,
        sigma=args.sigma,
        initial_weights=args.initial_weights,
        real_data_dir=args.real_data_dir,
        real_ratio=args.real_ratio,
    )


if __name__ == "__main__":
    main()
