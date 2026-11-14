# Command-Line & Python API Reference

This reference provides exhaustive documentation for all command-line interface (CLI) subcommands and programmatic Python APIs in the [`target_scorer`](file:///home/antho/target_scorer) package.

All CLI commands are accessible through the unified executable:
```bash
target-scorer <subcommand> [options]
```

---

## Table of Contents
1. [`target-scorer infer`](#1-target-scorer-infer)
2. [`target-scorer train`](#2-target-scorer-train)
3. [`target-scorer colab-train`](#3-target-scorer-colab-train)
4. [`target-scorer colab-progressive-train`](#4-target-scorer-colab-progressive-train)
5. [`target-scorer colab-autolabel`](#5-target-scorer-colab-autolabel)
6. [`target-scorer rectify`](#6-target-scorer-rectify)
7. [`target-scorer autolabel`](#7-target-scorer-autolabel)
8. [`target-scorer annotate`](#8-target-scorer-annotate)
9. [Python API Reference](#9-python-api-reference)

---

## 1. `target-scorer infer`

Scores an input target photograph or procedurally generates and scores a synthetic competition demo card. Executes automated perspective unwarping, deep learning heatmap regression, sub-pixel peak extraction, NRA inward break scoring, ballistic dispersion analysis, and visual scorecard rendering.

Source: [`src/target_scorer/infer.py`](file:///home/antho/target_scorer/src/target_scorer/infer.py)

### CLI Synopsis
```bash
target-scorer infer [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--image` | `str` | `None` | Path to target photograph, or `-` to stream raw bytes from `stdin`. If omitted, generates a deterministic synthetic demo target. |
| `--target` | `str` | `B-2` | NRA target type. Choices: `B-2` (50 ft Slow Fire) or `B-3` (50 ft Timed/Rapid Fire). Case-insensitive. |
| `--checkpoint` | `str` | `weights/target_scoring_hrnet_1024_best.pt` | Path to PyTorch model weights checkpoint (`.pt`). Searches `weights/` directory if relative path not found in cwd. |
| `--output` | `str` | `scored_inspection_result.png` | Destination filepath for the high-resolution annotated scorecard PNG. |
| `--debug-rect` | `str` | `rectification_debug.png` | Destination filepath for the perspective rectification diagnostic side-by-side visualization. Set to empty string `""` to disable. |
| `--distance` | `float` | `16.67` | Shooting distance to target in yards ($50\text{ ft} = 16.667\text{ yd}$). Used to convert millimeter dispersion to Minutes of Angle (MOA). |
| `--shots` | `int` | `10` | Maximum number of shots to evaluate. Standard NRA string length is `10`. |
| `--threshold` | `float` | `0.45` | Heatmap peak detection confidence threshold $[0.0, 1.0]$. Lower values ($0.30 - 0.40$) detect faint tears; higher values ($0.50 - 0.60$) reject cardboard debris. |
| `--device` | `str` | `None` (auto) | Compute device: `cuda`, `cpu`, or `mps`. If `None`, detects best available accelerator. |
| `--canvas-size` | `int` | `1024` | Override canonical canvas dimension in pixels (`256`, `512`, `768`, or `1024`). Defaults to production standard 1024. |
| `--scale` | `float` | `4.0` | Override metric resolution in pixels per millimeter ($4.0\text{ px/mm}$ for 1024). Defaults to production standard 4.0. |
| `--redact-pii` | flag | `False` | Masks competitor registration handwriting, relay numbers, and shooter names on margins with circular cardboard masking. |

### Copy-Pasteable Shell Examples

#### A. Score a Real Smartphone Target Photo (HEIC, RAW/DNG, WebP, PNG, JPEG)
```bash
target-scorer infer \
  --image sample_targets/iphone_photo.heic \
  --target B-3 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --canvas-size 1024 \
  --scale 4.0 \
  --shots 10 \
  --output visual_inspection_reports/scored_iphone_b3.png \
  --debug-rect visual_inspection_reports/rect_debug_b3.png
```

#### B. Score with Competitor Registration PII Redaction
```bash
target-scorer infer \
  --image range_data/match_relay_4.jpg \
  --target B-2 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --canvas-size 1024 \
  --scale 4.0 \
  --redact-pii \
  --output visual_inspection_reports/scored_match_redacted.png
```

#### C. Stream Image from `curl` or Standard Input Pipe (`-`)
```bash
curl -s https://example.com/target_relay.png | target-scorer infer \
  --image - \
  --target B-2 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --canvas-size 1024 \
  --scale 4.0 \
  --output scored_streamed.png
```

#### D. Generate and Score a Synthetic Competition Demo Card
```bash
# NRA B-2 Slow Fire (Scores 96 / 100)
target-scorer infer \
  --target B-2 \
  --canvas-size 1024 \
  --scale 4.0 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --output scored_demo_b2.png

# NRA B-3 Timed/Rapid Fire (Scores 100 - 7X / 100)
target-scorer infer \
  --target B-3 \
  --canvas-size 1024 \
  --scale 4.0 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --output scored_demo_b3.png
```

---

## 2. `target-scorer train`

Trains the [`HighResHeatmapUNet`](file:///home/antho/target_scorer/src/target_scorer/models/unet.py) model locally. Supports standard single-resolution training as well as the unified **Harmonic 256-Grid Progressive Resizing** curriculum ($256 \to 512 \to 768 \to 1024$). Supports pure procedural synthetic generation, pure real target fine-tuning (Option B), or blended hybrid training (Option A).

Source: [`src/target_scorer/train.py`](file:///home/antho/target_scorer/src/target_scorer/train.py)

### CLI Synopsis
```bash
target-scorer train [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--progressive` | flag | `False` | Run multi-stage Progressive Resizing curriculum across the Harmonic 256-Grid ($256 \to 512 \to 768 \to 1024$). |
| `--progressive-stages`| `str` | `None` | Custom comma-separated progressive stage schedule string (e.g. `"256:1.0:1.6:16:6:4.5e-4:500:50,512:2.0:2.2:12:5:3.0e-4:400:50"`). |
| `--checkpoint-dir` | `str` | `weights` | Directory to save intermediate stage checkpoints (`target_scoring_hrnet_{canvas}_best.pt`). |
| `--canvas-size` | `int` | `1024` | Canonical canvas resolution in pixels (for single-resolution mode). Must be divisible by 8. |
| `--scale` | `float` | `4.0` | Metric scale in pixels per millimeter ($4.0\text{ px/mm}$ for 1024). |
| `--epochs` | `int` | `15` | Total number of training epochs (in single-resolution mode). |
| `--batch-size` | `int` | `8` | Mini-batch size. |
| `--lr` | `float` | `3.5e-4` | Initial learning rate for AdamW optimizer. Follows Cosine Annealing decay. |
| `--train-samples`| `int` | `400` | Number of training samples per epoch. |
| `--val-samples` | `int` | `50` | Number of validation samples per epoch. |
| `--checkpoint` | `str` | `weights/target_scoring_hrnet_1024_best.pt` | Output filepath for saving the final production model weights. |
| `--workers` | `int` | `2` | Number of CPU worker subprocesses for PyTorch `DataLoader`. |
| `--sigma` | `float` | `None` (auto) | Heatmap Gaussian kernel standard deviation ($3.2\text{ px}$ for 1024). |
| `--amp` / `--no-amp` | flag | `--amp` (True) | Enables/disables PyTorch Automatic Mixed Precision (`torch.amp.autocast`). |
| `--initial-weights`| `str` | `None` | Path to existing `.pt` weights to warm-start or fine-tune from. |
| `--real-data-dir`| `str` | `None` | Path to directory containing real target annotations (`*.json`) and images to enable hybrid or real training. |
| `--real-ratio` | `float` | `0.35` | Sampling probability for real target images. `1.0` = pure real (Option B); `0.35` = hybrid 70% synth / 30% real (Option A); `0.0` = pure synthetic. |

### Copy-Pasteable Shell Examples

#### A. Unified Progressive Resizing Curriculum (Harmonic 256-Grid)
```bash
target-scorer train \
  --progressive \
  --real-data-dir private/data/real_targets \
  --real-ratio 0.35 \
  --checkpoint-dir weights \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt
```

#### B. Warm-Start Fine-Tuning Locally ($1024 \times 1024$ on GTX 1060 / RTX 3060)
```bash
target-scorer train \
  --canvas-size 1024 \
  --scale 4.0 \
  --epochs 12 \
  --batch-size 2 \
  --lr 1.5e-4 \
  --sigma 3.2 \
  --initial-weights weights/target_scoring_hrnet_1024_best.pt \
  --checkpoint weights/target_scoring_hrnet_1024_finetuned.pt
```

#### C. Pure Real Target Fine-Tuning (Option B: Rapid Adaptation)
```bash
target-scorer train \
  --canvas-size 1024 \
  --scale 4.0 \
  --epochs 10 \
  --batch-size 2 \
  --lr 5.0e-5 \
  --sigma 3.2 \
  --real-data-dir private/data/real_targets \
  --real-ratio 1.0 \
  --initial-weights weights/target_scoring_hrnet_1024_best.pt \
  --checkpoint weights/target_scoring_hrnet_1024_real_adapted.pt
```

---

## 3. `target-scorer colab-train`

Orchestrates automated single-resolution cloud GPU training using Google Colab CLI. Provisions cloud accelerators (NVIDIA A100, H100, T4, L4, or G4), syncs the repository bundle, executes mixed-precision training, verifies B-2 and B-3 competition scorecards, downloads the best checkpoint locally, and tears down the VM to conserve compute units.

Source: [`src/target_scorer/colab_runner.py`](file:///home/antho/target_scorer/src/target_scorer/colab_runner.py)

### Optional Google Colab CLI Setup
The `colab-train`, `colab-progressive-train`, and `colab-autolabel` commands interact with Google Colab via the standalone `colab` executable. If not already installed:

```bash
# Install via uv (recommended) or pip
uv tool install google-colab-cli
# Or: pip install google-colab-cli

# Authenticate with your Google account
colab auth

# Verify compute unit quota
colab usage
```
*(The Target Scorer orchestrator automatically resolves `colab` from `PATH` or `~/.local/bin/colab`.)*

### CLI Synopsis
```bash
target-scorer colab-train [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--canvas-size` | `int` | `1024` | Resolution preset: `256`, `512`, `768`, or `1024` (1024 production standard). |
| `--scale` | `float` | `4.0` | Pixels per millimeter scale. |
| `--epochs` | `int` | `15` | Total training epochs. |
| `--batch-size` | `int` | `8` | Mini-batch size on cloud GPU (`8` recommended for A100). |
| `--train-samples`| `int` | `400` | Training samples per epoch. |
| `--val-samples` | `int` | `50` | Validation samples per epoch. |
| `--lr` | `float` | `3.5e-4` | Initial learning rate. |
| `--gpu` | `str` | `A100` | Cloud GPU type: `A100`, `H100`, `T4`, `L4`, or `G4`. |
| `--high-mem` | flag | `False` | Request high-RAM cloud instance shape. |
| `--session` | `str` | `None` (auto) | Custom Colab session name. |
| `--checkpoint` | `str` | `None` (auto) | Local destination path for trained weights. Defaults to `weights/target_scoring_hrnet_{canvas}_best.pt`. |
| `--sigma` | `float` | `None` (auto) | Heatmap Gaussian kernel standard deviation ($3.2$ for 1024). |
| `--timeout` | `float` | `3600.0` | Maximum job execution timeout in seconds. |
| `--initial-weights`| `str` | `None` | Local `.pt` checkpoint to upload for warm-start fine-tuning in Colab. |
| `--real-data-dir`| `str` | `None` | Local directory containing real target annotations to package, upload, and blend during training. |
| `--real-ratio` | `float` | `0.35` | Real target sampling probability (`1.0` = pure real, `0.35` = hybrid). |
| `--list-sessions` | flag | `False` | Lists all active Colab cloud sessions and exits. |
| `--stop-session` | `str` | `None` | Terminates a specific running Colab session by name and exits. |

### Real Dataset Packaging & Chunked Streaming Transport

When `--real-data-dir` is provided, [`package_real_dataset_bundle()`](file:///home/antho/target_scorer/src/target_scorer/colab_runner.py#L65-L85) packages canonical rectified images (`*_canonical.png`) and annotations (`*.json`) while filtering out heavy raw camera formats (`.raw`, `.nef`, `.dng`) and review cards (`*_review.png`) to minimize bandwidth.

The bundle is then transmitted to the remote Colab VM via [`upload_file_chunked()`](file:///home/antho/target_scorer/src/target_scorer/colab_runner.py#L96-L167).

#### Why Built-in Chunked Streaming Was Chosen

For datasets in the **50 MB – 500 MB** range, chunked streaming via `colab upload` offers a key operational advantage:

1. **Zero External Dependencies:** Requires no GCP billing buckets, no Google Drive authorizations, and no SSH key configuration.
2. **1-Click Self-Contained Automation:** Running `./target_env/bin/python3 -m target_scorer.cli colab-train` packages the data, slices it, uploads it, trains the model, downloads the scorecards, and shuts down the instance with a single command.

If your dataset eventually scales beyond several gigabytes, we can readily wire in the `colab ssh --proxy-mode` (SCP) or GCS bucket transfer workflow!

#### Transport Mechanics & Failure Mode Mitigations

* **Why Standard Direct `scp` Cannot Reach Colab VMs:**
  Google Colab runtimes operate inside Google Container Engine sandbox containers behind Carrier-Grade NAT (CGNAT) and ingress firewalls. Colab VMs are not allocated public IP addresses, and direct inbound ports (including TCP port 22 for OpenSSH) are blocked and unroutable from the public internet. Direct `scp user@colab-ip:/path` cannot establish a socket connection.
* **How `colab ssh --proxy-mode` Works:**
  To support SSH connections, Google Colab provides an OpenSSH proxy tunnel. The Colab VM initiates an outbound duplex WebSocket connection to Google Cloud's command-line relay infrastructure. An OpenSSH client on the local machine connects via a `ProxyCommand` bridge through this WebSocket tunnel. While functional, it requires client-side SSH key management, OpenSSH client configuration, and active proxy bridging.
* **Why Chunked Streaming Avoids HTTP 500 / 413 Errors:**
  Colab CLI's standard `colab upload` communicates via HTTP POST to the Jupyter REST Contents API (`/api/contents/`). Uploading monolithic zip archives exceeding ~35 MB – 50 MB frequently saturates the Jupyter reverse-proxy buffer or exceeds payload limits, resulting in fatal `HTTP 500 Internal Server Error` or `HTTP 413 Payload Too Large` failures.
  [`upload_file_chunked`](file:///home/antho/target_scorer/src/target_scorer/colab_runner.py#L96-L167) slices large archives into deterministic binary chunks ($\le 35\text{ MB}$), uploads each part sequentially, and executes an automated remote Python reassembly script via `colab exec` to reconstruct the archive on the VM prior to unzipping.

---

## 4. `target-scorer colab-progressive-train`

Orchestrates the complete **4-stage Harmonic 256-Grid Progressive Resizing** curriculum ($256 \to 512 \to 768 \to 1024$) inside a **single cloud VM session** on Google Colab CLI. Trains all 4 stages sequentially, renders 8 diagnostic competition scorecards (both B-2 and B-3 for all 4 stages), downloads all intermediate weights and scorecards to your workstation, and automatically terminates the session.

Source: [`src/target_scorer/colab_runner.py`](file:///home/antho/target_scorer/src/target_scorer/colab_runner.py), [`scripts/train_progressive_colab.py`](file:///home/antho/target_scorer/scripts/train_progressive_colab.py)

### CLI Synopsis
```bash
target-scorer colab-progressive-train [OPTIONS]
```
*(Or invoke via the standalone runner script: `python scripts/train_progressive_colab.py [A100|T4]`)*

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--gpu` | `str` | `A100` | Cloud accelerator: `A100`, `H100`, `T4`, `L4`, or `G4`. |
| `--high-mem` | flag | `False` | Request high-RAM cloud instance shape. |
| `--session` | `str` | `None` (auto) | Custom Colab session name. |
| `--timeout` | `float` | `3600.0` | Maximum job execution timeout in seconds. |
| `--real-data-dir`| `str` | `None` | Local directory containing real target annotations to package and upload. |
| `--real-ratio` | `float` | `0.35` | Real target sampling probability for hybrid stages. |
| `--initial-weights`| `str` | `None` | Optional initial weights to seed Stage 1. |

### Copy-Pasteable Shell Example
```bash
# Execute full 4-stage progressive cloud training on NVIDIA A100:
target-scorer colab-progressive-train \
  --gpu A100 \
  --real-data-dir private/data/real_targets \
  --real-ratio 0.35
```

---

## 5. `target-scorer colab-autolabel`

Executes cloud-accelerated assisted auto-labeling on an entire directory of target photographs using Google Colab CLI. Automatically packages images, uploads them to a remote GPU instance (default: NVIDIA T4), runs geometric rectification and inference, and downloads the generated JSON annotations and visual review cards.

Source: [`src/target_scorer/colab_runner.py`](file:///home/antho/target_scorer/src/target_scorer/colab_runner.py)

### CLI Synopsis
```bash
target-scorer colab-autolabel [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input-dir` | `str` | **Required** | Local directory containing target photographs (`.heic`, `.jpg`, `.png`, `.dng`, `.tiff`). |
| `--output-dir` | `str` | `datasets/autolabeled_targets` | Local directory to download annotations and review cards into. |
| `--target` | `str` | `B-2` | NRA target type: `B-2` or `B-3`. |
| `--checkpoint` | `str` | `weights/target_scoring_hrnet_1024_best.pt` | Path to local model checkpoint to upload to Colab. |
| `--canvas-size` | `int` | `1024` | Canonical canvas resolution. |
| `--scale` | `float` | `4.0` | Metric scale in pixels per millimeter. |
| `--threshold` | `float` | `0.40` | Peak detection confidence threshold. |
| `--gpu` | `str` | `T4` | Cloud GPU type (`T4` is optimal and cost-efficient for auto-labeling). |
| `--session` | `str` | `None` (auto) | Custom Colab session name. |
| `--timeout` | `float` | `1800.0` | Maximum execution timeout in seconds (default: 30 minutes). |
| `--redact-pii` | flag | `False` | Redacts competitor registration info on card margins before generating outputs. |

---

## 6. `target-scorer rectify`

Executes standalone geometric rectification and perspective unwarping on raw target photographs across all formats. Detects the dark aiming black, fits a direct least-squares ellipse, computes a closed-form homography matrix, and saves an unwarped canonical crop alongside an optical diagnostic panel.

Source: [`src/target_scorer/geometry/rectifier.py`](file:///home/antho/target_scorer/src/target_scorer/geometry/rectifier.py)

### CLI Synopsis
```bash
target-scorer rectify [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--image` | `str` | **Required** | Path to target photograph (or `-` for `stdin`). Supports HEIC, RAW/DNG, WebP, PNG, JPEG, TIFF. |
| `--target` | `str` | `B-2` | NRA target type: `B-2` or `B-3`. |
| `--output` | `str` | `rectified_target.png` | Destination filepath for the unwarped canonical target crop. |
| `--debug` | `str` | `rectification_debug.png` | Destination filepath for the diagnostic visualization panel. Set to empty string to disable. |
| `--canvas-size` | `int` | `1024` | Dimensions of canonical output square canvas in pixels (default: 1024). |
| `--scale` | `float` | `4.0` | Pixels per millimeter metric scale (default: 4.0). |

---

## 7. `target-scorer autolabel`

Assisted auto-labeling and active learning tool running locally. Predicts bullet hole coordinates and bullseye centers using a trained checkpoint, yielding standardized [`TargetAnnotation`](file:///home/antho/target_scorer/src/target_scorer/data/schema.py) JSON files, unwarped canonical crops, and human-in-the-loop verification review cards.

Source: [`src/target_scorer/data/autolabel.py`](file:///home/antho/target_scorer/src/target_scorer/data/autolabel.py)

### CLI Synopsis
```bash
target-scorer autolabel [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--image` | `str` | `None` | Path to a single target photograph to process. |
| `--input-dir` | `str` | `None` | Directory of target photos to recursively batch auto-label. |
| `--output-dir` | `str` | `private/data/real_targets` | Output directory where JSON annotations, canonical images, and review cards are saved (defaults to private submodule path). |
| `--target` | `str` | `B-2` | NRA target type: `B-2` or `B-3`. |
| `--checkpoint` | `str` | `weights/target_scoring_hrnet_1024_best.pt` | Path to model weights checkpoint. |
| `--canvas-size` | `int` | `1024` | Canonical canvas resolution. |
| `--scale` | `float` | `4.0` | Pixels per millimeter metric scale. |
| `--threshold` | `float` | `0.40` | Peak detection confidence threshold. |
| `--shots` | `int` | `10` | Maximum number of shots to detect per target. |
| `--redact-pii` | flag | `False` | Redacts competitor registration info on card margins. |

---

## 8. `target-scorer annotate`

Launches the interactive desktop GUI for reviewing, adding, moving, and deleting bullet hole markers with real-time NRA inward scoring. Features unified target discovery (opening both labeled `.json` files and unlabeled raw target photos), in-editor neural auto-labeling (`✨ Auto-Label (A)`), manual metric rectification, batch auto-labeling, NRA ring guides, caliber adjustment, and the optical **Eagle Eye Scoring Gauge**.

Source: [`src/target_scorer/gui/annotation_editor.py`](file:///home/antho/target_scorer/src/target_scorer/gui/annotation_editor.py)

### CLI Synopsis
```bash
target-scorer annotate [PATH] [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` (positional) | `None` | Path to directory containing JSON annotations or unlabeled target images (`.jpg`, `.png`, `.heic`, `.dng`, `.tiff`, `.webp`), or a single JSON/image file. |
| `--dir` | `str` | `None` | Optional explicit flag for directory containing JSON annotations and/or unlabeled target images. |
| `--target` | `str` | `None` (auto) | Override NRA target type: `B-2` or `B-3`. If omitted on labeled cards, read from JSON metadata; for unlabeled images, selectable via in-GUI dropdown. |
| `--checkpoint` | `str` | `weights/target_scoring_hrnet_1024_best.pt` | Path to model weights checkpoint for in-editor neural auto-labeling. |
| `--threshold` | `float` | `0.40` | Peak confidence threshold for in-editor auto-labeling. |
| `--copy-to-dir` | `str` | `None` | Optional destination directory to copy raw photo, canonical image, and updated annotation JSON to on Save. |

### In-Editor Auto-Labeling Features

* **Unified Target Discovery:** Automatically indexes both `.json` files and raw images (`.jpg`, `.png`, `.heic`, `.tiff`, `.dng`, `.webp`, `.bmp`), excluding review cards (`*_review.png`) and deduplicating image/JSON pairs.
* **Target Queue Filter:** Filter active target list by `All`, `Unlabeled`, or `Labeled`.
* **Raw Photo Preview:** Displays orientation-corrected raw target photos with an `[UNLABELED]` state badge before processing.
* **`✨ Auto-Label (A)` (Shortcut `A`):** Rectifies and runs in-memory neural inference via cached [`AutoLabelPipeline`](file:///home/antho/target_scorer/src/target_scorer/data/autolabel.py), populating detected shots and bullseye center onto the canvas and HUD. Fully reversible via `Ctrl+Z` (`Undo`), and saved to disk via `Ctrl+S`.
* **`📐 Rectify (Manual)`:** Rectifies raw target photos to the $1024 \times 1024$ canonical metric grid with 0 shots for pure manual placement.
* **`⚡ Auto-Label All Unlabeled`:** Background multi-threaded batch auto-labeler with live progress dialog that processes and auto-saves all unlabeled images in the folder.
* **`⚙ Auto-Label Settings...`:** Modal dialog to configure model checkpoint `.pt`, peak confidence threshold, max shots, and competitor PII redaction.

---

## 9. Python API Reference

For developers embedding `target_scorer` into automated scoring pipelines, camera daemon loops, or web services.

### A. Progressive Resizing API (`target_scorer.train`)

```python
from target_scorer.train import run_progressive_training, parse_progressive_stages

# Run the default 4-stage Harmonic 256-Grid curriculum
best_checkpoint = run_progressive_training(
    stages=None,  # Uses DEFAULT_PROGRESSIVE_STAGES: (256, 512, 768, 1024)
    checkpoint_path="weights/target_scoring_hrnet_1024_best.pt",
    real_data_dir="private/data/real_targets",
    real_ratio=0.35,
    checkpoint_dir="weights",
)

# Custom stage definition
custom_stages = parse_progressive_stages("256:2:1e-3,512:2:5e-4,1024:2:1e-4")
run_progressive_training(stages=custom_stages)
```

---

### B. Colab Progressive Training API (`target_scorer.colab_runner`)

```python
from target_scorer.colab_runner import run_colab_progressive_training

primary_weights = run_colab_progressive_training(
    gpu="A100",
    high_mem=False,
    real_data_dir="private/data/real_targets",
    real_ratio=0.35,
    checkpoint_dir="weights",
    scorecards_dir="visual_inspection_reports",
    timeout_seconds=3600.0,
)
print(f"Optimal weights saved locally to: {primary_weights}")
```

---

### C. Inference & Metrology API (`target_scorer.infer`)

```python
from target_scorer.infer import run_inference_and_scoring

shot_results, metrics = run_inference_and_scoring(
    checkpoint_path="weights/target_scoring_hrnet_1024_best.pt",
    image_path="path/to/target.heic",  # Accepts str, Path, bytes, BinaryIO, or np.ndarray
    output_path="scorecard.png",
    target_type="B-3",
    distance_yards=16.67,
    peak_threshold=0.45,
    max_shots=10,
    canvas_size=1024,
    pixels_per_mm=4.0,
)

print(f"Total Score: {metrics.total_score} - {metrics.x_count}X")
print(f"Extreme Spread: {metrics.extreme_spread_mm:.2f} mm ({metrics.extreme_spread_moa:.2f} MOA)")
```

---

### D. In-Memory Zero-Disk Edge Camera Streaming

```python
import cv2
from target_scorer.geometry.rectifier import TargetRectifier
from target_scorer.infer import run_inference_and_scoring

rectifier = TargetRectifier(target_type="B-3", canonical_dim=1024, pixels_per_mm=4.0)

# Ingest live frame (H, W, 3) BGR array from video capture
cap = cv2.VideoCapture(0)
ret, frame_bgr = cap.read()

# 1. Unwarp perspective in memory
rect_result = rectifier.rectify(frame_bgr)

# 2. Run scoring directly on in-memory canonical crop
shots, metrics = run_inference_and_scoring(
    checkpoint_path="weights/target_scoring_hrnet_1024_best.pt",
    image_path=rect_result.canonical_crop,
    target_type="B-3",
    canvas_size=1024,
    pixels_per_mm=4.0,
    output_path="edge_scorecard.png"
)
```

---

### E. Modularized Synthetic Data Generation API (`target_scorer.synthetic`)

```python
from target_scorer.config import TargetConfig
from target_scorer.synthetic.canvas import generate_cardboard_texture, draw_standard_bullseye
from target_scorer.synthetic.holes import (
    sample_bullet_void_bgr,
    sample_shot_position,
    stamp_bullet_hole,
    stamp_double_shot,
    stamp_keyhole_shot,
)
from target_scorer.synthetic.markings import (
    render_official_nra_seal,
    render_printed_target_header,
    render_scorer_annotations,
)
from target_scorer.synthetic.generator import generate_synthetic_target_sample

# Configure competition target
cfg = TargetConfig(canvas_size=1024, pixels_per_mm=4.0, target_type="B-3")

# Generate complete stratified sample
image, heatmap, annotations, center, ring_radii = generate_synthetic_target_sample(
    config=cfg,
    num_singles=6,
    num_doubles=1,
    num_keyholes=1,
)

# Sample individual 3-tone void colors:
# 'scanner_white', 'dark_void', 'tan_backer', or 'mixed' (1/3 each)
void_color = sample_bullet_void_bgr(cfg)

# Sample 4-zone stratified placement (inner black 40%, outer rings 20%, margins 20%, edge-clip 20%)
shot_pos = sample_shot_position(center, ring_radii, cfg)
```

---

### F. Defense-in-Depth Metrology Filtering API (`target_scorer.metrology.peaks`)

```python
from target_scorer.metrology.peaks import extract_subpixel_peaks, filter_spurious_peaks

# 1. Extract raw continuous sub-pixel candidates via 2D Taylor series
raw_peaks = extract_subpixel_peaks(
    heatmap=bullet_heatmap,
    threshold=0.40,
    min_distance_px=10.0,
    max_peaks=30,
)

# 2. Filter spurious margin handwriting, 1D pen strokes, and printed seal emblems
verified_peaks = filter_spurious_peaks(
    peaks=raw_peaks,
    canonical_bgr=canonical_img,
    canvas_size=1024,
    pixels_per_mm=4.0,
    target_type="B-3",
    max_peaks=10,
    filter_seal=True,
    filter_header=True,
    filter_handwriting=True,
)
```

---

### G. Diagnostic HUD & Channel Normalization (`target_scorer.metrology.visualization`)

```python
from target_scorer.metrology.visualization import render_scoring_inspection_overlay

# Automatically handles (H, W) grayscale, (H, W, 3) BGR, and (H, W, 4) BGRA arrays
scorecard_bgr = render_scoring_inspection_overlay(
    image_bgr=canonical_image,  # 2D, 3D BGR, or 4D BGRA automatically converted
    bullet_heatmap=bullet_heatmap,
    shot_results=shot_results,
    bullseye_center=bullseye_center,
    metrics=metrics,
    pixels_per_mm=4.0,
    bullet_diameter_mm=5.69,
    redact_pii=False,
)
```

---

### H. Auto-Label Pipeline API (`target_scorer.data.autolabel`)

The [`AutoLabelPipeline`](file:///home/antho/target_scorer/src/target_scorer/data/autolabel.py) class provides a high-performance, stateful auto-labeling engine designed for interactive applications, GUI workstations, and high-throughput batch scripts. It maintains an in-memory cached instance of [`HighResHeatmapUNet`](file:///home/antho/target_scorer/src/target_scorer/models/unet.py) across calls to avoid model loading overhead.

```python
from pathlib import Path
from target_scorer.data.autolabel import AutoLabelPipeline

# 1. Initialize pipeline with cached model weights
pipeline = AutoLabelPipeline(
    checkpoint_path="weights/target_scoring_hrnet_1024_best.pt",
    device_str=None,  # Automatically detects "cuda", "mps", or "cpu"
)

# 2. In-memory prediction directly on raw file paths, numpy arrays, or PIL images
annotation, canonical_bgr = pipeline.predict(
    image_source="path/to/phone_snap.heic",  # str, Path, np.ndarray, or PIL.Image
    target_type="B-3",
    canvas_size=1024,
    pixels_per_mm=4.0,
    threshold=0.40,
    max_shots=10,
    bullet_diameter_mm=5.69,
    redact_pii=False,
    target_id="relay_04_card_12",
)

print(f"Detected {len(annotation.shots)} shots:")
for shot in annotation.shots:
    print(f"  Shot #{shot.id}: ({shot.center_x:.2f}, {shot.center_y:.2f}) -> {shot.score}{'X' if shot.is_x else ''} (conf: {shot.confidence:.3f})")

print(f"Total Score: {annotation.total_score} - {annotation.x_count}X")
print(f"Canonical Image Shape: {canonical_bgr.shape}")  # (1024, 1024, 3)
```

#### Legacy / Standalone Batch Auto-Labeling Function:
```python
from target_scorer.data.autolabel import autolabel_target_image

# Runs end-to-end auto-labeling and writes .json, _canonical.png, and _review.png to output directory
annotation, json_path, review_path = autolabel_target_image(
    image_source="path/to/raw_target.jpg",
    output_dir="private/data/real_targets",
    checkpoint_path="weights/target_scoring_hrnet_1024_best.pt",
    target_type="B-3",
    canvas_size=1024,
    pixels_per_mm=4.0,
    peak_threshold=0.40,
    max_shots=10,
    redact_pii=True,
)
```

