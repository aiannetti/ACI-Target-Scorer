# Target Scorer: Camera-Independent NRA Target Metrology & Scoring Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated, sub-millimeter computer vision and deep learning metrology engine designed to score official NRA Precision Pistol targets (**NRA B-2** 50 ft Slow Fire and **NRA B-3** 50 ft Timed/Rapid Fire) from arbitrary smartphone photographs.

---

## Key Features & Capabilities

* **Multi-Resolution Canonical Canvases:**
  * **$512 \times 512\text{ px}$** ($4.0\text{ px/mm}$, $128\text{ mm}$ FOV): Ultra-fast baseline and local testing.
  * **$800 \times 800\text{ px}$** ($4.0\text{ px/mm}$, $200\text{ mm}$ FOV): Wide aperture capturing outer scoring rings (6-ring and 5-ring).
  * **$1024 \times 1024\text{ px}$** ($4.0\text{ px/mm}$, $256\text{ mm}$ FOV): Official competition standard; resolves flyers, full cardboard fluting texture, and delivers **$0.335\text{ mm}$ sub-pixel precision** ($< 1.35\text{ px}$).
* **Official NRA Scoring Compliance:**
  * **NRA B-2 (Rule 4.2):** 50 ft Slow Fire. Innermost ring is the 10-ring ($0.90'' = 22.86\text{ mm}$ dia). No X-ring. Scored as `Score / Max` (e.g. `96 / 100`).
  * **NRA B-3 (Rule 4.3):** 50 ft Timed & Rapid Fire. Innermost ring is the X-ring ($0.90'' = 22.86\text{ mm}$ dia). Scored as `Score - X / Max` (e.g. `100 - 7X / 100`).
  * Evaluates full **10-shot competition strings** with caliber-aware physical line-breaking rules.
* **Cross-Platform Format-Agnostic Ingestion:**
  * Universal ingestion of **Apple HEIC/HEIF** and **Apple ProRAW DNG** (via `pillow-heif`), Android Ultra HDR / JPEG, desktop formats (PNG, TIFF, BMP, WebP), and direct in-memory edge buffers (`np.ndarray` from Raspberry Pi `Picamera2` or OpenCV video streams).
  * Automatic EXIF orientation normalization (tags 1–8) and Display P3 to standard sRGB color management system (CMS) mapping.
* **Camera Independence & Geometric Normalization:**
  * Auto-corrects perspective pitch and yaw tilt up to $\pm 45^\circ$, arbitrary smartphone aspect ratios, and orientations via direct least-squares algebraic ellipse fitting and closed-form homography rectification.
* **Privacy & PII Protection:**
  * Built-in automated margin redaction (`--redact-pii`) masking competitor registration handwriting, relay numbers, and shooter names with circular cardboard cropping.
* **Warm-Start Training & Cloud Orchestration:**
  * 1-click cloud GPU training via Google Colab CLI (`target-scorer colab-train`) on NVIDIA A100 / T4 GPUs.
  * Warm-start fine-tuning (`--initial-weights`) converging in 10–12 epochs with composite fitness checkpointing ($\text{Score} = \text{Match Rate} - 10 \times \text{MRE}$).

---

## Architecture Overview

```
                      [ Raw Target Image / Buffer / Stream ]
        (Apple HEIC, ProRAW DNG, Ultra HDR, JPEG, PNG, or Picamera2 RAM)
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 1: ImageCanonicalizer & TargetRectifier           │
       │  - Polymorphic ingestion (HEIC, RAW, JPEG, PNG, Buffers)│
       │  - EXIF orientation auto-rotation (tags 1–8)            │
       │  - Display P3 / ICC color profile mapping to sRGB (CMS) │
       │  - Laplacian variance motion blur check (> 75.0)        │
       │  - Specular glare suppression (Navier-Stokes inpaint)   │
       │  - Bullseye segmentation & morphological close          │
       │  - Direct least-squares algebraic ellipse fit (Fitzgib.)│
       │  - Roll-invariant 3x3 Homography -> Metric calibration   │
       └────────────────────────────┬─────────────────────────────┘
                                    │ Canonical Crop (512x512, 800x800, or 1024x1024 @ 4.0 px/mm)
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 2: HighResHeatmapUNet (PyTorch Deep Learning)     │
       │  - Dual-channel regression (Ch0: Bullets, Ch1: Bullseye) │
       │  - Full-resolution stem & high-frequency skip merges    │
       │  - Trained via Adaptive Wing Loss (A-Wing)              │
       └────────────────────────────┬─────────────────────────────┘
                                    │ Continuous 2D Heatmaps
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 3: Metrology & Scoring Engine                      │
       │  - Caliber-aware spatial Non-Maximum Suppression (NMS)   │
       │  - 2D Quadratic Taylor-series sub-pixel peak refinement  │
       │  - Official NRA B-2 / B-3 inward scoring (effective gauge)│
       │  - Ballistics: Extreme Spread (mm/MOA), Mean Radius, COI │
       │  - Full diagnostic scorecard generation with HUD banner  │
       └──────────────────────────────────────────────────────────┘
```

---

## Canvas Resolution Specifications

| Parameter | Baseline ($512 \times 512$) | High-Res ($800 \times 800$) | Competition ($1024 \times 1024$) |
| :--- | :---: | :---: | :---: |
| **Pixels Per Millimeter** | $4.0\text{ px/mm}$ | $4.0\text{ px/mm}$ | **$4.0\text{ px/mm}$** |
| **Physical Field of View (FOV)** | $128 \times 128\text{ mm}$ ($5.04''$) | $200 \times 200\text{ mm}$ ($7.87''$) | **$256 \times 256\text{ mm}$ ($10.08''$)** |
| **Rings Captured (NRA B-2)** | Rings 10, 9, 8, 7, partial 6 | Rings 10 down to 5 | **All Rings (10 down to 4) + Margins** |
| **Rings Captured (NRA B-3)** | Rings X, 10, 9, 8, partial 7 | Rings X down to 6 | **All Rings (X down to 6) + Margins** |
| **Heatmap Sigma ($\sigma$)** | $2.5\text{ px}$ | $2.8\text{ px}$ | **$3.2\text{ px}$** |
| **Sub-Pixel MRE** | $0.465\text{ mm}$ | $0.380\text{ mm}$ | **$0.335\text{ mm}$ ($< 1.35\text{ px}$)** |
| **Recommended Hardware** | CPU / GTX 1060 (batch 4) | GTX 1060 (batch 4) / T4 | **NVIDIA A100 / T4 (batch 8)** |

---

## Directory Structure

```
target_scorer/
├── src/
│   └── target_scorer/
│       ├── __init__.py           # Package exports (lazy loaders for clean imports)
│       ├── cli.py                # Unified CLI (infer, train, colab-train, rectify, autolabel)
│       ├── config.py             # TargetConfig, TrainConfig, TargetRingsConfig
│       ├── infer.py              # End-to-end inference & metrology pipeline
│       ├── train.py              # Synthetic and real/hybrid training pipeline
│       ├── colab_runner.py       # Automated Google Colab CLI cloud orchestrator
│       ├── data/
│       │   ├── schema.py         # TargetAnnotation & AnnotatedShot schema & JSON I/O
│       │   ├── real_dataset.py   # PyTorch dataset for real target photographs & augmentations
│       │   ├── hybrid_dataset.py # Blended sampler (Option A hybrid & Option B pure real)
│       │   └── autolabel.py      # Active learning model-assisted auto-labeler & review card generator
│       ├── geometry/
│       │   ├── canonical_loader.py # Format-agnostic loader (HEIC, RAW, buffers, EXIF, sRGB)
│       │   └── rectifier.py      # Classical perspective normalizer & ellipse fitting
│       ├── metrology/
│       │   ├── peaks.py          # 2D Taylor-series sub-pixel peak extraction
│       │   ├── scoring.py        # NRA B-2 / B-3 inward scoring & dispersion math
│       │   └── visualization.py  # Diagnostic scorecard overlay & HUD generator
│       ├── models/
│       │   ├── unet.py           # HighResHeatmapUNet architecture
│       │   └── loss.py           # Adaptive Wing Loss implementation
│       └── synthetic/
│           ├── generator.py      # Photorealistic cardboard, bullet, & handwriting synthesis
│           └── dataset.py        # On-the-fly procedural PyTorch dataset with compression jitter
├── datasets/
│   └── calibration_targets/      # Sample calibration targets with raw photos & ground truth JSON
├── tests/
│   ├── test_real_dataset.py      # Real dataset, schema, hybrid, and autolabel test suite
│   ├── test_canonical_loader.py  # Format-independent ingestion & in-memory buffer test suite
│   ├── test_low_res_metrology.py # 35-test scoring compliance & metrology test suite
│   ├── test_camera_angles.py     # Perspective rectification & tilt test suite
│   └── test_benchmark_low_res.py # 100-target statistical regression benchmark
├── weights/
│   ├── target_scoring_hrnet_1024_best.pt # Competition model (98.9% match, 0.335mm MRE)
│   ├── target_scoring_hrnet_512_best.pt  # 512x512 baseline model
│   └── target_scoring_hrnet_best.pt      # Initial development weights
├── visual_inspection_reports/    # High-resolution visual scorecards (PNG)
│   ├── colab_scorecard_1024_b2.png
│   ├── colab_scorecard_1024_b2_redacted.png
│   ├── colab_scorecard_1024_b3.png
│   └── colab_scorecard_1024_b3_redacted.png
├── scripts/
│   ├── create_colab_bundle.py    # Packages repository into clean cloud zip bundle
│   └── generate_benchmark_visuals.py
├── notebooks/
│   └── target_scorer_colab_training.ipynb
├── pyproject.toml                # Standard PEP 517 build configuration
└── README.md
```

---

## Python Environment Setup via `uv`

### Why `uv` is Required: Python & PyTorch Compatibility

PyTorch pre-built wheels and CUDA binaries **do not immediately support bleeding-edge Python releases** (such as Python 3.13 or 3.14+). Installing PyTorch on an unsupported or system-default Python version typically fails with:
```text
ERROR: Could not find a version that satisfies the requirement torch (from versions: none)
ERROR: No matching distribution found for torch
```
Additionally, modern Linux distributions (Ubuntu 23.04+, Debian 12+) enforce **PEP 668** ("externally managed environment"), which intentionally blocks system-wide `pip install`.

To resolve this, we use **[`uv`](https://github.com/astral-sh/uv)**—an extremely fast, standalone Python package and environment manager written in Rust. `uv` allows you to automatically download, isolate, and pin a fully compatible **Python 3.11** interpreter without needing system root privileges, manual compilation, or affecting system packages.

---

### Step-by-Step Setup Guide

#### Step 1: Install `uv`
On Linux / macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # Or source ~/.zshrc
```
On Windows (PowerShell):
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Step 2: Setup Project Directory & Create Pinned Python 3.11 Environment
`uv` will automatically download and bootstrap a standalone Python 3.11 build if one is not installed on your system:
```bash
# Create project directory (or git clone <repo-url> target_scorer)
mkdir target_scorer
cd target_scorer

# Download Python 3.11 and create isolated virtual environment
uv python install 3.11
uv venv target_env --python 3.11

# Activate the virtual environment
source target_env/bin/activate  # On Windows: target_env\Scripts\activate
```

#### Step 3: Install PyTorch with CUDA Acceleration
Install the official PyTorch CUDA 12.1 or CUDA 12.4 pre-built binaries via `uv pip`:

* **For NVIDIA GPU (CUDA 12.1 - Recommended for Linux / WSL2):**
  ```bash
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```
* **For NVIDIA GPU (CUDA 12.4):**
  ```bash
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```
* **For CPU Only:**
  ```bash
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
* **For Apple Silicon (Mac M1/M2/M3 with MPS):**
  ```bash
  uv pip install torch torchvision
  ```

#### Step 4: Install Target Scorer in Editable Mode
```bash
uv pip install -e .
```

#### Step 5: Verify Environment & Hardware Acceleration
Run the self-check snippet to ensure Python version, PyTorch, and CUDA are correctly wired:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```
Output:
```text
CUDA Available: True
Device: NVIDIA GeForce GTX 1060 6GB (or NVIDIA A100 / T4)
```

---

## User Execution Guide

The engine is controlled via the unified `target-scorer` command-line executable.

```
target-scorer <subcommand> [options]

Subcommands:
  infer        Score a target image, run inward scoring, and render visual scorecard
  train        Execute procedural synthetic training locally
  colab-train  Orchestrate cloud GPU training on Google Colab CLI (A100 / T4)
  rectify      Run standalone classical geometric unwarping & metric calibration
```

---

### 1. Scoring & Metrology (`target-scorer infer`)

Score real target photographs across any format or generate photorealistic synthetic competition demo cards.

#### A. Score Real Target Photographs (HEIC, RAW/DNG, WebP, PNG, JPEG)
The engine automatically detects and unifies all mobile, desktop, and professional photography formats:

```bash
# Score an Apple iPhone HEIC photograph (auto-corrected for EXIF rotation & Display P3 gamut)
target-scorer infer \
    --image /path/to/target.heic \
    --target B-2 \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --canvas-size 1024 \
    --scale 4.0 \
    --shots 10 \
    --output scored_target_heic.png

# Score a professional Camera RAW / Apple ProRAW DNG file
target-scorer infer \
    --image /path/to/target.dng \
    --target B-2 \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --canvas-size 1024 \
    --scale 4.0 \
    --output scored_target_dng.png

# Score a lossless WebP, PNG, or standard JPEG photo
target-scorer infer \
    --image /path/to/target.webp \
    --target B-3 \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --canvas-size 1024 \
    --scale 4.0 \
    --output scored_target_webp.png

# Stream binary image data directly from shell pipes or curl via stdin ('-')
cat /path/to/target.heic | target-scorer infer \
    --image - \
    --target B-2 \
    --canvas-size 1024 \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --output scored_target_piped.png
```

#### B. Score with Competitor Registration PII Redaction
Masks competitor registration handwriting, relay numbers, and names on card margins:
```bash
target-scorer infer \
    --image /path/to/target.heic \
    --target B-3 \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --canvas-size 1024 \
    --scale 4.0 \
    --redact-pii \
    --output scored_target_b3_redacted.png
```

#### C. In-Memory Buffer API (Zero-Disk-I/O Edge & Streaming Inference)
For edge applications (e.g., Raspberry Pi `Picamera2` video loops or web API endpoints), pass in-memory NumPy arrays or byte streams directly without disk serialization:
```python
import cv2
from target_scorer import run_inference_and_scoring

# Ingest an in-memory BGR or RGB NumPy frame directly from a camera
# frame_bgr: np.ndarray shape (H, W, 3) uint8
shots, metrics = run_inference_and_scoring(
    checkpoint_path="weights/target_scoring_hrnet_1024_best.pt",
    image_path=frame_bgr,  # Accepts in-memory np.ndarray, bytes, or file path
    target_type="B-2",
    canvas_size=1024,
    pixels_per_mm=4.0,
    max_shots=10,
    output_path="realtime_scorecard.png"
)

print(f"Total Score:     {metrics.total_score} / 100")
print(f"Extreme Spread:  {metrics.extreme_spread_mm:.2f} mm ({metrics.extreme_spread_moa:.2f} MOA)")
print(f"Mean Radius:     {metrics.mean_radius_mm:.2f} mm")
```

#### D. Generate and Score a Synthetic 10-Shot Competition Demo Card
When no `--image` is provided, a deterministic 10-shot competition target is procedurally rendered:
```bash
# NRA B-2 Slow Fire (50 ft): Scores 96 / 100
target-scorer infer \
    --target B-2 \
    --canvas-size 1024 \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --output visual_inspection_reports/colab_scorecard_1024_b2.png

# NRA B-3 Timed/Rapid Fire (50 ft): Scores 100 - 7X / 100
target-scorer infer \
    --target B-3 \
    --canvas-size 1024 \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --output visual_inspection_reports/colab_scorecard_1024_b3.png
```

#### Inference Flags Reference:
| Flag | Default | Description |
| :--- | :---: | :--- |
| `--image` | `None` | Path to raw smartphone photo (generates synthetic demo if omitted) |
| `--target` | `B-2` | NRA target face: `B-2` (Rule 4.2 Slow) or `B-3` (Rule 4.3 Timed/Rapid) |
| `--checkpoint` | `weights/...` | Path to trained PyTorch `.pt` model weights |
| `--canvas-size` | `512` | Canonical dimension (`512`, `800`, or `1024`) |
| `--scale` | `4.0` | Metric scale in pixels per millimeter |
| `--shots` | `10` | Number of shots to evaluate (10 for official NRA strings) |
| `--threshold` | `0.45` | Heatmap peak detection confidence threshold |
| `--distance` | `16.67` | Shooting distance in yards (50 ft = 16.67 yd) |
| `--device` | `auto` | Execution device: `cuda`, `cpu`, or `mps` |
| `--redact-pii` | `False` | Masks margin competitor registration info and crops card |
| `--debug-rect` | `rect_debug.png` | Saves side-by-side rectification diagnostic image |

---

### 2. Cloud GPU Training via Colab CLI (`target-scorer colab-train`)

The `colab-train` command provides fully automated, zero-setup cloud training:
1. Automatically packages the codebase (excluding large local weights and caches).
2. Provisions a Google Colab VM with an NVIDIA A100 or T4 accelerator via `colab` CLI.
3. Uploads the bundle and starting weights.
4. Executes mixed-precision (AMP) training with real-time log streaming.
5. Evaluates a post-training scorecard and downloads the best checkpoint locally.
6. Automatically terminates the cloud VM to conserve compute units.

#### A. Warm-Start Fine-Tuning on NVIDIA A100 ($1024 \times 1024$, ~6 minutes)
Start directly from pre-trained weights to reach $98.9\%$ recall in 10–12 epochs:
```bash
target-scorer colab-train \
    --canvas-size 1024 \
    --scale 4.0 \
    --epochs 12 \
    --batch-size 8 \
    --lr 1.5e-4 \
    --sigma 3.2 \
    --gpu A100 \
    --initial-weights weights/target_scoring_hrnet_1024_best.pt
```

#### B. Full 45-Epoch Cold Training on NVIDIA A100 ($1024 \times 1024$)
Train from scratch with Cosine Annealing:
```bash
target-scorer colab-train \
    --canvas-size 1024 \
    --scale 4.0 \
    --epochs 45 \
    --batch-size 8 \
    --lr 3.5e-4 \
    --sigma 3.2 \
    --gpu A100
```

#### C. Cloud Training at $800 \times 800$ Resolution (NVIDIA T4 or A100)
```bash
target-scorer colab-train \
    --canvas-size 800 \
    --scale 4.0 \
    --epochs 20 \
    --batch-size 8 \
    --lr 2.5e-4 \
    --sigma 2.8 \
    --gpu T4
```

#### Colab Training Flags Reference:
| Flag | Default | Description |
| :--- | :---: | :--- |
| `--canvas-size` | `800` | Resolution preset: `512`, `800`, or `1024` |
| `--scale` | `4.0` | Pixels per millimeter scale |
| `--epochs` | `15` | Total training epochs |
| `--batch-size` | `8` | Batch size (8 recommended for A100) |
| `--lr` | `3.5e-4` | Initial learning rate |
| `--sigma` | `None` | Gaussian sigma in pixels ($3.2$ for 1024, $2.8$ for 800) |
| `--gpu` | `A100` | Cloud accelerator: `A100`, `T4`, or `L4` |
| `--initial-weights` | `None` | Local `.pt` checkpoint to upload for warm-start fine-tuning |
| `--real-data-dir` | `None` | Local directory of real target annotations to upload & train |
| `--real-ratio` | `0.35` | Real target sampling probability (1.0 = pure real, 0.35 = hybrid) |
| `--session` | `auto` | Custom Colab session name |
| `--timeout` | `3600` | Maximum job execution timeout in seconds |

---

### 3. Local Model Training (`target-scorer train`)

Train or fine-tune models directly on your local workstation. Supports 100% synthetic procedural training, pure real target fine-tuning (Option B), or blended hybrid training (Option A).

#### A. Warm-Start Fine-Tuning Locally ($1024 \times 1024$ on GTX 1060 6GB)
```bash
target-scorer train \
    --canvas-size 1024 \
    --scale 4.0 \
    --epochs 12 \
    --batch-size 2 \
    --lr 1.5e-4 \
    --sigma 3.2 \
    --initial-weights weights/target_scoring_hrnet_1024_best.pt \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt
```
*(Note: `--batch-size 2` is used to fit 1024x1024 feature maps within 6GB VRAM on GTX 1060).*

#### B. Pure Real Target Fine-Tuning (Option B: Rapid Domain Adaptation)
Fine-tune directly on real target photographs to rapidly adapt weights to real cardboard fibers, range shadows, and physical bullet tears:
```bash
target-scorer train \
    --canvas-size 1024 \
    --scale 4.0 \
    --epochs 10 \
    --batch-size 2 \
    --lr 5.0e-5 \
    --sigma 3.2 \
    --real-data-dir datasets/calibration_targets \
    --real-ratio 1.0 \
    --initial-weights weights/target_scoring_hrnet_1024_best.pt \
    --checkpoint weights/target_scoring_hrnet_1024_real_adapted.pt
```

#### C. Hybrid Blended Training (Option A: 70% Synthetic + 30% Real)
Blend large-scale procedural synthetic edge cases with real paper photographs to maintain maximum geometric diversity while grounding in real range photography:
```bash
target-scorer train \
    --canvas-size 1024 \
    --scale 4.0 \
    --epochs 15 \
    --batch-size 2 \
    --lr 1.0e-4 \
    --sigma 3.2 \
    --real-data-dir datasets/calibration_targets \
    --real-ratio 0.35 \
    --initial-weights weights/target_scoring_hrnet_1024_best.pt \
    --checkpoint weights/target_scoring_hrnet_1024_hybrid.pt
```

#### D. Fast Baseline Training ($512 \times 512$)
```bash
target-scorer train \
    --canvas-size 512 \
    --scale 4.0 \
    --epochs 15 \
    --batch-size 4 \
    --checkpoint weights/target_scoring_hrnet_512_best.pt
```

#### E. Ultra-Fast Verification Sweep ($256 \times 256$ on CPU)
```bash
target-scorer train \
    --canvas-size 256 \
    --scale 2.0 \
    --epochs 2 \
    --train-samples 10 \
    --val-samples 5 \
    --batch-size 2
```

---

### 4. Assisted Auto-Labeling & Active Learning (`target-scorer autolabel`)

Accelerate dataset creation by using the trained model to propose candidate bullet hole locations and official NRA scores on newly photographed target cards. Generates standardized JSON annotations and visual human-in-the-loop review cards:

#### A. Auto-Label a Single Target Photograph
```bash
target-scorer autolabel \
    --image /path/to/phone_target.jpg \
    --target B-2 \
    --output-dir datasets/real_targets \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --canvas-size 1024 \
    --scale 4.0 \
    --threshold 0.40 \
    --shots 10
```

#### B. Batch Auto-Label an Entire Directory
```bash
target-scorer autolabel \
    --input-dir /path/to/range_photos/ \
    --target B-3 \
    --output-dir datasets/real_targets \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --threshold 0.40
```

Each processed image generates:
1. `*_canonical.png`: Unwarped $1024 \times 1024$ canonical crop at $4.0\text{ px/mm}$.
2. `*.json`: Standardized `TargetAnnotation` schema containing sub-pixel coordinates, ring scores, and target metadata.
3. `*_review.png`: High-resolution review card with green crosshairs, shot IDs, and ring scores for rapid human verification.

---

### 5. Standalone Target Rectification (`target-scorer rectify`)

Unwarp perspective distortion and extract an orientation-corrected, metric-calibrated canonical crop across all supported formats (HEIC, RAW/DNG, WebP, PNG, JPEG, or stdin):

#### A. Command-Line Rectification Across Disparate Formats
```bash
# Rectify an Apple iPhone HEIC photo (auto-corrects EXIF orientation and P3 color gamut)
target-scorer rectify \
    --image /path/to/skewed_target.heic \
    --target B-2 \
    --canvas-size 1024 \
    --scale 4.0 \
    --output rectified_target_b2.png \
    --debug rectification_debug.png

# Rectify a professional Camera RAW / Apple ProRAW DNG file
target-scorer rectify \
    --image /path/to/raw_target.dng \
    --target B-3 \
    --canvas-size 1024 \
    --scale 4.0 \
    --output rectified_target_b3.png

# Rectify via stdin streaming pipe
cat skewed_target.heic | target-scorer rectify \
    --image - \
    --target B-2 \
    --canvas-size 1024 \
    --output rectified_target_piped.png
```

#### B. Python In-Memory Rectification API (Picamera2 / Real-Time Video)
```python
import cv2
from target_scorer import TargetRectifier

rectifier = TargetRectifier(target_type="B-2", canonical_dim=1024, pixels_per_mm=4.0)

# Unwarp an in-memory frame directly from camera memory without writing to disk
# frame_bgr: np.ndarray shape (H, W, 3) uint8
result = rectifier.rectify(frame_bgr)

print(f"Canonical Dimensions: {result.canonical_crop.shape}")  # (1024, 1024, 3)
print(f"Optical Tilt:         {result.tilt_angle_deg:.1f}° off-normal")
print(f"Motion Blur Variance: {result.blur_metric:.1f}")

# Save or downstream-stream the normalized crop
cv2.imwrite("canonical_1024.png", result.canonical_crop)
```

---

## Running Automated Tests

Run the full battery of 54 unit and regression tests:

```bash
# Run the complete test suite (54 tests):
./target_env/bin/python3 -m unittest discover tests/ -v

# Run real dataset ingestion, schema, and active learning tests (11 tests):
./target_env/bin/python3 -m unittest tests/test_real_dataset.py -v

# Run format-independent ingestion & in-memory buffer tests (7 tests):
./target_env/bin/python3 -m unittest tests/test_canonical_loader.py -v

# Run metrology and scoring assertions (35 tests):
./target_env/bin/python3 -m unittest tests/test_low_res_metrology.py -v

# Run oblique camera angle tilt stress tests (5 tests):
./target_env/bin/python3 -m unittest tests/test_camera_angles.py -v

# Run 100-target procedural statistical benchmark:
./target_env/bin/python3 -m unittest tests/test_benchmark_low_res.py -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
