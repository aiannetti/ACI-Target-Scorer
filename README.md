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
                      [ Raw Smartphone Photograph ]
       (Arbitrary resolution, perspective tilt up to ~45°, any aspect ratio)
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 1: TargetRectifier (Classical CV Engine)          │
       │  - EXIF orientation auto-rotation (Pillow / OpenCV)     │
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
│       ├── cli.py                # Unified CLI (infer, train, colab-train, rectify)
│       ├── config.py             # TargetConfig, TrainConfig, TargetRingsConfig
│       ├── infer.py              # End-to-end inference & metrology pipeline
│       ├── train.py              # Procedural synthetic training pipeline
│       ├── colab_runner.py       # Automated Google Colab CLI cloud orchestrator
│       ├── geometry/
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
│           └── dataset.py        # On-the-fly procedural PyTorch dataset
├── tests/
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

## Installation

```bash
# Clone the repository
git clone <repo-url> target_scorer
cd target_scorer

# Create and activate Python 3.11 virtual environment
uv venv target_env --python 3.11
source target_env/bin/activate

# Install package in editable mode
pip install -e .
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

Score target photographs or generate photorealistic synthetic competition demo cards.

#### A. Score a Real Target Photo ($1024 \times 1024$ Competition Standard)
```bash
target-scorer infer \
    --image /path/to/target_photo.jpg \
    --target B-2 \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --canvas-size 1024 \
    --scale 4.0 \
    --shots 10 \
    --output scored_target_b2.png
```

#### B. Score an NRA B-3 Target with PII Redaction
Redacts competitor registration handwriting, relay numbers, and names:
```bash
target-scorer infer \
    --image /path/to/target_photo.jpg \
    --target B-3 \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --canvas-size 1024 \
    --scale 4.0 \
    --redact-pii \
    --output scored_target_b3_redacted.png
```

#### C. Generate and Score a Synthetic 10-Shot Competition Demo Card
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
| `--session` | `auto` | Custom Colab session name |
| `--timeout` | `3600` | Maximum job execution timeout in seconds |

---

### 3. Local Model Training (`target-scorer train`)

Train or fine-tune models directly on your local workstation.

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

#### B. Fast Baseline Training ($512 \times 512$)
```bash
target-scorer train \
    --canvas-size 512 \
    --scale 4.0 \
    --epochs 15 \
    --batch-size 4 \
    --checkpoint weights/target_scoring_hrnet_512_best.pt
```

#### C. Ultra-Fast Verification Sweep ($256 \times 256$ on CPU)
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

### 4. Standalone Target Rectification (`target-scorer rectify`)

Unwarp perspective distortion and extract an orientation-corrected canonical crop:

```bash
target-scorer rectify \
    --image /path/to/skewed_photo.jpg \
    --target B-2 \
    --canvas-size 1024 \
    --scale 4.0 \
    --output rectified_target.png \
    --debug rectification_debug.png
```

---

## Running Automated Tests

Run the full battery of 41 unit and regression tests:

```bash
# Run the complete test suite:
./target_env/bin/python3 -m unittest discover tests/ -v

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
