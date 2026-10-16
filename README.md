# Target Scorer: Camera-Independent NRA Target Metrology & Scoring Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated, high-precision computer vision and deep learning metrology engine designed to score official NRA Precision Pistol targets (**NRA B-2** 50 ft Slow Fire and **NRA B-3** 50 ft Timed/Rapid Fire) from arbitrary smartphone photographs.

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
                                    │ Canonical Crop (e.g. 512x512 @ 4.0 px/mm)
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

## Directory Structure

```
target_scorer/
├── src/
│   └── target_scorer/
│       ├── __init__.py           # Package exports & version
│       ├── config.py             # TargetConfig, TrainConfig, TargetRingsConfig
│       ├── cli.py                # Unified CLI (target-scorer infer/train/rectify)
│       ├── infer.py              # Main inference & metrology pipeline
│       ├── train.py              # Procedural training pipeline
│       ├── geometry/
│       │   ├── __init__.py
│       │   └── rectifier.py      # TargetRectifier & RectificationResult
│       ├── models/
│       │   ├── __init__.py
│       │   ├── unet.py           # HighResHeatmapUNet & ConvBlock
│       │   └── loss.py           # AdaptiveWingLoss
│       ├── metrology/
│       │   ├── __init__.py
│       │   ├── peaks.py          # 2D Taylor-series sub-pixel peak extraction
│       │   ├── scoring.py        # Official NRA inward scoring & ballistics
│       │   └── visualization.py  # Diagnostic inspection overlay & HUD
│       └── synthetic/
│           ├── __init__.py
│           ├── generator.py      # Procedural cardboard & bullet hole stamping
│           └── dataset.py        # SyntheticTargetDataset (Sprint 3 augmentations)
├── tests/
│   ├── __init__.py
│   ├── test_low_res_metrology.py # Comprehensive low-res metrology test battery
│   └── test_camera_angles.py     # Oblique camera tilt stress test suite
├── weights/
│   └── target_scoring_hrnet_best.pt  # PyTorch model checkpoint
├── docs/
│   ├── Camera_Independence_Plan.md
│   ├── target_scoring_system_context_document.md
│   └── target_scoring_testing_plan.md
├── samples/                      # Reference synthetic targets and diagnostics
├── pyproject.toml                # Standard PEP 517/518 build configuration
├── .gitignore
├── infer_and_score.py            # Backward-compatible CLI wrapper
├── train_heatmap_model.py        # Backward-compatible CLI wrapper
└── target_rectifier.py           # Backward-compatible CLI wrapper
```

---

## Installation

```bash
# Clone the repository
git clone <repo-url> target_scorer
cd target_scorer

# Activate virtual environment
source target_env/bin/activate

# Install in editable mode
pip install -e .
```

---

## Quick Start & Usage

### 1. Scoring & Metrology Inference
Score a target photo with automatic geometric unwarping and NRA inward scoring:

```bash
# Score from an image
target-scorer infer --image samples/synthetic_target_b-2.png --target B-2

# Or generate and score a synthetic demo target directly:
target-scorer infer --target B-2

# Fast low-resolution verification:
target-scorer infer --canvas-size 256 --scale 2.0
```

### 2. Standalone Target Rectification
Extract and calibrate a canonical crop from a tilted photo:

```bash
target-scorer rectify --image samples/synthetic_target_b-2.png --output rectified.png --debug rect_debug.png
```

### 3. Model Training
Run procedural training with real-world geometric and photometric augmentations:

```bash
# Production training:
target-scorer train --epochs 15 --batch-size 4

# Quick low-resolution convergence test:
target-scorer train --canvas-size 256 --scale 2.0 --epochs 3 --batch-size 4
```

---

## Running Unit & Metrology Tests

```bash
# Run the low-resolution metrology test suite:
python -m unittest tests/test_low_res_metrology.py -v

# Run the camera angles stress suite:
python -m unittest tests/test_camera_angles.py -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
