# Target Scorer: Camera-Independent NRA Target Metrology & Scoring Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated, sub-millimeter computer vision and deep learning metrology engine designed to score official NRA Precision Pistol targets (**NRA B-2** 50 ft Slow Fire and **NRA B-3** 50 ft Timed/Rapid Fire) from arbitrary smartphone photographs.

---

## Documentation & Comprehensive Guides

For deep dives into workflows, algorithms, and APIs, consult our documentation suite in [`docs/`](file:///home/antho/target_scorer/docs):

* 📖 **[CLI & Python API Reference](file:///home/antho/target_scorer/docs/cli_and_api_reference.md):** Complete command-by-command reference with all parameters, flags, types, default values, copy-pasteable shell examples, and programmatic Python APIs for `infer`, `train`, `colab-train`, `colab-autolabel`, `rectify`, `autolabel`, and `annotate`.
* 🔍 **[Interactive Annotation Editor & Eagle Eye Guide](file:///home/antho/target_scorer/docs/interactive_annotation_editor_guide.md):** Comprehensive walkthrough of the desktop GUI annotation editor: pan & zoom controls, ring overlays, the optical Eagle Eye Scoring Gauge, NRA Rule 14.3 inward breaks, caliber selection, and keyboard shortcuts.
* 🎯 **[End-to-End Training & Workflow Tutorial](file:///home/antho/target_scorer/docs/end_to_end_training_and_workflow_tutorial.md):** Step-by-step operational guide from raw range smartphone photos to assisted auto-labeling, active learning verification, hybrid model training (Option A & B), and competitor PII privacy protection.
* 🏗️ **[Technical Architecture & Developer Guide](file:///home/antho/target_scorer/docs/architecture_and_developer_guide.md):** System data flow diagrams (Mermaid / ASCII), U-Net heatmap regression, Adaptive Wing Loss formulation, Fitzgibbon conic homography unwarping, 2D Taylor-series sub-pixel peak extraction, and `TargetAnnotation` schema.

---

## Key Features & Capabilities

* **Multi-Resolution Canonical Canvases:**
  * **$512 \times 512\text{ px}$** ($4.0\text{ px/mm}$, $128\text{ mm}$ FOV): Fast baseline and local verification.
  * **$800 \times 800\text{ px}$** ($4.0\text{ px/mm}$, $200\text{ mm}$ FOV): Wide aperture capturing outer scoring rings (6-ring and 5-ring).
  * **$1024 \times 1024\text{ px}$** ($4.0\text{ px/mm}$, $256\text{ mm}$ FOV): Official competition standard; resolves flyers, full cardboard fluting texture, and delivers **$0.335\text{ mm}$ sub-pixel precision** ($< 1.35\text{ px}$).
* **Official NRA Scoring Metrology (Rules 4.2, 4.3, & 14.3):**
  * **NRA B-2 (Rule 4.2):** 50 ft Slow Fire. Innermost ring is the 10-ring ($0.90'' = 22.86\text{ mm}$ dia). Scored as `Score / Max` (e.g. `96 / 100`).
  * **NRA B-3 (Rule 4.3):** 50 ft Timed & Rapid Fire. Innermost ring is the X-ring ($0.90'' = 22.86\text{ mm}$ dia). Scored as `Score - X / Max` (e.g. `100 - 7X / 100`).
  * **Inward Break Evaluation (Rule 14.3):** Tangent touches on scoring ring boundaries award the higher score value, evaluated via caliber-aware flange geometry: $d_{\text{gauge}} = \max(0, d_{\text{radial}} - R_{\text{bullet}})$.
* **Interactive Annotation Editor & Eagle Eye Scoring Gauge:**
  * Real-time desktop GUI (`target-scorer annotate`) for point-and-click correction of bullet hole coordinates and bullseye origins.
  * Virtual high-magnification **Eagle Eye Loupe** ($6\times$ to $12\times$) with engraved caliber reticles (.22 LR, 9mm, .45 ACP), sub-pixel crosshairs, fine nudge pads, and automated line-break rulings.
* **Assisted Auto-Labeling & Active Learning Flywheel:**
  * Model-assisted labeling tool (`target-scorer autolabel`) generating canonical unwarped crops, standardized `TargetAnnotation` JSON annotations, and visual review cards with green crosshairs for rapid human review.
* **Format-Agnostic Ingestion & Zero-Disk Edge Streaming:**
  * Native decoding of **Apple HEIC/HEIF** and **Apple ProRAW DNG** (via `pillow-heif`), Android Ultra HDR / JPEG, desktop formats (PNG, TIFF, BMP, WebP), stdin binary streaming (`-`), and direct in-memory edge buffers (`np.ndarray` from Raspberry Pi `Picamera2` or OpenCV video feeds).
  * Automatic EXIF orientation normalization (tags 1–8) and Display P3 to standard sRGB color profile mapping.
* **Camera Independence & Geometric Normalization:**
  * Auto-corrects perspective pitch and yaw tilt up to $\pm 45^\circ$, arbitrary aspect ratios, and card orientations via direct least-squares algebraic ellipse fitting (Fitzgibbon conic constraint) and closed-form homography unwarping.
* **Competitor Privacy & PII Redaction:**
  * Automated margin redaction (`--redact-pii`) obliterates competitor registration handwriting, relay numbers, and names via circular boundary masking and aggressive Gaussian blurring.
* **1-Click Google Colab CLI Cloud Orchestration:**
  * Automated cloud GPU provisioning (NVIDIA A100 / T4) for training (`target-scorer colab-train`) and batch auto-labeling (`target-scorer colab-autolabel`) with automatic session termination to conserve compute units.

---

## Quickstart Cheat Sheet

Below are copy-pasteable command snippets for the 7 primary operational workflows:

```bash
# 1. Score an Apple iPhone HEIC target photo
target-scorer infer \
  --image sample_targets/iphone_target.heic \
  --target B-3 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --canvas-size 1024 \
  --scale 4.0 \
  --output scored_card_b3.png

# 2. Launch Interactive Annotation Editor & Eagle Eye Loupe
target-scorer annotate datasets/real_targets

# 3. Batch auto-label a directory of range photos with PII redaction
target-scorer autolabel \
  --input-dir /path/to/range_photos \
  --output-dir datasets/real_targets \
  --target B-3 \
  --threshold 0.40 \
  --redact-pii

# 4. 1-Click cloud GPU hybrid training on NVIDIA A100 (Google Colab CLI)
target-scorer colab-train \
  --canvas-size 1024 \
  --scale 4.0 \
  --epochs 15 \
  --batch-size 8 \
  --gpu A100 \
  --real-data-dir datasets/real_targets \
  --real-ratio 0.35 \
  --initial-weights weights/target_scoring_hrnet_1024_best.pt

# 5. Cloud-accelerated batch auto-labeling on Google Colab CLI
target-scorer colab-autolabel \
  --input-dir /path/to/phone_snaps \
  --output-dir datasets/cloud_annotated \
  --target B-3 \
  --gpu T4 \
  --redact-pii

# 6. Local hybrid training (70% synthetic + 30% real data)
target-scorer train \
  --canvas-size 1024 \
  --scale 4.0 \
  --epochs 15 \
  --batch-size 2 \
  --lr 1.0e-4 \
  --real-data-dir datasets/real_targets \
  --real-ratio 0.35 \
  --initial-weights weights/target_scoring_hrnet_1024_best.pt

# 7. Standalone camera rectification & perspective unwarping
target-scorer rectify \
  --image skewed_shot.heic \
  --target B-2 \
  --canvas-size 1024 \
  --output rectified_b2.png \
  --debug rectification_debug.png
```

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
                                    │ Canonical Crop (1024x1024 @ 4.0 px/mm)
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
├── docs/
│   ├── cli_and_api_reference.md               # Complete CLI & Python API reference
│   ├── interactive_annotation_editor_guide.md # Desktop GUI & Eagle Eye loupe guide
│   ├── end_to_end_training_and_workflow_tutorial.md # Active learning & operational tutorial
│   └── architecture_and_developer_guide.md    # Technical architecture & math specs
├── src/
│   └── target_scorer/
│       ├── __init__.py           # Lazy loaders for clean imports
│       ├── cli.py                # Unified CLI (infer, train, colab-train, colab-autolabel, rectify, autolabel, annotate)
│       ├── config.py             # TargetConfig, TrainConfig, TargetRingsConfig
│       ├── infer.py              # End-to-end inference & metrology pipeline
│       ├── train.py              # Synthetic and real/hybrid training pipeline
│       ├── colab_runner.py       # Automated Google Colab CLI cloud orchestrator
│       ├── data/
│       │   ├── schema.py         # TargetAnnotation & AnnotatedShot schema & JSON I/O
│       │   ├── real_dataset.py   # PyTorch dataset for real target photographs & augmentations
│       │   ├── hybrid_dataset.py # Blended sampler (Option A hybrid & Option B pure real)
│       │   └── autolabel.py      # Active learning model-assisted auto-labeler & review cards
│       ├── geometry/
│       │   ├── canonical_loader.py # Format-agnostic loader (HEIC, RAW, buffers, EXIF, sRGB)
│       │   └── rectifier.py      # Classical perspective normalizer & ellipse fitting
│       ├── gui/
│       │   └── annotation_editor.py # Interactive GUI & Eagle Eye optical scoring loupe
│       ├── metrology/
│       │   ├── peaks.py          # 2D Taylor-series sub-pixel peak extraction
│       │   ├── scoring.py        # NRA B-2 / B-3 inward scoring & dispersion math
│       │   └── visualization.py  # Diagnostic scorecard overlay & HUD generator
│       ├── models/
│       │   ├── unet.py           # HighResHeatmapUNet architecture
│       │   └── loss.py           # Adaptive Wing Loss implementation
│       └── synthetic/
│           ├── generator.py      # Photorealistic cardboard, bullet, & handwriting synthesis
│           └── dataset.py        # Procedural PyTorch dataset with compression jitter
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
├── notebooks/
│   └── target_scorer_colab_training.ipynb # Google Colab interactive training notebook
├── pyproject.toml                # Standard PEP 517 build configuration
└── README.md
```

---

## Python Environment Setup via `uv`

To avoid dependency conflicts and ensure binary compatibility with CUDA, we recommend [`uv`](https://github.com/astral-sh/uv) to pin an isolated Python 3.11 virtual environment.

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2. Create isolated Python 3.11 environment
uv python install 3.11
uv venv target_env --python 3.11
source target_env/bin/activate

# 3. Install PyTorch with CUDA 12.1 acceleration (or use cpu index)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install Target Scorer in editable development mode
uv pip install -e .

# 5. Verify CUDA acceleration
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## Automated Test Suite

Run the full battery of unit and regression tests:

```bash
# Run all 54 unit and regression tests:
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
