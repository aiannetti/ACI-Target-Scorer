# Target Scorer: Camera-Independent NRA Target Metrology & Scoring Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated, sub-millimeter computer vision and deep learning metrology engine designed to score official NRA Precision Pistol targets (**NRA B-2** 50 ft Slow Fire and **NRA B-3** 50 ft Timed/Rapid Fire) from arbitrary smartphone photographs.

---

## Documentation & Comprehensive Guides

For deep dives into workflows, algorithms, metrology grounding, and APIs, consult our documentation suite in [`docs/`](file:///home/antho/target_scorer/docs):

* 📖 **[CLI & Python API Reference](file:///home/antho/target_scorer/docs/cli_and_api_reference.md):** Complete command-by-command reference with all parameters, flags, types, default values, copy-pasteable shell examples, and programmatic Python APIs for `infer`, `train`, `colab-train`, `colab-progressive-train`, `colab-autolabel`, `rectify`, `autolabel`, and `annotate`.
* 🔍 **[Interactive Annotation Editor & Eagle Eye Guide](file:///home/antho/target_scorer/docs/interactive_annotation_editor_guide.md):** Comprehensive walkthrough of the desktop GUI annotation editor: pan & zoom controls, ring overlays, the optical Eagle Eye Scoring Gauge, NRA Rule 14.3 inward breaks, caliber selection, and keyboard shortcuts.
* 🎯 **[End-to-End Training & Workflow Tutorial](file:///home/antho/target_scorer/docs/end_to_end_training_and_workflow_tutorial.md):** Step-by-step operational guide from raw range smartphone photos to assisted auto-labeling, active learning verification, hybrid model training (Option A & B), and competitor PII privacy protection.
* 🏗️ **[Technical Architecture & Developer Guide](file:///home/antho/target_scorer/docs/architecture_and_developer_guide.md):** System data flow diagrams (Mermaid / ASCII), physical metrology grounding, Harmonic 256-Grid Progressive Resizing, U-Net heatmap regression, Adaptive Wing Loss formulation, Fitzgibbon conic homography unwarping, 2D Taylor-series sub-pixel peak extraction, and `TargetAnnotation` schema.
* 🏛️ **[Target Scoring System Context Document](file:///home/antho/target_scorer/docs/target_scoring_system_context_document.md):** High-level system context, project mission, historical development milestones, Sprint roadmaps (Sprints 1–11), and domain metrology specifications.

---

## Key Features & Capabilities

* **Harmonic 256-Grid Progressive Resizing Curriculum:**
  * Grounded in physical target metrology (NRA targets $266.7 \times 266.7\text{ mm}$, .22 LR grease rings $0.2\text{--}0.5\text{ mm}$, Rule 14.3 break tolerance $\pm 0.08\text{ mm}$).
  * Staged multi-resolution ladder: **$256 \times 256$** ($1.0\text{ px/mm}$) $\to$ **$512 \times 512$** ($2.0\text{ px/mm}$) $\to$ **$768 \times 768$** ($3.0\text{ px/mm}$) $\to$ **$1024 \times 1024$** ($4.0\text{ px/mm}$).
  * Dynamic spatial resizing in [`HighResHeatmapUNet`](file:///home/antho/target_scorer/src/target_scorer/models/unet.py) preserving AdamW momentum buffers across stages for a **$3.5\times$ training speedup**.
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
* **Defense-in-Depth Metrology Filtering:**
  * Multi-stage geometric and chrominance filtering in [`peaks.py`](file:///home/antho/target_scorer/src/target_scorer/metrology/peaks.py) to eliminate spurious candidates.
  * Combines ink chroma tests (rejection of blue/red ballpoint ink, colored felt markers, and graphite pencils), 1D pen stroke aspect ratio rejection ($\text{length} / \text{width} > 2.2$), physical bullet void confirmation (dark cavity + fiber tear margins), and official NRA seal/target header exclusion zones.
* **Modularized Photorealistic Synthetic Generator Engine:**
  * Decoupled architecture across [`canvas.py`](file:///home/antho/target_scorer/src/target_scorer/synthetic/canvas.py), [`holes.py`](file:///home/antho/target_scorer/src/target_scorer/synthetic/holes.py), [`markings.py`](file:///home/antho/target_scorer/src/target_scorer/synthetic/markings.py), and unified facade [`generator.py`](file:///home/antho/target_scorer/src/target_scorer/synthetic/generator.py).
  * Implements **3-tone bullet void penetration** (white target paper, dark aperture cavity, and tan cardboard backer) simulating realistic range backstops.
  * **4-zone stratified shot placement** (Zone 1: inner black 40%, Zone 2: outer scoring rings 20%, Zone 3: paper margins 20%, Zone 4: off-paper/clipped edges 20%).
  * Procedural official NRA seal stamping and cursive handwriting generation for zero-heatmap hard negative mining.
* **Competitor Privacy & PII Redaction:**
  * Automated margin redaction (`--redact-pii`) obliterates competitor registration handwriting, relay numbers, and names via circular boundary masking and aggressive Gaussian blurring.
* **1-Click Google Colab CLI Cloud Orchestration:**
  * Single-session cloud GPU training on NVIDIA A100 / T4 for single-resolution (`target-scorer colab-train`) or multi-stage progressive curriculum (`target-scorer colab-progressive-train`), plus cloud auto-labeling (`target-scorer colab-autolabel`).

---

## Quickstart Cheat Sheet

Below are copy-pasteable command snippets for all primary operational workflows:

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

# 3. 1-Click Multi-Stage Progressive Cloud Training (Harmonic 256-Grid on NVIDIA A100)
target-scorer colab-progressive-train \
  --gpu A100 \
  --real-data-dir datasets/real_targets \
  --real-ratio 0.35

# 4. Local Progressive Resizing Training (256 -> 512 -> 768 -> 1024)
target-scorer train \
  --progressive \
  --real-data-dir datasets/real_targets \
  --real-ratio 0.35 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt

# 5. Batch auto-label a directory of range photos with PII redaction
target-scorer autolabel \
  --input-dir /path/to/range_photos \
  --output-dir datasets/real_targets \
  --target B-3 \
  --threshold 0.40 \
  --redact-pii

# 6. Cloud-accelerated batch auto-labeling on Google Colab CLI
target-scorer colab-autolabel \
  --input-dir /path/to/phone_snaps \
  --output-dir datasets/cloud_annotated \
  --target B-3 \
  --gpu T4 \
  --redact-pii

# 7. Standalone camera rectification & perspective unwarping
target-scorer rectify \
  --image skewed_shot.heic \
  --target B-2 \
  --canvas-size 1024 \
  --output rectified_b2.png \
  --debug rectification_debug.png
```

---

## Harmonic 256-Grid Resolution Specifications

The progression enforces uniform step properties: **$+256\text{ px}$ dimension** and **$+1.0\text{ px/mm}$ scale** per stage.

| Stage | Resolution ($\text{px}$) | Metric Scale | Field of View (FOV) | Gaussian $\sigma$ | Recommended Batch | Physical Metrology Function |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | $256 \times 256$ | $1.0\text{ px/mm}$ | $256 \times 256\text{ mm}$ | $1.6\text{ px}$ | 16 (GPU) | **Macro Floor:** Coarse bullseye segmentation and global card alignment (> 900 img/s). |
| **Stage 2** | $512 \times 512$ | $2.0\text{ px/mm}$ | $256 \times 256\text{ mm}$ | $2.2\text{ px}$ | 12 (GPU) | **Ring Tracking:** Scoring ring edge envelopes and macro grease ring tracking. |
| **Stage 3** | $768 \times 768$ | $3.0\text{ px/mm}$ | $256 \times 256\text{ mm}$ | $2.8\text{ px}$ | 8 (GPU) | **Double Disentanglement:** Warp-aligned ($24 \times 32$); resolves overlapping double/triple holes. |
| **Stage 4** | **$1024 \times 1024$** | **$4.0\text{ px/mm}$** | **$256 \times 256\text{ mm}$** | **$3.2\text{ px}$** | **8 (A100) / 2 (GTX)**| **Competition Standard:** Sub-pixel error $< 0.08\text{ mm}$ MRE; Rule 14.3 tangent precision. |

---

## Directory Structure

```
target_scorer/
├── docs/
│   ├── cli_and_api_reference.md               # Complete CLI & Python API reference
│   ├── interactive_annotation_editor_guide.md # Desktop GUI & Eagle Eye loupe guide
│   ├── end_to_end_training_and_workflow_tutorial.md # Active learning & operational tutorial
│   ├── architecture_and_developer_guide.md    # Technical architecture & math specs
│   └── target_scoring_system_context_document.md # High-level system context & sprint roadmap
├── src/
│   └── target_scorer/
│       ├── __init__.py           # Lazy loaders for clean imports
│       ├── cli.py                # Unified CLI (infer, train, colab-train, colab-progressive-train, etc.)
│       ├── config.py             # TargetConfig, TrainConfig, TargetRingsConfig
│       ├── infer.py              # End-to-end inference & metrology pipeline
│       ├── train.py              # Single-resolution and Progressive Resizing training pipeline
│       ├── colab_runner.py       # Cloud orchestrator for Colab CLI (single-res & progressive)
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
│       │   ├── peaks.py          # 2D Taylor-series sub-pixel peak extraction & defense-in-depth filtering
│       │   ├── scoring.py        # NRA B-2 / B-3 inward scoring & dispersion math
│       │   └── visualization.py  # Diagnostic scorecard overlay & HUD generator
│       ├── models/
│       │   ├── unet.py           # Fully convolutional HighResHeatmapUNet architecture
│       │   └── loss.py           # Adaptive Wing Loss implementation
│       └── synthetic/
│           ├── __init__.py       # Submodule exports & facade integration
│           ├── canvas.py         # Photorealistic cardboard textures & bullseye geometry
│           ├── holes.py          # 3-tone bullet voids, ragged fiber tears, keyholes, & doubles
│           ├── markings.py       # Official NRA seal, procedural handwriting, & hard negative mining
│           ├── generator.py      # Facade orchestrator for synthetic target generation
│           └── dataset.py        # Procedural PyTorch dataset with compression jitter
├── datasets/
│   └── calibration_targets/      # Sample calibration targets with raw photos & ground truth JSON
├── tests/
│   ├── test_progressive_training.py # Dynamic spatial resizing & progressive curriculum test suite
│   ├── test_synthetic_holes.py   # Procedural holes, 3-tone voids, 4-zone placement & zero-heatmap markings
│   ├── test_real_dataset.py      # Real dataset, schema, hybrid, and autolabel test suite
│   ├── test_canonical_loader.py  # Format-independent ingestion & in-memory buffer test suite
│   ├── test_low_res_metrology.py # Scoring compliance & defense-in-depth metrology test suite
│   ├── test_camera_angles.py     # Perspective rectification & tilt test suite
│   ├── test_cli.py               # Unified CLI dispatcher & argument handling test suite
│   └── test_benchmark_low_res.py # Procedural statistical regression benchmark
├── weights/
│   ├── target_scoring_hrnet_1024_best.pt # Production competition model (0.335mm MRE)
│   ├── target_scoring_hrnet_768_best.pt  # Intermediate Stage 3 weights
│   ├── target_scoring_hrnet_512_best.pt  # Intermediate Stage 2 weights
│   └── target_scoring_hrnet_256_best.pt  # Intermediate Stage 1 weights
├── visual_inspection_reports/    # Diagnostic scorecards (PNG) across stages (B-2 Slow Fire & B-3 Timed/Rapid)
│   ├── colab_scorecard_1024_b2.png
│   ├── colab_scorecard_1024_b3.png
│   ├── colab_scorecard_768_b2.png
│   ├── colab_scorecard_768_b3.png
│   ├── colab_scorecard_512_b2.png
│   ├── colab_scorecard_512_b3.png
│   ├── colab_scorecard_256_b2.png
│   └── colab_scorecard_256_b3.png
├── scripts/
│   ├── train_progressive_colab.py # Standalone runner for 4-stage progressive cloud curriculum
│   ├── create_colab_bundle.py    # Packages repository into clean cloud zip bundle
│   └── generate_benchmark_visuals.py
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

Run the full battery of 103 unit, progressive, metrology, and regression tests:

```bash
# Run complete test suite (103 tests):
./target_env/bin/python3 -m unittest discover tests/ -v

# Run procedural holes, 3-tone voids, and zero-heatmap markings tests:
./target_env/bin/python3 -m unittest tests/test_synthetic_holes.py -v

# Run Progressive Resizing dynamic resolution tests:
./target_env/bin/python3 -m unittest tests/test_progressive_training.py -v

# Run real dataset ingestion, schema, and active learning tests:
./target_env/bin/python3 -m unittest tests/test_real_dataset.py -v

# Run format-independent ingestion & in-memory buffer tests:
./target_env/bin/python3 -m unittest tests/test_canonical_loader.py -v

# Run metrology, inward scoring, and defense-in-depth filtering assertions:
./target_env/bin/python3 -m unittest tests/test_low_res_metrology.py -v

# Run unified CLI dispatcher tests:
./target_env/bin/python3 -m unittest tests/test_cli.py -v

# Run oblique camera angle tilt stress tests:
./target_env/bin/python3 -m unittest tests/test_camera_angles.py -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).
