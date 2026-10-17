# Project Context: Camera-Independent NRA Target Metrology & Scoring Engine

## 1. Executive Summary & System Architecture

This project is an automated, high-precision computer vision and deep learning metrology engine designed to score official NRA Precision Pistol targets (**NRA B-2** 50 ft Slow Fire and **NRA B-3** 50 ft Timed/Rapid Fire) from arbitrary smartphone photographs. 

The architecture decouples geometric normalization from neural coordinate regression:
1. **Classical Geometric Rectification:** Automatically locates the aiming black in high-resolution, unconstrained photos, estimates perspective tilt via algebraic ellipse fitting, and warps the target into an exact metric-calibrated canonical plane supporting **$512 \times 512$**, **$800 \times 800$**, or **$1024 \times 1024$** resolutions at $4.0\text{ px/mm}$.
2. **Deep Continuous Heatmap Regression:** A High-Resolution Feature Pyramid U-Net regresses sub-pixel Gaussian heatmaps for bullet impacts and bullseye centers.
3. **Ballistic Metrology & Inward Scoring:** A Taylor-series sub-pixel peak extractor calculates continuous coordinates, applies official NRA inward scoring rules, and generates a visual diagnostic HUD with group dispersion statistics (Extreme Spread, Mean Radius, Windage/Elevation bias).
4. **Privacy & Redaction:** Automated competitor registration handwriting masking (`--redact-pii`) for competition compliance.

```
                      [ Raw Smartphone Photograph ]
       (Arbitrary resolution, perspective tilt up to ~45°, any aspect ratio)
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 1: TargetRectifier (Classical CV Engine)          │
       │  - EXIF orientation auto-rotation (Pillow / OpenCV)     │
       │  - Laplacian variance motion blur check (> 75.0)        │
       │  - Specular glare suppression (Navier-Stokes inpainting)│
       │  - Pyramid bullseye segmentation & morphological close   │
       │  - Direct least-squares algebraic ellipse fit (Fitzgib.)│
       │  - 3x3 Homography derivation -> Metric scaling          │
       └────────────────────────────┬─────────────────────────────┘
                                    │ Canonical Crop (512x512, 800x800, or 1024x1024 @ 4.0 px/mm)
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 2: HighResHeatmapUNet (PyTorch Deep Learning)     │
       │  - Dual-channel regression (Ch0: Bullets, Ch1: Bullseye) │
       │  - Full-resolution stem & high-frequency skip connections│
       │  - Trained via Adaptive Wing Loss (A-Wing)              │
       └────────────────────────────┬─────────────────────────────┘
                                    │ Continuous 2D Heatmaps
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 3: Metrology & Scoring Engine (infer.py)          │
       │  - Caliber-aware spatial Non-Maximum Suppression (NMS)   │
       │  - 2D Quadratic Taylor-series sub-pixel peak refinement  │
       │  - NRA B-2 / B-3 inward scoring tables (effective gauge) │
       │  - Ballistics: Extreme Spread (mm/MOA), Mean Radius, COI │
       │  - Full diagnostic scorecard generation with HUD banner  │
       └──────────────────────────────────────────────────────────┘
```

---

## 2. Technical Decisions & Mathematical Models

### A. Classical Computer Vision & Camera Independence (`rectifier.py`)
* **Projective Geometry of Circles:** When an orthogonal circular target of radius $R_{\text{nominal}}$ is observed under perspective tilt, its projection forms an ellipse with semi-major axis $a$ and semi-minor axis $b$.
* **Tilt Angle Formulation:** The off-normal optical pitch angle $\theta$ is derived in closed form:
  $$\theta = \arccos\left(\frac{b}{a}\right)$$
* **Scale Invariance:** Perspective foreshortening acts along the tilt axis; the major axis $2a$ remains unforeshortened. Metric calibration is established directly from official target specifications without reference markers:
  $$S_{\text{raw}} = \frac{a}{R_{\text{nominal}}} \quad [\text{raw pixels per mm}]$$
* **Conic Fitting Formulation:** Uses `cv2.fitEllipseDirect`, which implements the Fitzgibbon-Pilu-Fisher formulation to solve the constrained conic equation $4AC - B^2 = 1$, guaranteeing a mathematically valid ellipse rather than hyperbolic or parabolic degeneracies.
* **Homography Matrix $H$:** Synthesizes 16 conjugate point pairs along the fitted ellipse and projects them onto a flat canonical circle of radius $R_{\text{canon}} = R_{\text{nominal}} \times \text{pixels\_per\_mm}$ at $(c_{\text{mid}}, c_{\text{mid}})$, computing a $3 \times 3$ projective warp matrix via RANSAC.

### B. Target Specifications (Official NRA Rules)

| Specification | NRA B-2 (Rule 4.2: 50 ft Slow Fire) | NRA B-3 (Rule 4.3: 50 ft Timed & Rapid Fire) |
| :--- | :--- | :--- |
| **Aiming Black** | Rings **7, 8, 9, 10** | Rings **9, 10, X** |
| **Black Diameter** | $3.07''$ ($77.978\text{ mm}$) | $3.06''$ ($77.724\text{ mm}$) |
| **Black Radius ($R_{\text{nominal}}$)** | **$38.989\text{ mm}$** | **$38.862\text{ mm}$** |
| **Canonical Black Radius ($4.0\text{ px/mm}$)** | $155.96\text{ px}$ | $155.45\text{ px}$ |
| **Innermost Ring** | **10-Ring** ($0.90'' = 22.86\text{ mm}$ dia, $11.43\text{ mm}$ rad) | **X-Ring** ($0.90'' = 22.86\text{ mm}$ dia, $11.43\text{ mm}$ rad) |
| **X-Ring Present?** | **No** (Official Rule 4.2 has no X-ring at 50 ft Slow Fire) | **Yes** ($11.43\text{ mm}$ radius) |
| **Inner Scoring Rings** | 10 ($11.430\text{ mm}$), 9 ($19.558\text{ mm}$), 8 ($28.321\text{ mm}$) | X ($11.430\text{ mm}$), 10 ($22.860\text{ mm}$) |
| **Outer Scoring Rings** | 6 ($52.832\text{ mm}$), 5 ($70.612\text{ mm}$), 4 ($93.091\text{ mm}$) | 8 ($56.642\text{ mm}$), 7 ($77.978\text{ mm}$), 6 ($105.664\text{ mm}$) |
| **Score Format** | **`Score / Max`** (e.g. `96 / 100`) | **`Score - X / Max`** (e.g. `100 - 7X / 100`) |

### C. Multi-Resolution Canonical Canvases

The engine supports dynamic canvas configuration to balance execution speed, memory footprint, and physical field of view:

1. **$512 \times 512\text{ px}$ ($4.0\text{ px/mm}$, $\text{FOV} = 128\text{ mm}$):**
   * Captures the aiming black and immediate surrounding rings (down to 7-ring on B-3, 6-ring on B-2).
   * Ideal for rapid prototyping and low-memory devices.
2. **$800 \times 800\text{ px}$ ($4.0\text{ px/mm}$, $\text{FOV} = 200\text{ mm}$):**
   * Expands the observable field of view to capture 5-ring impacts and wide flyers.
   * Optimum resolution for NVIDIA T4 cloud inference.
3. **$1024 \times 1024\text{ px}$ ($4.0\text{ px/mm}$, $\text{FOV} = 256\text{ mm}$):**
   * Captures all scoring rings (down to 4-ring on B-2, 6-ring on B-3), full target card edges, and header text.
   * Delivers **$0.335\text{ mm}$ sub-pixel precision** ($< 1.35\text{ px}$) and $98.9\%$ recall.
   * Standard for official match evaluation.

### D. Deep Learning Architecture (`HighResHeatmapUNet`)
* **Spatial Resolution Preservation:** Uses a base dimension of 32, max-pools to a maximum of $1/8$ scale, and retains a full-resolution stem with skip connections to preserve high-frequency sub-millimeter edge gradients.
* **Activations:** Mish activations ($\text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x))$) throughout all residual convolution blocks to prevent dying ReLU gradients on weak Gaussian tails.
* **Loss Function (Adaptive Wing Loss):**
  $$L(y, \hat{y}) = \begin{cases} \omega \ln\left(1 + \left(\frac{|\hat{y} - y|}{\epsilon}\right)^{\alpha - y}\right) & \text{if } |\hat{y} - y| < \theta \\ A |\hat{y} - y| - C & \text{otherwise} \end{cases}$$
  Operates with parameters: $\omega = 14.0$, $\theta = 0.5$, $\epsilon = 1.0$, $\alpha = 2.1$.
* **Composite Fitness Selection:**
  $$\text{Score} = \text{Match Rate (\%)} - 10.0 \times \text{MRE (mm)}$$
  Prevents checkpoints from over-optimizing for low MRE at the cost of dropped bullet holes.

### E. Metrology & Scoring Mathematics
* **2D Quadratic Taylor-Series Sub-Pixel Extraction:**
  Given an integer local maximum $(x_0, y_0)$ from spatial NMS, continuous coordinates $(x^*, y^*)$ are resolved via 2nd-order expansion:
  $$\Delta \mathbf{p} = - \mathbf{H}^{-1} \nabla f$$
  where $\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} \end{bmatrix}^T$ is the central-difference gradient and $\mathbf{H}$ is the $2 \times 2$ Hessian matrix of 2nd derivatives evaluated over a $3 \times 3$ neighborhood.
* **Official NRA Inward Scoring Rule:**
  An impact earns the value of a higher-scoring ring if its physical caliber boundary touches or breaks the ring line:
  $$d_{\text{gauge}} = \max\left(0, d_{\text{shot}} - r_{\text{bullet}}\right)$$
  The ring label assigned satisfies: $\min \{R_i \mid d_{\text{gauge}} \le R_i\}$.

---

## 3. Codebase File Structure & Modular Implementations

```
src/target_scorer/
├── config.py             # TargetConfig, TrainConfig, TargetRingsConfig
├── cli.py                # Unified CLI entrypoint (target-scorer)
├── infer.py              # Inference, inward scoring, HUD overlay, and demo generation
├── train.py              # Procedural training pipeline with warm-start & composite fitness
├── colab_runner.py       # Automated cloud GPU orchestrator (Google Colab CLI)
├── geometry/
│   ├── canonical_loader.py # Format-agnostic ingestion (HEIC, RAW, buffers, EXIF, sRGB CMS)
│   └── rectifier.py      # Classical ellipse fitting & homography rectification
├── metrology/
│   ├── peaks.py          # 2D Taylor-series sub-pixel peak extraction
│   ├── scoring.py        # NRA B-2 / B-3 inward scoring & ballistics math
│   └── visualization.py  # Diagnostic scorecard overlay & PII redaction
├── models/
│   ├── unet.py           # HighResHeatmapUNet & ConvBlock
│   └── loss.py           # Adaptive Wing Loss
└── synthetic/
    ├── generator.py      # Procedural cardboard, bullet holes, handwriting, & scorer marks
    └── dataset.py        # PyTorch dataset with real-world augmentations
```

---

## 4. Execution & Training Guide

### A. Scoring Inference (`target-scorer infer`)
```bash
# Score an Apple iPhone HEIC photograph (auto-corrected for EXIF rotation & Display P3 gamut)
target-scorer infer --image target.heic --target B-2 --canvas-size 1024 --shots 10

# Score a professional Camera RAW / Apple ProRAW DNG file
target-scorer infer --image target.dng --target B-2 --canvas-size 1024

# Score a lossless WebP / PNG photo with competitor registration PII redacted
target-scorer infer --image target.webp --target B-3 --canvas-size 1024 --redact-pii

# Stream binary image data directly from shell pipes or curl via stdin ('-')
cat target.heic | target-scorer infer --image - --target B-2 --canvas-size 1024

# Generate deterministic 10-shot competition scorecards
target-scorer infer --target B-2 --canvas-size 1024 --output colab_scorecard_1024_b2.png
target-scorer infer --target B-3 --canvas-size 1024 --output colab_scorecard_1024_b3.png
```

```python
# Real-Time In-Memory Buffer Inference (Raspberry Pi Picamera2 / Video Capture Loop)
from target_scorer import run_inference_and_scoring

shots, metrics = run_inference_and_scoring(
    checkpoint_path="weights/target_scoring_hrnet_1024_best.pt",
    image_path=frame_bgr,  # In-memory np.ndarray, bytes, or file path
    target_type="B-2",
    canvas_size=1024,
    max_shots=10
)
```

### B. Cloud GPU Training via Colab CLI (`target-scorer colab-train`)
```bash
# 1-Click Warm-start fine-tuning on NVIDIA A100 (1024x1024, ~6 min)
target-scorer colab-train \
    --canvas-size 1024 \
    --scale 4.0 \
    --epochs 12 \
    --batch-size 8 \
    --lr 1.5e-4 \
    --sigma 3.2 \
    --gpu A100 \
    --initial-weights weights/target_scoring_hrnet_1024_best.pt

# Training from scratch on A100 (45 epochs)
target-scorer colab-train \
    --canvas-size 1024 \
    --scale 4.0 \
    --epochs 45 \
    --batch-size 8 \
    --lr 3.5e-4 \
    --sigma 3.2 \
    --gpu A100

# Training at 800x800 on NVIDIA T4
target-scorer colab-train \
    --canvas-size 800 \
    --scale 4.0 \
    --epochs 20 \
    --batch-size 8 \
    --lr 2.5e-4 \
    --sigma 2.8 \
    --gpu T4

# Hybrid real target training on A100 (Option A)
target-scorer colab-train \
    --canvas-size 1024 \
    --scale 4.0 \
    --epochs 15 \
    --batch-size 8 \
    --lr 1.0e-4 \
    --gpu A100 \
    --real-data-dir datasets/calibration_targets \
    --real-ratio 0.35 \
    --initial-weights weights/target_scoring_hrnet_1024_best.pt

# Session management & active job monitoring
target-scorer colab-train --list-sessions
target-scorer colab-train --stop-session target-train-1024-12345
```

### C. Cloud-Accelerated Auto-Labeling (`target-scorer colab-autolabel`)
```bash
# Auto-label a directory of target photos on cloud GPU (T4/A100)
target-scorer colab-autolabel \
    --input-dir /path/to/phone_targets \
    --output-dir datasets/autolabeled_targets \
    --target B-2 \
    --gpu T4
```

### D. Local Workstation Training (`target-scorer train`)
```bash
# Local fine-tuning on GTX 1060 (batch size 2 to fit 6GB VRAM)
target-scorer train \
    --canvas-size 1024 \
    --scale 4.0 \
    --epochs 12 \
    --batch-size 2 \
    --lr 1.5e-4 \
    --sigma 3.2 \
    --initial-weights weights/target_scoring_hrnet_1024_best.pt \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt

# # Pure real target fine-tuning (Option B: Rapid domain adaptation to range textures)
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

# Hybrid blended training (Option A: 70% synthetic + 30% real)
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

# Fast 512x512 baseline training
target-scorer train --canvas-size 512 --epochs 15 --batch-size 4
```

### E. Assisted Auto-Labeling & Active Learning (`target-scorer autolabel`)
```bash
# Auto-label a single smartphone photograph (unwarps, detects shots, renders review card)
target-scorer autolabel \
    --image /path/to/phone_target.jpg \
    --target B-2 \
    --output-dir datasets/real_targets \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --threshold 0.40

# Batch auto-label an entire directory of target photographs
target-scorer autolabel \
    --input-dir /path/to/range_photos/ \
    --target B-3 \
    --output-dir datasets/real_targets \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --threshold 0.40
```

### F. Standalone Geometric Rectification (`target-scorer rectify`)
```bash
# Rectify an Apple iPhone HEIC photo (auto-corrects EXIF orientation and Display P3 gamut)
target-scorer rectify --image target.heic --target B-2 --canvas-size 1024 --scale 4.0 --output rectified.png --debug debug.png

# Rectify a professional Camera RAW / Apple ProRAW DNG file
target-scorer rectify --image target.dng --target B-3 --canvas-size 1024 --scale 4.0 --output rectified.png

# Stream binary image data directly from shell pipes or curl via stdin ('-')
cat target.heic | target-scorer rectify --image - --target B-2 --canvas-size 1024 --output rectified.png
```

```python
# Real-Time In-Memory Buffer Rectification (Raspberry Pi Picamera2 / Video Capture Loop)
from target_scorer import TargetRectifier

rectifier = TargetRectifier(target_type="B-2", canonical_dim=1024, pixels_per_mm=4.0)
result = rectifier.rectify(frame_bgr)  # Accepts np.ndarray directly
canonical_crop = result.canonical_crop
```

### G. Python Environment Setup via `uv` (PyTorch & Platform Compatibility)

#### Why `uv` is Required
1. **PyTorch Wheel Compatibility Lag:** Official PyTorch pre-compiled binaries and CUDA distributions (e.g., `cu121`, `cu124`) lag behind modern bleeding-edge Python releases (Python 3.13, 3.14+). Attempting to install PyTorch on bleeding-edge Python yields:
   ```text
   ERROR: Could not find a version that satisfies the requirement torch (from versions: none)
   ERROR: No matching distribution found for torch
   ```
2. **PEP 668 ("Externally Managed Environments"):** Modern Linux operating systems (Ubuntu 23.04+, Debian 12+, Fedora) intentionally disallow direct `pip install` against system Python packages.
3. **Standalone Interpreter Isolation:** [`uv`](https://github.com/astral-sh/uv) (built in Rust) automatically fetches and isolates a dedicated, compatible **Python 3.11** interpreter without requiring system package manager (`apt`/`dnf`) overrides or `root` permissions.

#### Step-by-Step Setup
```bash
# 1. Install uv (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # Or source ~/.zshrc (On Windows PowerShell: irm https://astral.sh/uv/install.ps1 | iex)

# 2. Create project directory, download Python 3.11, and create isolated virtual environment
mkdir target_scorer
cd target_scorer
uv python install 3.11
uv venv target_env --python 3.11
source target_env/bin/activate  # On Windows: target_env\Scripts\activate

# 3. Install PyTorch with Hardware Acceleration
# For NVIDIA GPU (CUDA 12.1 - Recommended for Linux / WSL2):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# (For CPU Only: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu)
# (For Apple Silicon: uv pip install torch torchvision)

# 4. Install target_scorer in editable mode
uv pip install -e .

# 5. Verify CUDA Hardware Acceleration & Python Version
python -c "import sys, torch; print(f'Python: {sys.version.split()[0]} | CUDA: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 5. Completed Sprints & Roadmap Status

```
[Sprint 1: Rectification Engine]            --> COMPLETED (geometry/rectifier.py)
[Sprint 2: Dual Target Metrology]           --> COMPLETED (NRA B-2 & B-3 Rulebook compliance)
[Sprint 3: Training Augmentations]          --> COMPLETED (Synthetic handwriting & fluting noise)
[Sprint 4: High-Res 1024x1024 Retraining]   --> COMPLETED (98.9% match, 0.335mm MRE on A100)
[Sprint 5: Privacy & PII Masking]           --> COMPLETED (--redact-pii margin masking)
[Sprint 6: Cloud Orchestration]             --> COMPLETED (Colab CLI automated runner)
[Sprint 7: Format-Agnostic Ingestion]       --> COMPLETED (HEIC, RAW/DNG, Picamera2 memory buffers)
[Sprint 8: Real Target Training & Tooling]  --> COMPLETED (target_scorer/data/, autolabel, hybrid dataset)
[Sprint 9: Edge Deployment]                 --> PENDING (ONNX conversion for Pi 4)
```