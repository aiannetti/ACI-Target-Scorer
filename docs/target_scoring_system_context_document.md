# Project Context: Camera-Independent NRA Target Metrology & Scoring Engine

## 1. Executive Summary & System Architecture

This project is an automated, high-precision computer vision and deep learning metrology engine designed to score official NRA Precision Pistol targets (**NRA B-2** 50 ft Slow Fire and **NRA B-3** 50 ft Timed/Rapid Fire) from arbitrary smartphone photographs. 

The architecture decouples geometric normalization from neural coordinate regression:
1. **Classical Geometric Rectification:** Automatically locates the aiming black in high-resolution, unconstrained photos, estimates perspective tilt via algebraic ellipse fitting, and warps the target into an exact metric-calibrated canonical plane adhering to the **Harmonic 256-Grid Progression** ($256 \times 256$, $512 \times 512$, $768 \times 768$, and competition-standard **$1024 \times 1024$** at $4.0\text{ px/mm}$).
2. **Deep Continuous Heatmap Regression:** A High-Resolution Feature Pyramid U-Net ([`HighResHeatmapUNet`](file:///home/antho/target_scorer/src/target_scorer/models/unet.py)) regresses sub-pixel Gaussian heatmaps for bullet impacts (Channel 0) and bullseye centers (Channel 1).
3. **Ballistic Metrology & Inward Scoring:** A 2D Taylor-series sub-pixel peak extractor calculates continuous coordinates, applies multi-layer defense-in-depth spurious peak filtering (void verification, ink chroma tests, line aspect ratio, seal/header masking), applies official NRA Rule 14.3 inward break scoring, and generates a visual diagnostic HUD with group dispersion statistics (Extreme Spread, Mean Radius, Windage/Elevation bias).
4. **Privacy & Redaction:** Automated competitor registration handwriting masking (`--redact-pii`) for competition compliance.
5. **Interactive Desktop GUI & Virtual Loupe:** [`annotation_editor.py`](file:///home/antho/target_scorer/src/target_scorer/gui/annotation_editor.py) provides real-time point-and-click marker adjustments and an optical **Eagle Eye Scoring Gauge** dialog simulating physical 4x–12x scoring magnifiers.

```
                      [ Raw Smartphone Photograph / Buffer ]
        (Apple HEIC, ProRAW DNG, Ultra HDR, JPEG, PNG, or Picamera2 RAM)
                                     │
                                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │ Stage 1: TargetRectifier (Classical CV Engine)          │
        │  - Universal ingestion (HEIC, RAW, buffers, EXIF, sRGB) │
        │  - Laplacian variance motion blur check (> 75.0)        │
        │  - Specular glare suppression (Navier-Stokes inpainting)│
        │  - Pyramid bullseye segmentation & morphological close   │
        │  - Direct least-squares algebraic ellipse fit (Fitzgib.)│
        │  - Roll-invariant 3x3 Homography -> Metric calibration   │
        └────────────────────────────┬─────────────────────────────┘
                                     │ Canonical Crop (1024x1024 @ 4.0 px/mm)
                                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │ Stage 2: HighResHeatmapUNet (PyTorch Deep Learning)     │
        │  - Fully convolutional dynamic spatial resizing         │
        │  - Dual-channel regression (Ch0: Bullets, Ch1: Bullseye) │
        │  - Full-resolution stem & high-frequency skip merges    │
        │  - Trained via Adaptive Wing Loss (A-Wing)              │
        │  - Harmonic 256-Grid Progressive Resizing (256->1024)   │
        └────────────────────────────┬─────────────────────────────┘
                                     │ Continuous 2D Heatmaps
                                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │ Stage 3: Metrology & Scoring Engine (infer.py / peaks.py)│
        │  - Caliber-aware spatial Non-Maximum Suppression (NMS)   │
        │  - 2D Quadratic Taylor-series sub-pixel peak refinement  │
        │  - Defense-in-depth spurious peak filtering:             │
        │    * Ink chroma tests (blue/violet/red handwriting)     │
        │    * 1D stroke line aspect ratio filter (> 2.2)         │
        │    * Bullet void penetration verification               │
        │    * Official NRA seal & header exclusion zones         │
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

### C. Physical Metrology Grounding & Harmonic 256-Grid Progression
Rather than arbitrary dimensions (such as 800 violating 32-warp alignment), canonical resolutions are grounded directly in physical metrology:
* **Physical Cardboard Size:** Official NRA targets are $10.5 \times 10.5\text{ in}$ ($266.7 \times 266.7\text{ mm}$).
* **Bullet Hole & Lead Wipe Dimensions:** .22 LR bullet diameter is $5.69\text{ mm}$, surrounded by a $0.2\text{--}0.5\text{ mm}$ lead wipe grease ring.
* **Rule 14.3 Precision Bound:** Inward break line touches require resolving $\pm 0.08\text{ mm}$ ($3.15\text{ mils}$).
* **Sub-Pixel Derivation:** At $4.0\text{ px/mm}$ ($0.250\text{ mm/px}$), 2D Taylor-series quadratic interpolation achieves an MRE floor of $\le 0.0625\text{ mm} < 0.08\text{ mm}$. A $256\text{ mm}$ FOV at $4.0\text{ px/mm}$ requires a $1024 \times 1024$ canvas.

The **Harmonic 256-Grid Progression** enforces uniform step properties ($+256\text{ px}$ dimension and $+1.0\text{ px/mm}$ scale per stage):

| Stage | Resolution ($\text{px}$) | Metric Scale | Gaussian $\sigma$ | Recommended Batch | Physical Metrology Function |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | $256 \times 256$ | $1.0\text{ px/mm}$ | $1.6\text{ px}$ | 16 (GPU) | **Macro Floor:** Coarse card orientation and bullseye segmentation (> 900 img/s). |
| **Stage 2** | $512 \times 512$ | $2.0\text{ px/mm}$ | $2.2\text{ px}$ | 12 (GPU) | **Ring Tracking:** Scoring ring edge envelopes and macro grease ring tracking. |
| **Stage 3** | $768 \times 768$ | $3.0\text{ px/mm}$ | $2.8\text{ px}$ | 8 (GPU) | **Double Disentanglement:** Warp-aligned ($24 \times 32$); resolves overlapping double/triple holes. |
| **Stage 4** | **$1024 \times 1024$** | **$4.0\text{ px/mm}$** | **$3.2\text{ px}$** | **8 (A100) / 2 (GTX)**| **Competition Standard:** Sub-pixel error $< 0.08\text{ mm}$ MRE; Rule 14.3 tangent precision. |

### D. Deep Learning Architecture (`HighResHeatmapUNet`)
* **Spatial Resolution Preservation:** Uses a base dimension of 32, max-pools to a maximum of $1/8$ scale, and retains a full-resolution stem with skip connections to preserve high-frequency sub-millimeter edge gradients.
* **Fully Convolutional:** Zero dense/linear layers, allowing seamless dynamic spatial resizing across $256, 512, 768, 1024\text{ px}$ without resetting weights.
* **Activations:** Mish activations ($\text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x))$) throughout all residual convolution blocks to prevent dying ReLU gradients on weak Gaussian tails.
* **Loss Function (Adaptive Wing Loss):**
  $$L(y, \hat{y}) = \begin{cases} \omega \ln\left(1 + \left(\frac{|\hat{y} - y|}{\epsilon}\right)^{\alpha - y}\right) & \text{if } |\hat{y} - y| < \theta \\ A |\hat{y} - y| - C & \text{otherwise} \end{cases}$$
  Operates with parameters: $\omega = 14.0$, $\theta = 0.5$, $\epsilon = 1.0$, $\alpha = 2.1$, foreground weight $= 10.0$.
* **Composite Fitness Selection:**
  $$\text{Score} = \text{Match Rate (\%)} - 10.0 \times \text{MRE (mm)}$$

### E. Metrology, Sub-Pixel Extraction, & Inward Scoring
* **2D Quadratic Taylor-Series Sub-Pixel Extraction (`peaks.py`):**
  Given an integer local maximum $(x_0, y_0)$ from spatial NMS, continuous coordinates $(x^*, y^*)$ are resolved via 2nd-order expansion:
  $$\Delta \mathbf{p} = - \mathbf{H}^{-1} \nabla f$$
  where $\nabla f$ is the central-difference gradient and $\mathbf{H}$ is the $2 \times 2$ Hessian matrix evaluated over a $3 \times 3$ neighborhood.
* **Defense-in-Depth Spurious Peak Filtering (`peaks.py`):**
  1. *Border Exclusion:* Rejects peaks within $4\text{ mm}$ ($16\text{ px}$) of canvas borders.
  2. *Header Text Exclusion:* Rejects top printed registration text ($y < 65\text{ px} \cdot sf$).
  3. *Official NRA Seal Exclusion:* Protects lower-left margin seal zone ($x \in [50, 260]\cdot sf, y \in [870, 1024]\cdot sf$).
  4. *Ink Chroma Tests:* Detects blue/violet ballpoint pen ink ($\text{mean}(B-G) > 7.0$) and red grading pen ink ($\text{mean}(R-B) > 25.0 \land \text{mean}(R-G) > 20.0$).
  5. *1D Stroke Line Aspect Ratio:* Rejects elongated pen strokes with bounding box aspect ratio $> 2.2$.
  6. *Authentic Bullet Void Penetration Verification:* Validates dark voids, tan backer voids, or scanner white voids surrounded by dark lead wipe grease rings.
* **Official NRA Inward Scoring Rule (`scoring.py`):**
  An impact earns the value of a higher-scoring ring if its physical caliber boundary touches or breaks the ring line:
  $$d_{\text{gauge}} = \max\left(0, d_{\text{shot}} - r_{\text{bullet}}\right)$$
  The ring label assigned satisfies: $\min \{R_i \mid d_{\text{gauge}} \le R_i\}$.
* **Input Validation:**
  Rigorous assertions enforce finite numerical coordinates (`math.isfinite(bullet_diameter_mm)`, `math.isfinite(bull_x)`, `math.isfinite(sx)`).

---

## 3. Codebase File Structure & Modular Implementations

```
src/target_scorer/
├── config.py             # TargetConfig, TrainConfig, TargetRingsConfig
├── cli.py                # Unified CLI entrypoint (target-scorer)
├── infer.py              # Inference, inward scoring, HUD overlay, FileNotFoundError validation
├── train.py              # Single-resolution & Progressive Resizing training pipeline
├── colab_runner.py       # Automated cloud GPU orchestrator (Google Colab CLI)
├── geometry/
│   ├── canonical_loader.py # Format-agnostic ingestion (HEIC, RAW, buffers, EXIF, sRGB CMS)
│   └── rectifier.py      # Classical ellipse fitting & homography rectification
├── gui/
│   └── annotation_editor.py # Interactive GUI & Eagle Eye optical scoring loupe
├── metrology/
│   ├── peaks.py          # 2D Taylor-series peaks & defense-in-depth spurious filtering
│   ├── scoring.py        # NRA B-2 / B-3 inward scoring, math.isfinite validation, ballistics
│   └── visualization.py  # Diagnostic scorecard overlay, 2D/4D channel auto-conversion, PII redaction
├── models/
│   ├── unet.py           # Fully convolutional HighResHeatmapUNet & ConvBlock
│   └── loss.py           # Adaptive Wing Loss
└── synthetic/
    ├── canvas.py         # Cardboard textures, standard bullseyes, continuous Gaussian heatmaps
    ├── holes.py          # Bullet stamping, ragged fibers, 3-tone voids, 4-zone stratified placement
    ├── markings.py       # Official NRA seal, printed headers, looped handwriting & scorer marks (0.0 hmap)
    ├── generator.py      # Facade re-exporting canvas, holes, and markings routines
    └── dataset.py        # Procedural PyTorch dataset with real-world augmentations
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

### B. Cloud GPU Progressive Training via Colab CLI (`target-scorer colab-progressive-train`)
```bash
# 1-Click Multi-Stage Progressive Cloud Training (Harmonic 256-Grid on NVIDIA A100)
target-scorer colab-progressive-train \
    --gpu A100 \
    --real-data-dir private/data/real_targets \
    --real-ratio 0.35

# Standalone execution script
python scripts/train_progressive_colab.py A100

# Single-resolution training on A100 (1024x1024)
target-scorer colab-train \
    --canvas-size 1024 \
    --scale 4.0 \
    --epochs 15 \
    --batch-size 8 \
    --lr 1.0e-4 \
    --gpu A100 \
    --real-data-dir private/data/real_targets \
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
    --gpu T4 \
    --redact-pii
```

### D. Local Workstation Training (`target-scorer train`)
```bash
# Harmonic 256-Grid Progressive Resizing locally (256 -> 512 -> 768 -> 1024)
target-scorer train \
    --progressive \
    --real-data-dir private/data/real_targets \
    --real-ratio 0.35 \
    --checkpoint-dir weights \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt

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

# Pure real target fine-tuning (Option B: Rapid domain adaptation to range textures)
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

### E. Interactive Annotation GUI (`target-scorer annotate`)
```bash
# Launch interactive GUI on auto-labeled targets
target-scorer annotate private/data/real_targets
```

### F. Assisted Auto-Labeling & Active Learning (`target-scorer autolabel`)
```bash
# Auto-label a single smartphone photograph (unwarps, detects shots, renders review card)
target-scorer autolabel \
    --image /path/to/phone_target.jpg \
    --target B-2 \
    --output-dir private/data/real_targets \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --threshold 0.40

# Batch auto-label an entire directory of target photographs with PII redaction
target-scorer autolabel \
    --input-dir /path/to/range_photos/ \
    --target B-3 \
    --output-dir private/data/real_targets \
    --checkpoint weights/target_scoring_hrnet_1024_best.pt \
    --threshold 0.40 \
    --redact-pii
```

### G. Standalone Geometric Rectification (`target-scorer rectify`)
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

### H. Python Environment Setup via `uv` (PyTorch & Platform Compatibility)

#### Why `uv` is Required
1. **PyTorch Wheel Compatibility Lag:** Official PyTorch pre-compiled binaries and CUDA distributions (e.g., `cu121`, `cu124`) lag behind modern bleeding-edge Python releases (Python 3.13, 3.14+).
2. **PEP 668 ("Externally Managed Environments"):** Modern Linux operating systems intentionally disallow direct `pip install` against system Python packages.
3. **Standalone Interpreter Isolation:** [`uv`](https://github.com/astral-sh/uv) (built in Rust) automatically fetches and isolates a dedicated, compatible **Python 3.11** interpreter without requiring system package manager (`apt`/`dnf`) overrides or `root` permissions.

#### Step-by-Step Setup
```bash
# 1. Install uv (Linux/macOS)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2. Create isolated Python 3.11 environment
uv python install 3.11
uv venv target_env --python 3.11
source target_env/bin/activate  # On Windows: target_env\Scripts\activate

# 3. Install PyTorch with Hardware Acceleration (CUDA 12.1)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install target_scorer in editable development mode
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
[Sprint 9: Interactive Annotation GUI]      --> COMPLETED (annotation_editor.py, Eagle Eye Loupe)
[Sprint 10: Progressive Resizing Pipeline]  --> COMPLETED (Harmonic 256-Grid curriculum 256->1024)
[Sprint 11: Modularization & Defense-in-Depth] --> COMPLETED (canvas/holes/markings split, peak filtering)
```