# Project Context: Camera-Independent NRA Target Metrology & Scoring Engine

## 1. Executive Summary & System Architecture

This project is an automated, high-precision computer vision and deep learning metrology engine designed to score official NRA Precision Pistol targets (**NRA B-2** 50 ft Slow Fire and **NRA B-3** 50 ft Timed/Rapid Fire) from arbitrary smartphone photographs. 

The architecture decouples geometric normalisation from neural coordinate regression:
1. **Classical Geometric Rectification:** Automatically locates the aiming black in high-resolution, unconstrained photos, estimates perspective tilt via algebraic ellipse fitting, and warps the target into an exact metric-calibrated canonical plane ($512 \times 512$ at $4.0\text{ px/mm}$).
2. **Deep Continuous Heatmap Regression:** A High-Resolution Feature Pyramid U-Net regresses sub-pixel Gaussian heatmaps for bullet impacts and bullseye centers.
3. **Ballistic Metrology & Inward Scoring:** A Taylor-series sub-pixel peak extractor calculates continuous coordinates, applies official NRA inward scoring rules, and generates a visual diagnostic HUD with group dispersion statistics (Extreme Spread, Mean Radius, Windage/Elevation bias).

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
       │  - Direct least-squares algebraic ellipse fit (Fitzgibbon)│
       │  - 3x3 Homography derivation -> Metric scaling          │
       └────────────────────────────┬─────────────────────────────┘
                                    │ Canonical Crop (512x512 @ 4.0 px/mm)
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
       │ Stage 3: Metrology & Scoring Engine (infer_and_score.py) │
       │  - Caliber-aware spatial Non-Maximum Suppression (NMS)   │
       │  - 2D Quadratic Taylor-series sub-pixel peak refinement  │
       │  - NRA B-2 / B-3 inward scoring tables (effective gauge) │
       │  - Ballistics: Extreme Spread (mm/MOA), Mean Radius, COI │
       │  - Full diagnostic scorecard generation with HUD banner  │
       └──────────────────────────────────────────────────────────┘
```

<!-- STREAMING_CHUNK:Documenting key technical decisions and mathematical models... -->
---

## 2. Technical Decisions & Mathematical Models

### A. Classical Computer Vision & Camera Independence (`target_rectifier.py`)
* **Projective Geometry of Circles:** When an orthogonal circular target of radius $R_{\text{nominal}}$ is observed under perspective tilt, its projection forms an ellipse with semi-major axis $a$ and semi-minor axis $b$.
* **Tilt Angle Formulation:** The off-normal optical pitch angle $\theta$ is derived in closed form:
  $$\theta = \arccos\left(\frac{b}{a}\right)$$
* **Scale Invariance:** Perspective foreshortening acts along the tilt axis; the major axis $2a$ remains unforeshortened. Metric calibration is established directly from official target specifications without reference markers:
  $$S_{\text{raw}} = \frac{a}{R_{\text{nominal}}} \quad [\text{raw pixels per mm}]$$
* **Conic Fitting Formulation:** Uses `cv2.fitEllipseDirect`, which implements the Fitzgibbon-Pilu-Fisher formulation to solve the constrained conic equation $4AC - B^2 = 1$, guaranteeing a mathematically valid ellipse rather than hyperbolic or parabolic degeneracies.
* **Homography Matrix $H$:** Synthesizes 16 conjugate point pairs along the fitted ellipse and projects them onto a flat canonical circle of radius $R_{\text{canon}} = R_{\text{nominal}} \times 4.0\text{ px/mm}$ at $(256, 256)$, computing a $3 \times 3$ projective warp matrix via RANSAC.

### B. Target Specifications (Official NRA Rules)

| Specification | NRA B-2 (50 ft Slow Fire) | NRA B-3 (50 ft Timed & Rapid Fire) |
| :--- | :--- | :--- |
| **Aiming Black** | Rings 7, 8, 9, 10 | Rings 9, 10, X |
| **Black Diameter** | $3.07''$ ($77.978\text{ mm}$) | $3.06''$ ($77.724\text{ mm}$) |
| **Black Radius ($R_{\text{nominal}}$)** | **$38.989\text{ mm}$** | **$38.862\text{ mm}$** |
| **Canonical Black Radius ($4.0\text{ px/mm}$)** | $155.96\text{ px}$ | $155.45\text{ px}$ |
| **Inner Scoring Rings** | X ($5.715\text{ mm}$), 10 ($11.430\text{ mm}$), 9 ($19.558\text{ mm}$), 8 ($28.321\text{ mm}$) | X ($11.430\text{ mm}$), 10 ($22.860\text{ mm}$) |
| **Outer Scoring Rings** | 6 ($52.832\text{ mm}$), 5 ($70.612\text{ mm}$), 4 ($93.091\text{ mm}$) | 8 ($56.642\text{ mm}$), 7 ($77.978\text{ mm}$), 6 ($105.664\text{ mm}$) |

### C. Deep Learning Architecture (`HighResHeatmapUNet`)
* **Spatial Resolution Preservation:** Rather than standard deep classification backbones (e.g., ResNet) that downsample to $1/32$ scale, `HighResHeatmapUNet` uses a base dimension of 32, max-pools to a maximum of $1/8$ scale, and retains a full-resolution stem with skip connections to preserve high-frequency sub-millimeter edge gradients.
* **Activations:** Mish activations ($\text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x))$) throughout all residual convolution blocks to prevent dying ReLU gradients on weak Gaussian tails.
* **Loss Function (Adaptive Wing Loss):**
  $$L(y, \hat{y}) = \begin{cases} \omega \ln\left(1 + \left(\frac{|\hat{y} - y|}{\epsilon}\right)^{\alpha - y}\right) & \text{if } |\hat{y} - y| < \theta \\ A |\hat{y} - y| - C & \text{otherwise} \end{cases}$$
  * Operates with parameters: $\omega = 14.0$, $\theta = 0.5$, $\epsilon = 1.0$, $\alpha = 2.1$.
  * Exerts dynamic exponential gradient curvature on near-peak errors ($\alpha - y$ increases gradient sensitivity where ground truth $y \to 1.0$) while smoothly decaying to linear penalty on large offsets, preventing empty cardboard background pixels from zeroing out gradient propagation.

<!-- STREAMING_CHUNK:Detailing metrology mathematics and subpixel extraction... -->
### D. Metrology & Scoring Mathematics
* **2D Quadratic Taylor-Series Sub-Pixel Extraction:**
  Given an integer local maximum $(x_0, y_0)$ from spatial NMS, continuous coordinates $(x^*, y^*)$ are resolved via 2nd-order expansion:
  $$\Delta \mathbf{p} = - \mathbf{H}^{-1} \nabla f$$
  where $\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} & \frac{\partial f}{\partial y} \end{bmatrix}^T$ is the central-difference gradient and $\mathbf{H}$ is the $2 \times 2$ Hessian matrix of 2nd derivatives evaluated over a $3 \times 3$ neighborhood. Offsets are clamped to $[-0.9, 0.9]$ to prevent unstable extrapolation.
* **Official NRA Inward Scoring Rule:**
  An impact earns the value of a higher-scoring ring if its physical caliber boundary touches or breaks the ring line. With bullet radius $r_{\text{bullet}} = 5.69\text{ mm} / 2 = 2.845\text{ mm}$ and radial distance to impact center $d_{\text{shot}}$, the effective scoring gauge distance is:
  $$d_{\text{gauge}} = \max\left(0, d_{\text{shot}} - r_{\text{bullet}}\right)$$
  The ring label assigned satisfies: $\min \{R_i \mid d_{\text{gauge}} \le R_i\}$.
* **Minutes of Angle (MOA) Calculation:**
  $$\text{MOA} = \frac{\text{Spread}_{\text{mm}}}{\text{Distance}_{\text{yards}} \times \tan(1/60^\circ) \times 914.4} \approx \frac{\text{Spread}_{\text{mm}}}{\text{Distance}_{\text{yards}} \times 0.26598}$$

<!-- STREAMING_CHUNK:Summarizing codebase files and implementations... -->
---

## 3. Codebase File Structure & Implementations

### 1. `target_rectifier.py` (Standalone Geometric Rectifier)
* **`RectificationResult` (Dataclass):** Stores the $512 \times 512$ rectified canonical crop, $3 \times 3$ homography matrix $H$, detected pitch tilt angle, raw and canonical pixels/mm scale, sub-pixel raw center coordinates, ellipse axis lengths, circularity score, and motion blur metric.
* **`TargetRectifier` (Class):**
  * `load_image_with_exif()`: Loads phone photos with Pillow EXIF transposition to prevent 90°/180° rotation bugs.
  * `evaluate_blur_metric()`: Computes $\text{Var}(\nabla^2 I)$ to flag blurry images ($< 75.0$).
  * `suppress_specular_glare()`: Fast Navier-Stokes inpainting on range lighting flash reflections inside dark ink.
  * `_segment_bullseye_contour()`: Multi-scale Otsu and adaptive Gaussian thresholding with morphological closing to bridge bullet holes that rupture the bullseye contour.
  * `process_raw_image()`: End-to-end normalization pipeline with fallback to aspect-preserving centered crop if target black is entirely obscured.
  * `render_diagnostic_panel()`: Visualizes raw image with fitted ellipse axes alongside the rectified canonical plane.

### 2. `infer_and_score.py` (Metrology, Scoring & Overlay Engine)
* **`TargetRingsConfig` (Dataclass):** NRA B-2 and B-3 ring definitions, radii, and scoring tables.
* **`extract_subpixel_peaks()`:** Caliber-aware spatial Non-Maximum Suppression with confidence sorting and 2D Taylor-series refinement. Eliminates low-confidence background noise clusters.
* **`score_target_group()`:** Evaluates inward scoring, group Extreme Spread (ES), Mean Radius, Center of Impact (COI), and Windage/Elevation biases.
* **`render_scoring_inspection_overlay()`:** Renders colored plug gauges (Gold for X, Green for 10, Cyan for 9, Orange for 7–8, Red for <7), ES chord line, semi-transparent heat trace, and a comprehensive dark HUD scoreboard panel.
* **CLI Arguments:** Supports `--checkpoint`, `--image`, `--target` (B-2/B-3), `--distance`, `--shots` (default 10), and `--threshold`.

### 3. `train_heatmap_model.py` (Procedural Training Pipeline)
* **Self-Contained Generator:** Fully embedded procedural target generator (cardboard textures, fluting ridges, fiber noise, irregular bullet holes, folded flaps, doubles, keyholes, and Gaussian heatmaps). Zero external module dependencies.
* **`SyntheticTargetDataset`:** On-the-fly procedural PyTorch dataset alternating between NRA B-2 and B-3 targets.
* **Training Hyperparameters:** Cosine annealing learning rate schedule, gradient norm clipping ($1.0$), and validation Mean Radial Error (MRE) tracking in metric millimeters.

<!-- STREAMING_CHUNK:Documenting prerequisites, environment setup, and troubleshooting... -->
---

## 4. Prerequisites & Environment Setup

### System & Hardware Compatibility

| Platform | Role | Configuration / Recommendations |
| :--- | :--- | :--- |
| **NVIDIA GTX 1060 (6 GB)** | **Training** | **FP32 only** (Pascal CC 6.1 lacks Tensor Cores; FP16 is slow and causes Gaussian underflow). `batch_size = 4`, `num_workers = 2-4`, `canvas_size = 512` or `768`. |
| **WSL2 (Ubuntu)** | **Development OS** | Preferred over native Windows due to POSIX `fork` in `DataLoader` (eliminates Windows `spawn` CPU starvation). Keep code in `/home/<user>/`, **not** `/mnt/c/`. |
| **Raspberry Pi 4 (8 GB)** | **Edge Deployment** | 64-bit OS required (`aarch64`). CPU inference takes $\approx 0.9\text{–}1.3\text{ s}$ per target. Peak RAM $\approx 600\text{ MB}$. |

### Clean Virtual Environment Setup (via `uv`)
Because Python 3.14+ breaks system packages via PEP 668 and lacks pre-built PyTorch CUDA wheels, use standalone Python 3.11:

```bash
# 1. Install uv in WSL2
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2. Create isolated Python 3.11 virtual environment
cd ~/target_scorer
uv venv target_env --python 3.11
source target_env/bin/activate

# 3. Install PyTorch with CUDA 12.1 and computer vision libraries
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install opencv-python numpy pillow
```

---

## 5. Troubleshooting Log & Resolved Issues

1. **PEP 668 Externally Managed Environment & Python 3.14:**
   * *Issue:* System package manager blocked global `pip`, and PyTorch wheels did not exist for Python 3.14.
   * *Resolution:* Deployed `uv` to instantiate an isolated user-space Python 3.11 environment.
2. **`ModuleNotFoundError: No module named 'synthetic_target_generator'`:**
   * *Issue:* Training script attempted an external import of generator components.
   * *Resolution:* Consolidated all cardboard, bullet hole, and heatmap synthesis logic directly into `train_heatmap_model.py`.
3. **`AttributeError: 'list' object has no attribute 'numpy'` in Validation:**
   * *Issue:* `custom_collate_fn` handles variable shot counts per target as a list of tensors, but validation code attempted `.numpy()` on the container list.
   * *Resolution:* Iterated over batch items individually to evaluate coordinate Euclidean distances.
4. **PyTorch Unpickler `AttributeError: Can't get attribute 'TrainConfig'`:**
   * *Issue:* `torch.load` failed when unpickling checkpoints referencing `__main__.TrainConfig` across different scripts.
   * *Resolution:* Mirrored `TrainConfig` inside `infer_and_score.py` and registered it via `torch.serialization.add_safe_globals([TrainConfig])`.
5. **5,800+ Detected "Ghost Shots" on Untrained/Early Models:**
   * *Issue:* A $3 \times 3$ dilation peak finder picked up random 2D background noise ripples hovering around $0.32$.
   * *Resolution:* Upgraded to a caliber-aware radial greedy NMS ($10\text{+ px}$ suppression radius), confidence-ranked filtering, a `--shots 10` limit, and raised threshold to $0.45$.
6. **`ModuleNotFoundError: No module named 'target_rectifier'` & Filename Discrepancies:**
   * *Issue:* Files were named with verbose names while scripts and docs referenced `target_rectifier.py`, `infer_and_score.py`, and `train_heatmap_model.py`.
   * *Resolution:* Created clean compatibility modules and defensive imports ensuring execution works across both conventions.
7. **Homography Target Roll / Spurious Rotation:**
   * *Issue:* `_derive_homography_from_ellipse` forced the major axis to horizontal canonical X, spinning orthogonal/near-flat targets randomly and corrupting Windage and Elevation bias.
   * *Resolution:* Adjusted canonical destination vectors `pt_canon` to preserve basis vectors `u` and `v`, eliminating in-plane target spinning.
8. **Unconstrained Sub-pixel Hessian Inversion:**
   * *Issue:* Inverting Hessian on `abs(det) > 1e-6` allowed saddle points ($\det < 0$) and local minima.
   * *Resolution:* Constrained inversion strictly to negative definite local maxima ($\det > 1e-6$ and $dxx < 0$ and $dyy < 0$).

---

## 6. Unfinished Tasks & Immediate Roadmap

```
[Sprint 1: Rectification Engine]      --> COMPLETED (target_rectifier.py)
[Sprint 2: Dual Target Metrology]     --> COMPLETED (NRA B-2 and B-3 tables integrated)
[Sprint 3: Training Augmentations]    --> COMPLETED (Affine translation, scale jitter, photometric noise)
[Sprint 4: Full Retraining]           --> PENDING (Run 15-20 epochs on GTX 1060)
[Sprint 5: Mobile / Edge Export]      --> PENDING (ONNX conversion for Pi 4)
```

### Immediate Next Tasks

1. **Implement Synthetic Training Augmentations (Sprint 3):**
   * *Objective:* Even with `TargetRectifier`, real-world captures have residual tilt ($\pm 3^\circ$) and minor cropping offsets ($\pm 15\text{ px}$).
   * *Action:* Add random translation ($\pm 32\text{ px}$), scale jitter ($0.92\text{–}1.08$), random 4-point homography tilt, and cardboard illumination noise to `SyntheticTargetDataset` in `train_heatmap_model.py`.
2. **Run Production Retraining (Sprint 4):**
   * *Objective:* Drive Mean Radial Error from initial $0.828\text{ mm}$ test down to competition-grade $< 0.15\text{ mm}$ and eliminate all background noise.
   * *Settings:* `train_samples_per_epoch = 400`, `val_samples_per_epoch = 50`, `epochs = 15` (Runtime: $\approx 22\text{ minutes}$ on GTX 1060 6GB).
3. **Real-World Target Validation:**
   * Test `infer_and_score.py` against real camera photos taken under diverse angles ($0^\circ, 15^\circ, 30^\circ, 45^\circ$) and lighting conditions.
4. **Edge Deployment to Raspberry Pi 4:**
   * Export `HighResHeatmapUNet` to ONNX format.
   * Benchmark execution with `onnxruntime` utilizing ARM NEON SIMD to target $\approx 300\text{ ms}$ inference on the Pi 4.
```