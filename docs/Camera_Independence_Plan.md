# Production Plan: 100% Camera-Independent Target Scoring

## 1. Executive Summary & Problem Diagnosis

Smartphone target photographs suffer from significant variations in capture geometry:
1. **Geometric Scale Invariance Failure:** Hardcoded physical scales break when shooting distances vary. A 1-pixel detection error corresponds to completely different physical millimeter deviations, invalidating NRA scoring rings.
2. **Perspective & Aspect Distortion Failure:** Resizing arbitrary rectangular photos (4:3 or 16:9) directly distorts circles into non-uniform ellipses. Furthermore, tilting the camera introduces perspective keystoning (parallax), shifting radial distances non-linearly.

### The Objective
Transform the pipeline into a closed-loop canonical normalization architecture: regardless of camera distance, orientation, aspect ratio, optical tilt (up to $\pm 45^\circ$), or off-center framing, the front-end mathematically rectifies and re-scales the target into an exact metric-calibrated canonical plane (**$512 \times 512$**, **$800 \times 800$**, or **$1024 \times 1024$** at $4.0\text{ px/mm}$) before the neural network evaluates it.

```
          Raw Smartphone Photo (Any Aspect, Tilted, Distant, Angled)
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 1: Optical Quality & Pre-Filtering Gate            │
       │  - EXIF orientation auto-rotation                        │
       │  - Motion blur validation (Laplacian variance > 75.0)    │
       │  - Specular glare / flash reflection suppression         │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 2: Classical Geometric Normalization & Pose Est.   │
       │  - Multi-scale Bullseye segmentation (Otsu + Morph close)│
       │  - Direct Least-Squares Ellipse Fitting (Taubin/Fitzg.)  │
       │  - Camera Normal Vector & Tilt Angles (Pitch, Yaw, Roll) │
       │  - Closed-form 3x3 Homography Matrix H                   │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 3: Metric Auto-Calibration & Canonical Cropping    │
       │  - Match detected bullseye radius to official NRA specs: │
       │      * B-2: 38.989 mm physical radius                    │
       │      * B-3: 38.862 mm physical radius                    │
       │  - Resample to 512x512, 800x800, or 1024x1024 @ 4.0 px/mm│
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 4: High-Res Heatmap U-Net Inference                │
       │  - Network sees 100% normalized, centered, flat target   │
       │  - Outputs continuous sub-pixel bullet coordinates in mm │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │ Stage 5: Inward Scoring & Ballistics Metrology HUD       │
       │  - Official NRA ring table evaluation (Rules 4.2 & 4.3)  │
       │  - Metric overlay, PII redaction, and Extreme Spread     │
       └──────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### Component A: Optical Quality & Pre-Filtering Gate
* **EXIF Orientation Normalization:** Mobile image headers store rotation flags (Tag 274). If ignored, OpenCV opens photos sideways or upside down.
  * *Action:* Intercept EXIF metadata and apply lossless transpose/flip matrices before any array allocations.
* **Motion Blur Gate:** Evaluate the variance of the 2D Laplacian operator across the image:
  $$\text{Blur Metric} = \mathrm{Var}\left( \nabla^2 I \right)$$
  If $\text{Blur Metric} < 75.0$, flag warning to steady the camera.
* **Specular Glare Suppression:** Range overhead lighting often creates high-specular hotspots on black target ink.
  * *Action:* Identify clipping white pixels inside dark regions and interpolate them using fast Navier-Stokes inpainting (`cv2.inpaint`).

### Component B: Perspective Rectification via Ellipse Decomposition
When an orthogonal circular target of radius $R$ is viewed obliquely by a camera at an angle $\theta$, its projection on the image sensor is an ellipse:
* **Sub-Pixel Ellipse Fitting (Direct Least-Squares):** Fit an algebraic conic section $A x^2 + B x y + C y^2 + D x + E y + F = 0$ constrained by $4AC - B^2 = 1$ (Fitzgibbon-Pilu-Fisher formulation) to the boundary points of the segmented bullseye.
* **Closed-Form Homography Estimation:**
  * The tilt angle (camera pitch off-normal) is:
    $$\theta = \arccos\left(\frac{b}{a}\right)$$
  * The major axis $2a$ is completely uncompressed by perspective tilt and represents the true physical diameter in the camera plane.
  * Derive $3 \times 3$ projective matrix $H$ mapping ellipse axes onto orthogonal canonical circle coordinates while preserving upright roll angle.

### Component C: Multi-Resolution Metric Calibration

| Canvas Dimension | Metric Scale | Field of View | Observable Area | Use Case |
| :---: | :---: | :---: | :---: | :--- |
| **$512 \times 512\text{ px}$** | $4.0\text{ px/mm}$ | $128 \times 128\text{ mm}$ ($5.04''$) | Bullseye + inner rings | Fast verification / low memory |
| **$800 \times 800\text{ px}$** | $4.0\text{ px/mm}$ | $200 \times 200\text{ mm}$ ($7.87''$) | All rings through 5-ring | Extended grouping analysis |
| **$1024 \times 1024\text{ px}$** | $4.0\text{ px/mm}$ | $256 \times 256\text{ mm}$ ($10.08''$) | Full card, all rings, margins | Official competition standard ($0.335\text{ mm}$ MRE) |

The required half-width bounding crop in raw pixels centered at $(x_c, y_c)$ is:
$$W_{\text{crop\_raw}} = \frac{\text{Canvas Dim} / 2.0}{\text{pixels\_per\_mm}} \times S_{\text{raw}} = \frac{\text{Canvas Dim}}{2 \times \text{pixels\_per\_mm} \times R_{\text{nominal}}} \times a$$

---

## 3. Execution Commands

### Standalone Geometric Rectification
```bash
# Rectify to 1024x1024 canonical plane
target-scorer rectify \
    --image /path/to/tilted_photo.jpg \
    --target B-2 \
    --canvas-size 1024 \
    --scale 4.0 \
    --output rectified_1024.png \
    --debug rect_debug.png

# Rectify to 800x800 canonical plane
target-scorer rectify \
    --image /path/to/tilted_photo.jpg \
    --target B-3 \
    --canvas-size 800 \
    --scale 4.0 \
    --output rectified_800.png
```

### End-to-End Rectification and Scoring
```bash
target-scorer infer \
    --image /path/to/tilted_photo.jpg \
    --target B-2 \
    --canvas-size 1024 \
    --shots 10 \
    --redact-pii \
    --output final_scorecard.png
```
