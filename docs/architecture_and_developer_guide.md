# Technical Architecture & Developer Guide

This document details the engineering principles, mathematical formulations, coordinate transformations, neural network architectures, and metrology algorithms underlying the [`target_scorer`](file:///home/antho/target_scorer) engine.

---

## Table of Contents
1. [End-to-End System Architecture](#1-end-to-end-system-architecture)
2. [Geometric Normalization & Rectification Pipeline](#2-geometric-normalization--rectification-pipeline)
3. [Deep Learning Engine: HighResHeatmapUNet](#3-deep-learning-engine-highresheatmapunet)
4. [Loss Formulation: Adaptive Wing Loss (A-Wing)](#4-loss-formulation-adaptive-wing-loss-a-wing)
5. [Sub-Pixel Peak Extraction via 2D Taylor-Series Expansion](#5-sub-pixel-peak-extraction-via-2d-taylor-series-expansion)
6. [Metrology, Coordinate Systems, & Inward Scoring Math](#6-metrology-coordinate-systems--inward-scoring-math)
7. [TargetAnnotation JSON Schema & Data Model](#7-targetannotation-json-schema--data-model)

---

## 1. End-to-End System Architecture

The pipeline processes raw camera frames through three decoupled, deterministic stages:

```mermaid
flowchart TD
    subgraph S1["Stage 1: Geometric Normalization & Ingestion"]
        In["Raw Target Source (HEIC / RAW / JPEG / Picamera2 RAM)"]
        In --> Canon["ImageCanonicalizer: EXIF Rotation & sRGB Mapping"]
        Canon --> Blur["Motion Blur Check (Laplacian Variance > 75.0)"]
        Blur --> Inpaint["Navier-Stokes Specular Glare Inpainting"]
        Inpaint --> Seg["Aiming Black Segmentation & Morphological Close"]
        Seg --> Ellipse["Direct Least-Squares Ellipse Fit (Fitzgibbon)"]
        Ellipse --> Homography["Roll-Invariant 3x3 Homography Warping"]
    end

    subgraph S2["Stage 2: Deep Learning Sub-Pixel Regression"]
        Homography --> Tensor["Normalize Tensor (ImageNet mean & std)"]
        Tensor --> UNet["HighResHeatmapUNet (Stem + 3-level U-Net)"]
        UNet --> Hmap0["Channel 0: Continuous Bullet Hole Heatmap"]
        UNet --> Hmap1["Channel 1: Bullseye Center Heatmap"]
    end

    subgraph S3["Stage 3: Metrology & Scoring Engine"]
        Hmap0 --> NMS["5x5 Morphological Dilation NMS"]
        NMS --> Taylor["2D Quadratic Taylor-Series Sub-Pixel Peak Refinement"]
        Hmap1 --> Center["Constrained Central Bullseye Search"]
        Taylor & Center --> Radial["Calibrated Radial Distance Metric: d = hypot(dx, dy) / scale"]
        Radial --> Inward["NRA Rule 14.3 Inward Break Evaluation (Flange Touch)"]
        Inward --> Dispersion["Ballistic Dispersion: Extreme Spread, Mean Radius, COI, MOA"]
        Dispersion --> Out["Inspection Scorecard Overlay + TargetAnnotation JSON"]
    end
```

---

## 2. Geometric Normalization & Rectification Pipeline

Source: [`src/target_scorer/geometry/rectifier.py`](file:///home/antho/target_scorer/src/target_scorer/geometry/rectifier.py)

Real-world smartphone target photographs suffer from perspective foreshortening, camera pitch/yaw, rotation, and non-uniform lighting. The geometric stage eliminates all camera dependencies before deep learning inference.

### A. Format Ingestion & Orientation
The [`ImageCanonicalizer`](file:///home/antho/target_scorer/src/target_scorer/geometry/canonical_loader.py) decodes Apple HEIC/HEIF (via `pillow-heif`), Apple ProRAW DNG, Android Ultra HDR, TIFF, WebP, PNG, and JPEG. It transposes EXIF orientation tags (1–8) and converts embedded ICC color profiles (e.g. Apple Display P3) to standardized sRGB to prevent intensity clipping.

### B. Motion Blur Metric
The image sharpness is evaluated using the variance of the 2D Laplacian:
$$\text{Blur Metric} = \mathrm{Var}\left(\nabla^2 I_{\text{gray}}\right)$$
If the variance drops below $\tau_{\text{blur}} = 75.0$, a diagnostic warning is flagged for potential motion blur.

### C. Navier-Stokes Glare Suppression
Harsh range lights or camera flashes produce specular reflections on target ink. Glare regions are segmented in HSV space:
$$\text{Glare Mask} = (V > 230) \land (S < 45) \land (\text{Dark Mask} > 0)$$
The mask is dilated and inpainted using fast Telea / Navier-Stokes morphological inpainting, restoring dark contrast.

### D. Direct Least-Squares Conic Fitting
The outermost contour of the dark aiming bullseye is extracted and fitted to an algebraic conic equation:
$$F(\mathbf{a}, \mathbf{x}) = a x^2 + b x y + c y^2 + d x + e y + f = 0$$
Using the Fitzgibbon direct least-squares method under the ellipse constraint $4ac - b^2 = 1$, the equation solves via generalized eigenvectors:
$$\mathbf{D}^T \mathbf{D} \mathbf{u} = \lambda \mathbf{C} \mathbf{u}$$
This yields the ellipse center $(x_0, y_0)$, semi-major axis $a$, semi-minor axis $b$, and roll orientation angle $\theta$.

### E. Closed-Form Metric Homography
The perspective tilt angle $\phi$ is determined by the foreshortening ratio:
$$\cos \phi = \frac{b}{a} \implies \phi = \arccos\left(\frac{b}{a}\right)$$
The nominal physical radius of the NRA bullseye is known:
* **NRA B-2 (50 ft Slow Fire):** $r_{\text{nominal}} = 38.989\text{ mm}$ (3.07 in diameter)
* **NRA B-3 (50 ft Timed/Rapid):** $r_{\text{nominal}} = 38.862\text{ mm}$ (3.06 in diameter)

The canonical circle has radius $R_{\text{can}} = r_{\text{nominal}} \times \text{scale}$ centered at $(C_{\text{can}}, C_{\text{can}})$. Four conjugate control points are mapped from the fitted raw ellipse vertices to the canonical circle perimeter to compute the $3 \times 3$ projective matrix $\mathbf{H}$. The image is warped via perspective transformation:
$$\mathbf{x}_{\text{can}} \sim \mathbf{H} \, \mathbf{x}_{\text{raw}}$$

---

## 3. Deep Learning Engine: HighResHeatmapUNet

Source: [`src/target_scorer/models/unet.py`](file:///home/antho/target_scorer/src/target_scorer/models/unet.py)

The neural network regresses two continuous 2D heatmap channels over the canonical $1024 \times 1024$ image:
* **Channel 0:** Gaussian peaks centered at each bullet hole impact.
* **Channel 1:** Gaussian peak centered at the target bullseye origin.

```
Input: (B, 3, 1024, 1024)
  │
  ▼
[Stem Block] ──> Conv(3->32, 3x3) + BN + Mish + ResConvBlock(32->32) ──────────┐
  │                                                                           │ (Skip s0)
  ▼ MaxPool(2x)                                                               │
[Down 1] ─────> ResConvBlock(32->64) ──────────────────────────┐              │
  │                                                            │ (Skip s1)    │
  ▼ MaxPool(2x)                                                │              │
[Down 2] ─────> ResConvBlock(64->128) ──────────┐              │              │
  │                                             │ (Skip s2)    │              │
  ▼ MaxPool(2x)                                 │              │              │
[Down 3] ─────> ResConvBlock(128->256)          │              │              │
  │                                             │              │              │
  ▼ Upsample(2x, Bilinear)                      │              │              │
[Dec 3]  ─────> Concat(s2) -> ResConvBlock(256+128 -> 128)     │              │
  │                                                            │              │
  ▼ Upsample(2x, Bilinear)                                     │              │
[Dec 2]  ─────> Concat(s1) -> ResConvBlock(128+64 -> 64)       │              │
  │                                                            │              │
  ▼ Upsample(2x, Bilinear)                                     │              │
[Dec 1]  ─────> Concat(s0) -> ResConvBlock(64+32 -> 32) ───────┘              │
  │                                                                           │
  ▼                                                                           │
[Final Head] ─> Conv(32->32) + BN + Mish + Conv(32->2, 1x1) + Sigmoid ────────┘
  │
  ▼
Output: (B, 2, 1024, 1024)  [Channel 0: Bullets, Channel 1: Bullseye]
```

### Key Architectural Characteristics:
1. **Full-Resolution Stem:** Processes the $1024 \times 1024$ image at stride 1 with a 32-channel feature representation, retaining subtle bullet hole grease rings and cardboard fiber raggedness before downsampling.
2. **Residual ConvBlocks:** Each block features dual $3 \times 3$ convolutions with Batch Normalization, Mish activations ($\text{Mish}(x) = x \tanh(\ln(1 + e^x))$), and a $1 \times 1$ projection residual shortcut.
3. **Bilinear Upsampling:** Avoids checkerboard deconvolution artifacts common to transposed convolutions.

---

## 4. Loss Formulation: Adaptive Wing Loss (A-Wing)

Source: [`src/target_scorer/models/loss.py`](file:///home/antho/target_scorer/src/target_scorer/models/loss.py)

Heatmap regression poses a severe class imbalance problem: over 99.5% of the target canvas is zero-valued cardboard background, while the ground truth bullet impacts are sharp Gaussians ($\sigma = 3.2\text{ px}$). Standard Mean Squared Error (MSE) suffers from vanishing gradients on small residuals, causing the network to blur peak centers.

We employ **Adaptive Wing Loss (A-Wing)**:
$$\mathcal{L}(\hat{y}, y) = \begin{cases} \omega \ln\left(1 + \left|\frac{\hat{y} - y}{\epsilon}\right|^{\alpha - y}\right) & \text{if } |\hat{y} - y| < \theta \\ A |\hat{y} - y| - C & \text{otherwise} \end{cases}$$

Where the transition constants ensure continuity at $|\hat{y} - y| = \theta$:
$$A = \frac{\omega}{\epsilon} (\alpha - y) \left(\frac{\theta}{\epsilon}\right)^{\alpha - y - 1} \frac{1}{1 + (\theta / \epsilon)^{\alpha - y}}$$
$$C = \theta A - \omega \ln\left(1 + \left(\frac{\theta}{\epsilon}\right)^{\alpha - y}\right)$$

```
Loss Gradient Magnitude |d L / d delta|
 ▲
 │         Adaptive Wing (High peak gradient for small delta)
 │        /
 │       /
 │      /
 │     /            Smooth L1 (Constant linear slope)
 │    /            /
 │   /            /
 │  /            /
 │ /            /
 └────────────────────────────► Absolute Error delta = |y_hat - y|
```

### Hyperparameters:
* $\omega = 14.0$: Controls the maximum wing loss penalty.
* $\theta = 0.5$: Transition threshold between logarithmic and linear regimes.
* $\epsilon = 1.0$: Scale factor preventing gradient explosion near zero.
* $\alpha = 2.1$: Non-linear power factor modulating sensitivity near the peak top ($y \approx 1$).
* $\text{Foreground Weight} = 10.0$: Boosts loss for pixels where $y > 0.05$ to focus gradient updates on bullet craters.

---

## 5. Sub-Pixel Peak Extraction via 2D Taylor-Series Expansion

Source: [`src/target_scorer/metrology/peaks.py`](file:///home/antho/target_scorer/src/target_scorer/metrology/peaks.py)

To measure groups down to fractions of a millimeter ($< 0.35\text{ mm}$ error), peak locations cannot be restricted to integer grid coordinates. Continuous sub-pixel coordinates are extracted through a 5-step algorithm:

```
  [ Predicted Heatmap ]
            │
            ▼
  1. 5x5 Morphological Dilation: peaks = (H == dilate(H, 5x5)) & (H > threshold)
            │
            ▼
  2. Connected Components: Extract integer centroids (x_int, y_int) and areas
            │
            ▼
  3. Spatial Suppression: Enforce minimum inter-shot distance (min_dist_px = 10.0 px)
            │
            ▼
  4. Local 3x3 Patch Sampling:
        [ H(-1,-1)  H(0,-1)  H(1,-1) ]
        [ H(-1, 0)  H(0, 0)  H(1, 0) ]
        [ H(-1, 1)  H(0, 1)  H(1, 1) ]
            │
            ▼
  5. 2D Quadratic Taylor Expansion:
        offset = - inv(Hessian) @ Gradient
        refined_coord = (x_int + offset_x, y_int + offset_y)
```

### Mathematical Formulation:
For each local peak $(x_{\text{int}}, y_{\text{int}})$, the local continuous surface is modeled by quadratic Taylor expansion:
$$H(\Delta x, \Delta y) \approx H_0 + \mathbf{g}^T \mathbf{\Delta} + \frac{1}{2} \mathbf{\Delta}^T \mathcal{H} \mathbf{\Delta}$$

The gradient vector $\mathbf{g} = [g_x, g_y]^T$ is computed via central differences:
$$g_x = \frac{H(x+1, y) - H(x-1, y)}{2}, \quad g_y = \frac{H(x, y+1) - H(x, y-1)}{2}$$

The $2 \times 2$ Hessian matrix $\mathcal{H}$ captures local curvature:
$$\mathcal{H}_{xx} = H(x+1, y) - 2H(x,y) + H(x-1, y)$$
$$\mathcal{H}_{yy} = H(x, y+1) - 2H(x,y) + H(x, y-1)$$
$$\mathcal{H}_{xy} = \frac{H(x+1, y+1) - H(x+1, y-1) - H(x-1, y+1) + H(x-1, y-1)}{4}$$

Setting $\nabla H = 0$ yields the continuous peak displacement:
$$\mathbf{\Delta}^* = -\mathcal{H}^{-1} \mathbf{g} = -\frac{1}{\det \mathcal{H}} \begin{bmatrix} \mathcal{H}_{yy} & -\mathcal{H}_{xy} \\ -\mathcal{H}_{xy} & \mathcal{H}_{xx} \end{bmatrix} \begin{bmatrix} g_x \\ g_y \end{bmatrix}$$

The displacement is clamped to $[-0.9, 0.9]\text{ px}$ to prevent singularity divergence. The final sub-pixel coordinate is:
$$x^* = x_{\text{int}} + \Delta_x^*, \quad y^* = y_{\text{int}} + \Delta_y^*$$

---

## 6. Metrology, Coordinate Systems, & Inward Scoring Math

Source: [`src/target_scorer/metrology/scoring.py`](file:///home/antho/target_scorer/src/target_scorer/metrology/scoring.py)

### A. Coordinate Transformations
The metrology engine operates across three distinct coordinate frames:

```
  1. Raw Camera Frame: (x_raw, y_raw) in pixels
        │
        │ Projective Warp: x_can ~ H @ x_raw
        ▼
  2. Canonical Canvas Frame: (x_can, y_can) in pixels [0, 1024]
        │
        │ Origin Shift & Metric Scaling:
        │   dx_mm = (x_can - x_bull) / pixels_per_mm
        │   dy_mm = -(y_can - y_bull) / pixels_per_mm
        ▼
  3. Physical Target Frame: (x_mm, y_mm) in millimeters (Bullseye at 0, 0)
```

### B. Official NRA Target Specifications
Dimensions conform strictly to the NRA Precision Pistol Rulebook:

#### NRA B-2 Target (50 ft Slow Fire, Rule 4.2)
* **Bullseye (Aiming Black):** Rings 10, 9, 8 (3.07 in / 77.978 mm diameter).
* **Innermost Ring:** 10-ring ($0.90'' = 22.86\text{ mm}$ dia, radius $11.43\text{ mm}$). No X-ring.
* **Scoring Table:**
  * 10-Ring: $r \le 11.430\text{ mm}$ (10 points)
  * 9-Ring: $r \le 27.432\text{ mm}$ (9 points)
  * 8-Ring: $r \le 38.989\text{ mm}$ (8 points)
  * 7-Ring: $r \le 55.880\text{ mm}$ (7 points)
  * 6-Ring: $r \le 72.771\text{ mm}$ (6 points)
  * 5-Ring: $r \le 89.662\text{ mm}$ (5 points)
  * 4-Ring: $r \le 106.553\text{ mm}$ (4 points)

#### NRA B-3 Target (50 ft Timed & Rapid Fire, Rule 4.3)
* **Bullseye (Aiming Black):** Rings X, 10, 9, 8 (3.06 in / 77.724 mm diameter).
* **Innermost Ring:** X-ring ($0.90'' = 22.86\text{ mm}$ dia, radius $11.43\text{ mm}$).
* **Scoring Table:**
  * X-Ring: $r \le 11.430\text{ mm}$ (10 points, inner-X flag)
  * 10-Ring: $r \le 22.860\text{ mm}$ (10 points)
  * 9-Ring: $r \le 30.861\text{ mm}$ (9 points)
  * 8-Ring: $r \le 38.862\text{ mm}$ (8 points)
  * 7-Ring: $r \le 52.578\text{ mm}$ (7 points)
  * 6-Ring: $r \le 66.294\text{ mm}$ (6 points)

### C. NRA Rule 14.3 Inward Break Scoring
Official scoring uses the **inward scoring advantage**: if the outer boundary of a bullet hole or the flange of a scoring gauge touches or breaks a scoring ring line, the shot is awarded the higher value.

Mathematically, for a bullet of caliber diameter $D_{\text{bullet}}$ (radius $R_{\text{bullet}} = D_{\text{bullet}} / 2$), the center-to-bullseye radial distance is:
$$d_{\text{radial}} = \sqrt{(x_{\text{shot}} - x_{\text{bull}})^2 + (y_{\text{shot}} - y_{\text{bull}})^2} \; / \; \text{scale}$$

The **effective gauge distance** represents the closest approach of the bullet's outer leaded perimeter toward the target bullseye:
$$d_{\text{gauge}} = \max\left(0, d_{\text{radial}} - R_{\text{bullet}}\right)$$

The shot is assigned the highest value ring where:
$$d_{\text{gauge}} \le R_{\text{ring}}$$

### D. Ballistic Group Statistics
1. **Extreme Spread ($ES$):** Maximum Euclidean distance between any pair of shots in the group:
   $$ES = \max_{i < j} \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$
2. **Center of Impact ($COI$):** Centroid of all impacts:
   $$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i, \quad \bar{y} = \frac{1}{N} \sum_{i=1}^N y_i$$
3. **Mean Radius ($MR$):** Average distance of all shots from the Center of Impact:
   $$MR = \frac{1}{N} \sum_{i=1}^N \sqrt{(x_i - \bar{x})^2 + (y_i - \bar{y})^2}$$
4. **Minutes of Angle (MOA):** Conversion of millimeter dispersion to angular measurement at distance $D_{\text{yd}}$ ($50\text{ ft} = 16.667\text{ yd}$):
   $$\text{MOA} = \frac{\text{size}_{\text{mm}}}{D_{\text{yd}} \times \tan(1'/60^\circ) \times 914.4} \approx \frac{\text{size}_{\text{mm}}}{D_{\text{yd}} \times 0.26598}$$

---

## 7. TargetAnnotation JSON Schema & Data Model

Source: [`src/target_scorer/data/schema.py`](file:///home/antho/target_scorer/src/target_scorer/data/schema.py)

Annotations are serialized into human-readable JSON files adhering to standard precision pistol schemas.

### JSON Schema Structure
```json
{
  "version": "1.0",
  "target_id": "relay_14_card_03",
  "image_filename": "relay_14_card_03_canonical.png",
  "canvas_size": 1024,
  "pixels_per_mm": 4.0,
  "target_type": "B-3",
  "bullseye_center": [512.0, 512.0],
  "shots": [
    {
      "id": 1,
      "center_x": 516.42,
      "center_y": 508.19,
      "score": 10,
      "is_x": true,
      "caliber_mm": 5.69,
      "shot_type": "single",
      "confidence": 0.982
    },
    {
      "id": 2,
      "center_x": 492.15,
      "center_y": 521.84,
      "score": 10,
      "is_x": false,
      "caliber_mm": 5.69,
      "shot_type": "single",
      "confidence": 0.945
    }
  ],
  "metadata": {
    "autolabeled": true,
    "manually_verified": true,
    "verified_at": "2026-10-18T18:42:00",
    "total_score": 100,
    "x_count": 6,
    "extreme_spread_mm": 21.45,
    "mean_radius_mm": 7.32,
    "source_blur": 142.5,
    "source_tilt": 4.2
  }
}
```

### Python Data Classes:
* **[`AnnotatedShot`](file:///home/antho/target_scorer/src/target_scorer/data/schema.py#L14-L41):** Individual shot record. Includes sub-pixel coordinates, score, inner-X flag, bullet caliber, impact type (`single`, `double`, `keyhole`), and detection confidence.
* **[`TargetAnnotation`](file:///home/antho/target_scorer/src/target_scorer/data/schema.py#L43-L133):** Root card container. Includes target type, canonical canvas dimensions, scale factor, list of shots, bullseye coordinates, and flexible metadata dictionary. Provides `.to_json()` and `.from_json()` serialization methods.
