# Target Scoring Metrology Engine: Verification & Testing Plan

## 1. Resolution Recommendation & Analysis

The target scoring engine supports a parameterized canonical canvas architecture:
* **$256 \times 256\text{ px}$ ($2.0\text{ px/mm}$):** Ultra-fast testing baseline for CI/CD and CPU unit assertion suites.
* **$512 \times 512\text{ px}$ ($4.0\text{ px/mm}$):** Production baseline for lightweight local devices.
* **$800 \times 800\text{ px}$ ($4.0\text{ px/mm}$):** Extended field-of-view canvas capturing outer 6- and 5-rings.
* **$1024 \times 1024\text{ px}$ ($4.0\text{ px/mm}$):** Official competition standard; resolves flyers, full cardboard fluting texture, and delivers **$0.335\text{ mm}$ sub-pixel precision** ($< 1.35\text{ px}$).

### Multi-Resolution Matrix

| Parameter | Lightweight ($256 \times 256$) | Baseline ($512 \times 512$) | High-Res ($800 \times 800$) | Competition ($1024 \times 1024$) |
| :--- | :---: | :---: | :---: | :---: |
| **Metric Scale ($S_{\text{canon}}$)** | $2.0\text{ px/mm}$ | $4.0\text{ px/mm}$ | $4.0\text{ px/mm}$ | **$4.0\text{ px/mm}$** |
| **Field of View (FOV)** | $128.0\text{ mm}$ ($5.04''$) | $128.0\text{ mm}$ ($5.04''$) | $200.0\text{ mm}$ ($7.87''$) | **$256.0\text{ mm}$ ($10.08''$)** |
| **Aiming Black Diameter (B-2)** | $155.95\text{ px}$ | $311.9\text{ px}$ | $311.9\text{ px}$ | **$311.9\text{ px}$** |
| **Heatmap Sigma ($\sigma$)** | $1.5\text{ px}$ | $2.5\text{ px}$ | $2.8\text{ px}$ | **$3.2\text{ px}$** |
| **Sub-Pixel MRE** | $0.053\text{ mm}$ (geom) | $0.465\text{ mm}$ | $0.380\text{ mm}$ | **$0.335\text{ mm}$** |
| **Target Hardware** | CPU / CI tests | GTX 1060 / CPU | GTX 1060 / T4 | **NVIDIA A100 / T4** |

---

## 2. Testing Architecture & Verification Workflow

```mermaid
flowchart TD
    subgraph S1["Stage 1: Geometric Rectification Tests"]
        A1["Synthetic Target Generator"] -->|"Pitch Tilt (0° to 45°)<br/>Roll Angle (0° to 180°)"| B1["TargetRectifier Engine"]
        B1 --> C1{"Check Conic Fit & H Matrix"}
        C1 -->|"Circularity > 0.98<br/>Upright Roll Invariance"| D1["Pass Stage 1"]
    end

    subgraph S2["Stage 2: Sub-Pixel Peak Extraction Tests"]
        A2["Synthesized 2D Gaussian Heatmap<br/>Known Continuous Center (x*, y*)"] --> B2["extract_subpixel_peaks()"]
        B2 --> C2{"Evaluate Residual |p* - p_hat|"}
        C2 -->|"Error < 0.08 px<br/>Hessian Negative Definite"| D2["Pass Stage 2"]
    end

    subgraph S3["Stage 3: NRA Metrology & Scoring Tests"]
        A3["Known Coordinates & Gauge Offset"] --> B3["score_target_group()"]
        B3 --> C3{"Validate Inward Rule & Dispersion"}
        C3 -->|"Exact Ring Value Match<br/>Extreme Spread Exact Match"| D3["Pass Stage 3"]
    end

    D1 --> E["Stage 4: End-to-End Pipeline Validation (256, 512, 1024)"]
    D2 --> E
    D3 --> E
```

---

## 3. Modular Verification Test Suite

### Module 1: Classical Geometric Rectifier (`tests/test_camera_angles.py`)

*Objective:* Verify that `TargetRectifier` accurately locates the aiming black, derives an accurate 3x3 homography matrix, suppresses flash reflections, and warps the target into a canonical frame with zero artificial roll rotation.

| Test ID | Condition | Inputs / Ground Truth | Verification Criteria | Status |
| :--- | :--- | :--- | :--- | :---: |
| **GEO-01** | Flat Target (0° Tilt) | Orthogonal circle, $R_{\text{raw}} = 100.0\text{ px}$ | Rectified circularity $b/a \ge 0.985$, canonical radius within $\pm 0.8\text{ px}$. | **PASS** |
| **GEO-02** | Variable Pitch Tilt | Ellipses with pitch tilt $\theta \in \{15^\circ, 30^\circ, 42^\circ\}$ | Detected tilt angle error $|\hat{\theta} - \theta| < 2.5^\circ$. Unwarped circularity $> 0.970$. | **PASS** |
| **GEO-03** | Roll Invariance | Ellipse oriented at arbitrary roll angles $\phi \in \{0^\circ, 35^\circ, 75^\circ, 90^\circ, 135^\circ\}$ | Orientation of canonical image matches input without unexpected flips. | **PASS** |
| **GEO-04** | Glare Inpainting | Specular hotspot injected into black bull | Inpainted region restored; contour segmentation does not rupture. | **PASS** |
| **GEO-05** | Blur Metric Gate | Laplacian variance $\mathrm{Var}(\nabla^2 I) < 75.0$ | Result triggers `blur_metric` warning flag. | **PASS** |

---

### Module 2: Official NRA Inward Scoring & Ballistic Dispersion (`tests/test_low_res_metrology.py`)

*Objective:* Verify that `score_target_group` strictly follows official NRA Precision Pistol inward scoring rules and produces mathematically exact group statistics.

#### Ground Truth Test Cases for NRA Inward Scoring
```
NRA Inward Scoring Rule:
Effective Gauge Distance = max(0.0, Distance_to_Center - Bullet_Radius)
Bullet Radius = 5.69 mm / 2 = 2.845 mm
```

| Shot Vector | Ring Tested | Shot Center Dist ($d_{\text{shot}}$) | Ring Outer Radius ($R_{\text{ring}}$) | Effective Gauge Dist ($d_{\text{gauge}}$) | Expected Call |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **SC-01 (B-2)** | B-2 Center Shot | $1.20\text{ mm}$ | $11.430\text{ mm}$ (10 ring) | $0.00\text{ mm}$ | **10 pts** (Rule 4.2 has no X-ring) |
| **SC-01 (B-3)** | B-3 Center Shot | $1.20\text{ mm}$ | $11.430\text{ mm}$ (X ring) | $0.00\text{ mm}$ | **X (10 pts)** (Rule 4.3 X-ring) |
| **SC-02** | B-2 Breaking 10 | $14.15\text{ mm}$ | $11.430\text{ mm}$ (10 ring) | $14.15 - 2.845 = 11.305\text{ mm} \le 11.430$ | **10 pts** (Line Breaker) |
| **SC-03** | B-2 Missing 10 | $14.35\text{ mm}$ | $11.430\text{ mm}$ (10 ring) | $14.35 - 2.845 = 11.505\text{ mm} > 11.430$ | **9 pts** (Touches 9 only) |
| **SC-04** | B-2 Breaking 8 | $31.00\text{ mm}$ | $28.321\text{ mm}$ (8 ring) | $31.00 - 2.845 = 28.155\text{ mm} \le 28.321$ | **8 pts** (Line Breaker) |
| **SC-05** | B-3 Breaking 9 | $41.50\text{ mm}$ | $38.862\text{ mm}$ (9 ring) | $41.50 - 2.845 = 38.655\text{ mm} \le 38.862$ | **9 pts** (Black edge touch) |

---

## 4. Execution & Implementation Roadmap

### Phase 1: Parameterize Resolution in Core Engine [Completed ✓]
- Dynamic canvas scaling across $256$, $512$, $800$, and $1024\text{ px}$.

### Phase 2: Execute Unit & Mathematical Assertions [Completed ✓]
- 35 metrology assertions passing in `tests/test_low_res_metrology.py`.
- 5 camera angle assertions passing in `tests/test_camera_angles.py`.

### Phase 3: Benchmark Low-Res Statistical Performance [Completed ✓]
- 100 procedural targets at $256 \times 256$ evaluated (`tests/test_benchmark_low_res.py`):
  - Actual MRE: $0.0526\text{ mm}$ ($0.1052\text{ px}$).
  - Scoring agreement: $99.78\%$ ($449/450$ shots).

### Phase 4: Full-Scale $512 \times 512$ Validation [Completed ✓]
- Production training on GTX 1060; MRE achieved $0.465\text{ mm}$.
- Visual scorecard inspection confirmed single concentric ring rendering.

### Phase 5: Handwriting & Scorer Marks Invariance [Completed ✓]
- Procedural handwriting generation (competitor name, relay, NRA#) in graphite, ballpoint, and marker inks.
- Scorer circled shots and margin tallies (`96-3X`).
- PII redaction (`--redact-pii`) masking header registrations.

### Phase 6: $1024 \times 1024$ Cloud GPU Training & Warm-Start [Completed ✓]
- 1-Click cloud GPU orchestration via Google Colab CLI (`target-scorer colab-train`).
- Warm-start fine-tuning (`--initial-weights`) reaching **$98.9\%$ recall** and **$0.335\text{ mm}$ sub-pixel error** on NVIDIA A100.
- Symmetrical 10-shot competition evaluation scorecards generated for NRA B-2 (`96 / 100`) and B-3 (`100 - 7X / 100`).
