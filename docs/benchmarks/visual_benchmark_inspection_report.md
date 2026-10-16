# Target Scorer Metrology & Benchmark Visual Inspection Report

This visual inspection report provides graphical proof and verification for all mathematical, geometric, and neural metrology benchmarks established in the [Testing Plan](../target_scoring_testing_plan.md).

All diagnostic visual panels in this report can be re-generated on demand using the reproduction script:
```bash
./target_env/bin/python scripts/generate_benchmark_visuals.py
```

---

## 1. Geometric Perspective Rectification (GEO-01 – GEO-05)

![Geometric Rectification](./assets/benchmark_01_geometric_rectification.png)

- **GEO-01 (Orthogonal Target):** At $0^\circ$ optical tilt, the rectifier maintains exact vertical orientation without spurious spinning, achieving circularity metric $1.000$ and zero tilt angle error.
- **GEO-02 (Perspective Pitch Tilt):** At $25^\circ$ optical tilt, the fitted ellipse cleanly unwarps back to the canonical $256 \times 256$ circular target frame ($\text{circularity} = 0.906$, recovered angle $= 25.0^\circ$).
- **GEO-03 (Compound Pitch & Roll):** At $38^\circ$ pitch and $35^\circ$ roll, the major axis alignment prevents orientation skewing and produces an unwarped circular bull ($\text{recovered angle} = 38.1^\circ$).
- **GEO-04 (Specular Flash Inpainting):** Specular glare hotspots inside the dark bullseye are masked via morphological thresholding and Telea inpainting prior to homography unwarping, preventing contour splitting.

---

## 2. Sub-Pixel Taylor Series 2D Metrology & NMS (MET-01 – MET-04)

![Sub-Pixel Metrology](./assets/benchmark_02_subpixel_metrology.png)

- **Continuous 2D Peak Recovery:** Resolves sub-pixel bullet coordinates $(\hat{x}, \hat{y})$ from the discrete heatmap grid via continuous 2D quadratic Taylor expansion:
  $$\Delta \mathbf{p} = -\mathbf{H}^{-1} \nabla f$$
- **Precision:** Sub-pixel recovery error is **$0.0032\text{ mm}$ ($0.0063\text{ px}$)** on standard Gaussian surfaces—well below the $< 0.24\text{ px}$ test plan threshold.
- **Caliber-Aware Spatial NMS:** Overlapping double-shot holes and keyhole tumbling impacts are cleanly resolved into discrete bullet centers using caliber-radius greedy non-maximum suppression without spurious multi-peak splitting.

---

## 3. Official NRA Rule 14.3 Inward Scoring (SC-01 – SC-05)

![NRA Inward Scoring](./assets/benchmark_03_inward_scoring.png)

Visual verification of the outward gauge radius expansion ($r_{\text{gauge}} = 2.845\text{ mm}$ for .22 LR / 5.69 mm):
- **SC-01 (Inner X Ring):** Shot center at $1.20\text{ mm} \le 5.715\text{ mm} \rightarrow$ inner X score awarded (gold outline).
- **SC-02 (Breaking 10-Ring Edge):** Shot center at $14.15\text{ mm}$ gives gauge distance $11.305\text{ mm} \le 11.43\text{ mm} \rightarrow$ outer gauge edge touches the 10-ring line $\rightarrow$ **Awarded 10** (green outline).
- **SC-03 (Near-Miss 10-Ring):** Shot center at $14.35\text{ mm}$ gives gauge distance $11.505\text{ mm} > 11.43\text{ mm} \rightarrow$ misses line by only $0.075\text{ mm} \rightarrow$ **Awarded 9** (cyan outline).
- **SC-04 (Breaking 8-Ring Edge):** Shot center at $31.00\text{ mm}$ gives gauge distance $28.155\text{ mm} \le 28.321\text{ mm} \rightarrow$ **Awarded 8** (orange outline).
- **SC-05 (NRA B-3 9-Ring Black Edge):** Shot center at $41.50\text{ mm}$ gives gauge distance $38.655\text{ mm} \le 38.862\text{ mm} \rightarrow$ **Awarded 9**.

---

## 4. Phase 3 100-Target Statistical Benchmark Dashboard

![Statistical Benchmark Dashboard](./assets/benchmark_04_statistical_100_targets.png)

Key metrics measured over 100 procedural targets (446 total evaluated shots):
1. **Radial Error (Taylor vs GT):**
   - Mean Radial Error: **$0.0362\text{ mm}$ ($0.0723\text{ px}$)**
   - Specification Requirement: $< 0.12\text{ mm}$ ($< 0.24\text{ px}$) — **Exceeded by 3.3×**.
2. **Execution Throughput:**
   - Measured Speed: **$241.6\text{ targets/second}$ on CPU**
   - Specification Requirement: $> 35\text{ targets/second}$ — **Exceeded by 6.9×**.
3. **Scoring Agreement:**
   - Identical agreement: **$100.00\%$** ($446 / 446$ shots).

---

## 5. End-to-End Production Inspection HUD Scorecard (E2E-01)

![Production HUD Scorecard](./assets/benchmark_05_e2e_scorecard_hud.png)

The composite HUD output demonstrates:
- **Left inspection canvas:** Canonical target with aiming black, ring contours, shot center crosshairs, caliber circles, and the orange Extreme Spread chord line.
- **Right metrology HUD:**
  - Total score display (`39 - 2X / 40`).
  - Ballistic dispersion: Extreme Spread ($28.37\text{ mm} / 6.40\text{ MOA}$), Mean Radius ($9.25\text{ mm} / 2.09\text{ MOA}$), Farthest Pair (`Shot #1 <-> #4`), and Windage/Elevation bias.
  - Per-shot metrology log with radial center distance and effective gauge boundary distance.
