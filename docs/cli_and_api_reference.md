# Command-Line & Python API Reference

This reference provides exhaustive documentation for all command-line interface (CLI) subcommands and programmatic Python APIs in the [`target_scorer`](file:///home/antho/target_scorer) package.

All CLI commands are accessible through the unified executable:
```bash
target-scorer <subcommand> [options]
```

---

## Table of Contents
1. [`target-scorer infer`](#1-target-scorer-infer)
2. [`target-scorer train`](#2-target-scorer-train)
3. [`target-scorer colab-train`](#3-target-scorer-colab-train)
4. [`target-scorer colab-autolabel`](#4-target-scorer-colab-autolabel)
5. [`target-scorer rectify`](#5-target-scorer-rectify)
6. [`target-scorer autolabel`](#6-target-scorer-autolabel)
7. [`target-scorer annotate`](#7-target-scorer-annotate)
8. [Python API Reference](#8-python-api-reference)

---

## 1. `target-scorer infer`

Scores an input target photograph or procedurally generates and scores a synthetic competition demo card. Executes automated perspective unwarping, deep learning heatmap regression, sub-pixel peak extraction, NRA inward break scoring, ballistic dispersion analysis, and visual scorecard rendering.

Source: [`src/target_scorer/infer.py`](file:///home/antho/target_scorer/src/target_scorer/infer.py)

### CLI Synopsis
```bash
target-scorer infer [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--image` | `str` | `None` | Path to target photograph, or `-` to stream raw bytes from `stdin`. If omitted, generates a deterministic synthetic demo target. |
| `--target` | `str` | `B-2` | NRA target type. Choices: `B-2` (50 ft Slow Fire) or `B-3` (50 ft Timed/Rapid Fire). Case-insensitive. |
| `--checkpoint` | `str` | `target_scoring_hrnet_best.pt` | Path to PyTorch model weights checkpoint (`.pt`). Searches `weights/` directory if relative path not found in cwd. |
| `--output` | `str` | `scored_inspection_result.png` | Destination filepath for the high-resolution annotated scorecard PNG. |
| `--debug-rect` | `str` | `rectification_debug.png` | Destination filepath for the perspective rectification diagnostic side-by-side visualization. Set to empty string `""` to disable. |
| `--distance` | `float` | `16.67` | Shooting distance to target in yards ($50\text{ ft} = 16.667\text{ yd}$). Used to convert millimeter dispersion to Minutes of Angle (MOA). |
| `--shots` | `int` | `10` | Maximum number of shots to evaluate. Standard NRA string length is `10`. |
| `--threshold` | `float` | `0.45` | Heatmap peak detection confidence threshold $[0.0, 1.0]$. Lower values ($0.30 - 0.40$) detect faint tears; higher values ($0.50 - 0.60$) reject cardboard debris. |
| `--device` | `str` | `None` (auto) | Compute device: `cuda`, `cpu`, or `mps`. If `None`, detects best available accelerator. |
| `--canvas-size` | `int` | `None` (from ckpt) | Override canonical canvas dimension in pixels (e.g. `512`, `800`, or `1024`). Defaults to checkpoint training configuration. |
| `--scale` | `float` | `None` (from ckpt) | Override metric resolution in pixels per millimeter (e.g. `4.0` for 1024, `2.0` for 256). Defaults to checkpoint training configuration. |
| `--redact-pii` | flag | `False` | Masks competitor registration handwriting, relay numbers, and shooter names on margins with circular cardboard masking. |

### Copy-Pasteable Shell Examples

#### A. Score a Real Smartphone Target Photo (HEIC, RAW/DNG, WebP, PNG, JPEG)
```bash
target-scorer infer \
  --image sample_targets/iphone_photo.heic \
  --target B-3 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --canvas-size 1024 \
  --scale 4.0 \
  --shots 10 \
  --output visual_inspection_reports/scored_iphone_b3.png \
  --debug-rect visual_inspection_reports/rect_debug_b3.png
```

#### B. Score with Competitor Registration PII Redaction
```bash
target-scorer infer \
  --image range_data/match_relay_4.jpg \
  --target B-2 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --canvas-size 1024 \
  --scale 4.0 \
  --redact-pii \
  --output visual_inspection_reports/scored_match_redacted.png
```

#### C. Stream Image from `curl` or Standard Input Pipe (`-`)
```bash
curl -s https://example.com/target_relay.png | target-scorer infer \
  --image - \
  --target B-2 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --canvas-size 1024 \
  --scale 4.0 \
  --output scored_streamed.png
```

#### D. Generate and Score a Synthetic Competition Demo Card
```bash
# NRA B-2 Slow Fire (Scores 96 / 100)
target-scorer infer \
  --target B-2 \
  --canvas-size 1024 \
  --scale 4.0 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --output scored_demo_b2.png

# NRA B-3 Timed/Rapid Fire (Scores 100 - 7X / 100)
target-scorer infer \
  --target B-3 \
  --canvas-size 1024 \
  --scale 4.0 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --output scored_demo_b3.png
```

---

## 2. `target-scorer train`

Trains the [`HighResHeatmapUNet`](file:///home/antho/target_scorer/src/target_scorer/models/unet.py) model locally. Supports pure procedural synthetic generation, pure real target fine-tuning (Option B), or blended hybrid training (Option A). Evaluates sub-pixel localization error against ground-truth coordinates every epoch.

Source: [`src/target_scorer/train.py`](file:///home/antho/target_scorer/src/target_scorer/train.py)

### CLI Synopsis
```bash
target-scorer train [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--canvas-size` | `int` | `512` | Canonical canvas resolution in pixels. Must be divisible by 8 (e.g. `256`, `512`, `800`, `1024`). |
| `--scale` | `float` | `4.0` | Metric scale in pixels per millimeter ($4.0\text{ px/mm}$ for 512, 800, 1024; $2.0\text{ px/mm}$ for 256). |
| `--epochs` | `int` | `15` | Total number of training epochs. |
| `--batch-size` | `int` | `4` | Mini-batch size. For 1024 resolution on 6GB–8GB GPUs, use `2`; for 16GB–24GB GPUs, use `4` or `8`. |
| `--lr` | `float` | `3.5e-4` | Initial learning rate for AdamW optimizer. Follows Cosine Annealing decay. |
| `--train-samples`| `int` | `400` | Number of procedural synthetic or hybrid training samples per epoch. |
| `--val-samples` | `int` | `50` | Number of validation samples per epoch. |
| `--checkpoint` | `str` | `target_scoring_hrnet_best.pt` | Output filepath for saving the best model weights. |
| `--workers` | `int` | `2` | Number of CPU worker subprocesses for PyTorch `DataLoader`. |
| `--sigma` | `float` | `None` (auto) | Heatmap Gaussian kernel standard deviation in pixels. Defaults to `0.8 * pixels_per_mm` ($3.2\text{ px}$ for 1024, $2.5\text{ px}$ for 512). |
| `--amp` / `--no-amp` | flag | `--amp` (True) | Enables/disables PyTorch Automatic Mixed Precision (`torch.amp.autocast`). |
| `--initial-weights`| `str` | `None` | Path to existing `.pt` weights to warm-start or fine-tune from. |
| `--real-data-dir`| `str` | `None` | Path to directory containing real target annotations (`*.json`) and images to enable hybrid or real training. |
| `--real-ratio` | `float` | `0.35` | Sampling probability for real target images. `1.0` = pure real training (Option B); `0.35` = hybrid 70% synthetic / 30% real (Option A); `0.0` = pure synthetic. |

### Copy-Pasteable Shell Examples

#### A. Warm-Start Fine-Tuning Locally ($1024 \times 1024$ on GTX 1060 / RTX 3060)
```bash
target-scorer train \
  --canvas-size 1024 \
  --scale 4.0 \
  --epochs 12 \
  --batch-size 2 \
  --lr 1.5e-4 \
  --sigma 3.2 \
  --initial-weights weights/target_scoring_hrnet_1024_best.pt \
  --checkpoint weights/target_scoring_hrnet_1024_finetuned.pt
```

#### B. Pure Real Target Fine-Tuning (Option B: Rapid Adaptation)
```bash
target-scorer train \
  --canvas-size 1024 \
  --scale 4.0 \
  --epochs 10 \
  --batch-size 2 \
  --lr 5.0e-5 \
  --sigma 3.2 \
  --real-data-dir datasets/real_targets \
  --real-ratio 1.0 \
  --initial-weights weights/target_scoring_hrnet_1024_best.pt \
  --checkpoint weights/target_scoring_hrnet_1024_real_adapted.pt
```

#### C. Hybrid Blended Training (Option A: 70% Synthetic + 30% Real)
```bash
target-scorer train \
  --canvas-size 1024 \
  --scale 4.0 \
  --epochs 15 \
  --batch-size 2 \
  --lr 1.0e-4 \
  --sigma 3.2 \
  --real-data-dir datasets/real_targets \
  --real-ratio 0.35 \
  --initial-weights weights/target_scoring_hrnet_1024_best.pt \
  --checkpoint weights/target_scoring_hrnet_1024_hybrid.pt
```

#### D. Fast Baseline Training ($512 \times 512$)
```bash
target-scorer train \
  --canvas-size 512 \
  --scale 4.0 \
  --epochs 15 \
  --batch-size 4 \
  --checkpoint weights/target_scoring_hrnet_512_best.pt
```

---

## 3. `target-scorer colab-train`

Orchestrates automated high-resolution training on Google Cloud GPUs using the Google Colab CLI. Provisions cloud accelerators (NVIDIA A100, H100, T4, L4, or G4), syncs the repository bundle, executes mixed-precision training, verifies B-2 and B-3 competition scorecards, downloads the best checkpoint locally, and tears down the VM to conserve compute units.

Source: [`src/target_scorer/colab_runner.py`](file:///home/antho/target_scorer/src/target_scorer/colab_runner.py)

### CLI Synopsis
```bash
target-scorer colab-train [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--canvas-size` | `int` | `800` | Resolution preset: `256`, `512`, `800`, or `1024`. (`1024` recommended for competition). |
| `--scale` | `float` | `4.0` | Pixels per millimeter scale. |
| `--epochs` | `int` | `15` | Total training epochs. |
| `--batch-size` | `int` | `8` | Mini-batch size on cloud GPU (`8` recommended for A100). |
| `--train-samples`| `int` | `400` | Training samples per epoch. |
| `--val-samples` | `int` | `50` | Validation samples per epoch. |
| `--lr` | `float` | `3.5e-4` | Initial learning rate. |
| `--gpu` | `str` | `A100` | Cloud GPU type: `A100`, `H100`, `T4`, `L4`, or `G4`. |
| `--high-mem` | flag | `False` | Request high-RAM cloud instance shape. |
| `--session` | `str` | `None` (auto) | Custom Colab session name. |
| `--checkpoint` | `str` | `None` (auto) | Local destination path for trained weights. Defaults to `weights/target_scoring_hrnet_{canvas}_best.pt`. |
| `--sigma` | `float` | `None` (auto) | Heatmap Gaussian kernel standard deviation ($3.2$ for 1024). |
| `--timeout` | `float` | `3600.0` | Maximum job execution timeout in seconds. |
| `--initial-weights`| `str` | `None` | Local `.pt` checkpoint to upload for warm-start fine-tuning in Colab. |
| `--real-data-dir`| `str` | `None` | Local directory containing real target annotations to package, upload, and blend during training. |
| `--real-ratio` | `float` | `0.35` | Real target sampling probability (`1.0` = pure real, `0.35` = hybrid). |
| `--list-sessions` | flag | `False` | Lists all active Colab cloud sessions and exits. |
| `--stop-session` | `str` | `None` | Terminates a specific running Colab session by name and exits. |

### Copy-Pasteable Shell Examples

#### A. Warm-Start Fine-Tuning on NVIDIA A100 ($1024 \times 1024$, ~6 minutes)
```bash
target-scorer colab-train \
  --canvas-size 1024 \
  --scale 4.0 \
  --epochs 12 \
  --batch-size 8 \
  --lr 1.5e-4 \
  --sigma 3.2 \
  --gpu A100 \
  --initial-weights weights/target_scoring_hrnet_1024_best.pt \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt
```

#### B. Cloud Hybrid Real Target Training (Option A: 70% Synthetic + 30% Real)
```bash
target-scorer colab-train \
  --canvas-size 1024 \
  --scale 4.0 \
  --epochs 15 \
  --batch-size 8 \
  --lr 1.0e-4 \
  --sigma 3.2 \
  --gpu A100 \
  --real-data-dir datasets/real_targets \
  --real-ratio 0.35 \
  --initial-weights weights/target_scoring_hrnet_1024_best.pt
```

#### C. Session Management
```bash
# Check current compute units and list active sessions:
target-scorer colab-train --list-sessions

# Force-stop an orphaned or lingering session:
target-scorer colab-train --stop-session target-train-1024-48210
```

---

## 4. `target-scorer colab-autolabel`

Executes cloud-accelerated assisted auto-labeling on an entire directory of target photographs using Google Colab CLI. Automatically packages images, uploads them to a remote GPU instance (default: NVIDIA T4), runs geometric rectification and inference, and downloads the generated JSON annotations and visual review cards.

Source: [`src/target_scorer/colab_runner.py`](file:///home/antho/target_scorer/src/target_scorer/colab_runner.py)

### CLI Synopsis
```bash
target-scorer colab-autolabel [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input-dir` | `str` | **Required** | Local directory containing target photographs (`.heic`, `.jpg`, `.png`, `.dng`, `.tiff`). |
| `--output-dir` | `str` | `datasets/autolabeled_targets` | Local directory to download annotations and review cards into. |
| `--target` | `str` | `B-2` | NRA target type: `B-2` or `B-3`. |
| `--checkpoint` | `str` | `weights/target_scoring_hrnet_1024_best.pt` | Path to local model checkpoint to upload to Colab. |
| `--canvas-size` | `int` | `1024` | Canonical canvas resolution. |
| `--scale` | `float` | `4.0` | Metric scale in pixels per millimeter. |
| `--threshold` | `float` | `0.40` | Peak detection confidence threshold. |
| `--gpu` | `str` | `T4` | Cloud GPU type (`T4` is optimal and cost-efficient for auto-labeling). |
| `--session` | `str` | `None` (auto) | Custom Colab session name. |
| `--timeout` | `float` | `1800.0` | Maximum execution timeout in seconds (default: 30 minutes). |
| `--redact-pii` | flag | `False` | Redacts competitor registration info on card margins before generating outputs. |

### Copy-Pasteable Shell Example
```bash
target-scorer colab-autolabel \
  --input-dir /home/user/range_photos/b3_rapid \
  --output-dir datasets/b3_autolabeled \
  --target B-3 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --gpu T4 \
  --threshold 0.38 \
  --redact-pii
```

---

## 5. `target-scorer rectify`

Executes standalone geometric rectification and perspective unwarping on raw target photographs across all formats. Detects the dark aiming black, fits a direct least-squares ellipse, computes a closed-form homography matrix, and saves an unwarped canonical crop alongside an optical diagnostic panel.

Source: [`src/target_scorer/geometry/rectifier.py`](file:///home/antho/target_scorer/src/target_scorer/geometry/rectifier.py)

### CLI Synopsis
```bash
target-scorer rectify [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--image` | `str` | **Required** | Path to target photograph (or `-` for `stdin`). Supports HEIC, RAW/DNG, WebP, PNG, JPEG, TIFF. |
| `--target` | `str` | `B-2` | NRA target type: `B-2` or `B-3`. |
| `--output` | `str` | `rectified_target.png` | Destination filepath for the unwarped canonical target crop. |
| `--debug` | `str` | `rectification_debug.png` | Destination filepath for the diagnostic visualization panel. Set to empty string to disable. |
| `--canvas-size` | `int` | `512` | Dimensions of canonical output square canvas in pixels. |
| `--scale` | `float` | `4.0` | Pixels per millimeter metric scale. |

### Copy-Pasteable Shell Examples

#### A. Rectify an Apple iPhone HEIC Photo
```bash
target-scorer rectify \
  --image smartphone_snaps/target_angle.heic \
  --target B-2 \
  --canvas-size 1024 \
  --scale 4.0 \
  --output rectified_b2_1024.png \
  --debug rect_diagnostic_b2.png
```

#### B. Rectify Camera RAW / Apple ProRAW DNG
```bash
target-scorer rectify \
  --image professional_photos/target_canon.dng \
  --target B-3 \
  --canvas-size 1024 \
  --scale 4.0 \
  --output rectified_b3_1024.png \
  --debug rect_diagnostic_b3.png
```

---

## 6. `target-scorer autolabel`

Assisted auto-labeling and active learning tool running locally. Predicts bullet hole coordinates and bullseye centers using a trained checkpoint, yielding standardized [`TargetAnnotation`](file:///home/antho/target_scorer/src/target_scorer/data/schema.py) JSON files, unwarped canonical crops, and human-in-the-loop verification review cards.

Source: [`src/target_scorer/data/autolabel.py`](file:///home/antho/target_scorer/src/target_scorer/data/autolabel.py)

### CLI Synopsis
```bash
target-scorer autolabel [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--image` | `str` | `None` | Path to a single target photograph to process. |
| `--input-dir` | `str` | `None` | Directory of target photos to recursively batch auto-label. |
| `--output-dir` | `str` | `datasets/real_targets` | Output directory where JSON annotations, canonical images, and review cards are saved. |
| `--target` | `str` | `B-2` | NRA target type: `B-2` or `B-3`. |
| `--checkpoint` | `str` | `weights/target_scoring_hrnet_1024_best.pt` | Path to model weights checkpoint. |
| `--canvas-size` | `int` | `1024` | Canonical canvas resolution. |
| `--scale` | `float` | `4.0` | Pixels per millimeter metric scale. |
| `--threshold` | `float` | `0.40` | Peak detection confidence threshold. |
| `--shots` | `int` | `10` | Maximum number of shots to detect per target. |
| `--redact-pii` | flag | `False` | Redacts competitor registration info on card margins. |

> [!NOTE]
> Either `--image` or `--input-dir` must be specified.

### Copy-Pasteable Shell Examples

#### A. Auto-Label a Single Target Photograph
```bash
target-scorer autolabel \
  --image raw_photos/relay_1_target.jpg \
  --target B-2 \
  --output-dir datasets/real_targets \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --canvas-size 1024 \
  --threshold 0.40 \
  --redact-pii
```

#### B. Batch Auto-Label a Range Photo Directory
```bash
target-scorer autolabel \
  --input-dir /home/user/range_photos/match_2026 \
  --output-dir datasets/match_2026_annotated \
  --target B-3 \
  --checkpoint weights/target_scoring_hrnet_1024_best.pt \
  --threshold 0.38 \
  --shots 10 \
  --redact-pii
```

---

## 7. `target-scorer annotate`

Launches the interactive desktop GUI for reviewing, adding, moving, and deleting bullet hole markers with real-time NRA inward scoring. Features smooth pan & zoom, NRA ring guides, caliber adjustment, and the optical **Eagle Eye Scoring Gauge**.

Source: [`src/target_scorer/gui/annotation_editor.py`](file:///home/antho/target_scorer/src/target_scorer/gui/annotation_editor.py)

### CLI Synopsis
```bash
target-scorer annotate [PATH] [OPTIONS]
```

### Options & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `path` | `str` (positional) | `None` | Path to directory containing JSON annotations (or single JSON file). If omitted, searches default locations (`autolabeled_photos`, `datasets/real_targets`). |
| `--dir` | `str` | `None` | Optional explicit flag for annotation directory path. |
| `--target` | `str` | `None` (auto) | Override NRA target type: `B-2` or `B-3`. If omitted, read from JSON metadata. |

### Copy-Pasteable Shell Examples

#### A. Open a Directory of Auto-Labeled Targets
```bash
target-scorer annotate datasets/match_2026_annotated
```

#### B. Open a Specific Target File with Forced B-3 Rules
```bash
target-scorer annotate datasets/real_targets/target_001.json --target B-3
```

---

## 8. Python API Reference

For developers embedding `target_scorer` into automated scoring pipelines, camera daemon loops, or web services.

### A. Inference & Metrology API (`target_scorer.infer`)

```python
from target_scorer.infer import run_inference_and_scoring

shot_results, metrics = run_inference_and_scoring(
    checkpoint_path="weights/target_scoring_hrnet_1024_best.pt",
    image_path="path/to/target.heic",  # Accepts str, Path, bytes, BinaryIO, or np.ndarray
    output_path="scorecard.png",
    target_type="B-2",                 # "B-2" or "B-3"
    distance_yards=16.67,              # 50 ft
    peak_threshold=0.45,
    max_shots=10,
    device_str="cuda",                 # "cuda", "cpu", or "mps"
    canvas_size=1024,
    pixels_per_mm=4.0,
    redact_pii=False,
)
```

#### Return Signatures:
* **`shot_results`**: `List[ShotResult]`
  * `shot_id: int`
  * `center_px: Tuple[float, float]`
  * `center_mm: Tuple[float, float]`
  * `radial_dist_mm: float`
  * `effective_gauge_dist_mm: float`
  * `score_label: str` (`"10"`, `"X"`, `"9"`, etc.)
  * `score_value: int`
  * `is_inner_x: bool`
  * `confidence: float`
* **`metrics`**: `GroupMetrics`
  * `total_shots: int`
  * `total_score: int`
  * `max_possible_score: int`
  * `x_count: int`
  * `extreme_spread_mm: float`
  * `extreme_spread_moa: float`
  * `mean_radius_mm: float`
  * `mean_radius_moa: float`
  * `center_of_impact_mm: Tuple[float, float]`
  * `windage_bias_mm: float`
  * `elevation_bias_mm: float`
  * `farthest_pair: Tuple[int, int]`
  * `target_distance_yards: float`

---

### B. In-Memory Zero-Disk Edge Camera Streaming

Ingest live video frames directly from camera buffers (e.g. Picamera2 or OpenCV capture) without writing to disk:

```python
import cv2
from target_scorer.geometry.rectifier import TargetRectifier
from target_scorer.infer import run_inference_and_scoring

rectifier = TargetRectifier(target_type="B-3", canonical_dim=1024, pixels_per_mm=4.0)

# Capture frame from video feed (H, W, 3) BGR array
cap = cv2.VideoCapture(0)
ret, frame_bgr = cap.read()

# 1. Unwarp perspective in memory
rect_result = rectifier.rectify(frame_bgr)
print(f"Detected tilt: {rect_result.tilt_angle_deg:.1f}°")

# 2. Run scoring directly on unwarped crop
shots, metrics = run_inference_and_scoring(
    checkpoint_path="weights/target_scoring_hrnet_1024_best.pt",
    image_path=rect_result.canonical_crop,
    target_type="B-3",
    canvas_size=1024,
    pixels_per_mm=4.0,
    output_path="edge_scorecard.png"
)
print(f"Score: {metrics.total_score} - {metrics.x_count}X")
```

---

### C. Data Schema Serialization (`target_scorer.data.schema`)

```python
from target_scorer.data.schema import TargetAnnotation, AnnotatedShot

# Create annotation object
annotation = TargetAnnotation(
    target_id="relay_12_card_01",
    image_filename="relay_12_card_01_canonical.png",
    target_type="B-3",
    bullseye_center=(512.0, 512.0),
    shots=[
        AnnotatedShot(id=1, center_x=515.2, center_y=510.4, score=10, is_x=True, caliber_mm=5.69),
        AnnotatedShot(id=2, center_x=495.0, center_y=520.1, score=10, is_x=False, caliber_mm=5.69),
    ],
    canvas_size=1024,
    pixels_per_mm=4.0,
)

# Serialize to JSON file
annotation.to_json("datasets/real_targets/relay_12_card_01.json")

# Deserialize from file
loaded = TargetAnnotation.from_json("datasets/real_targets/relay_12_card_01.json")
coords = loaded.get_coords_array()  # (N, 2) np.ndarray
```
