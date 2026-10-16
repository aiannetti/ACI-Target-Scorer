Production Plan: 100% Camera-Independent Target Scoring1. Executive Summary & Problem DiagnosisThe current prototype relies on two rigid assumptions that fail in real-world smartphone photos:Geometric Scale Invariance Failure: The metrology engine hardcodes pixels_per_mm = 4.0. If a user shoots from 14 inches away versus 24 inches away, a 1-pixel detection error corresponds to completely different physical millimeter deviations, invalidating the NRA scoring rings.Perspective & Aspect Distortion Failure: Resizing arbitrary rectangular photos (4:3 or 16:9) directly to $512 \times 512$ distorts circles into non-uniform ellipses. Furthermore, tilting the camera introduces perspective keystoning (parallax), shifting radial distances non-linearly.The ObjectiveTransform the pipeline into a closed-loop canonical normalization architecture: regardless of camera distance, orientation, aspect ratio, optical tilt (up to $\pm 35^\circ$), or off-center framing, the front-end will mathematically rectify and re-scale the target into an exact metric-calibrated canonical plane before the neural network ever touches it.Raw Smartphone Photo (Any Aspect, Tilted, Distant, Angled)
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 1: Optical Quality & Pre-Filtering Gate          │
 │  - EXIF orientation auto-rotation                       │
 │  - Motion blur validation (Laplacian variance > 120)    │
 │  - Specular glare / flash reflection suppression        │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 2: Classical Geometric Normalization & Pose Est.  │
 │  - Multi-scale Bullseye segmentation (Otsu + HSV mask) │
 │  - Direct Least-Squares Ellipse Fitting (Taubin/Fitzg.) │
 │  - Camera Normal Vector & Tilt Angles (Pitch, Yaw, Roll)│
 │  - Closed-form 3x3 Homography Matrix H                  │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 3: Metric Auto-Calibration & Canonical Cropping   │
 │  - Match detected bullseye radius to official NRA specs:│
 │      * B-2: 38.989 mm physical radius                   │
 │      * B-3: 38.862 mm physical radius                   │
 │  - Warp perspective to canonical 512x512 @ 4.0 px/mm    │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 4: High-Res Heatmap U-Net Inference               │
 │  - Network sees 100% normalized, centered, flat target  │
 │  - Outputs sub-pixel bullet coordinates in canonical mm │
 └─────────────────────────┬───────────────────────────────┘
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 5: Inward Scoring & Ballistics Metrology HUD      │
 │  - Official NRA ring table evaluation                   │
 │  - Metric overlay and Extreme Spread calculations       │
 └─────────────────────────────────────────────────────────┘
2. Component SpecificationsComponent A: Optical Quality & Pre-Filtering GateSmartphone captures suffer from three common sensor artifacts: wrong EXIF rotation, hand-shake blur, and phone flash glare.EXIF Orientation Normalization:Mobile image headers store rotation flags (Tag 274). If ignored, OpenCV opens photos sideways or upside down.Action: Intercept EXIF metadata and apply lossless transpose/flip matrices before any array allocations.Motion Blur Gate:Evaluate the variance of the 2D Laplacian operator across the image:$$\text{Blur Metric} = \mathrm{Var}\left( \nabla^2 I \right)$$If $\text{Blur Metric} < 100.0$, reject the frame with user feedback: "Target blurry; steady camera."Specular Glare Suppression:Range overhead lighting often creates high-specular hotspots on black target ink.Action: Identify clipping white pixels ($L^* > 245$ in CIE $L^*a^*b^*$) inside high-density dark regions and interpolate them using fast Navier-Stokes inpainting (cv2.inpaint).Component B: Perspective Rectification via Ellipse DecompositionWhen an orthogonal circular target of radius $R$ is viewed obliquely by a camera at an angle $\theta$, its projection on the image sensor is an ellipse.    Camera Line of Sight
            \  θ (Pitch / Tilt Angle)
             \
              ▼
       ╭─────────────╮
      │   Ellipse     │   Major Axis (2a) = Proportional to Distance
       \             /    Minor Axis (2b) = a * cos(θ)
        ╰───────────╯
              │
              ▼
   [Homography Matrix H] ──> Flattens to True Circle (Radius R)
1. Robust Bullseye ExtractionDownscale the raw image to an analysis pyramid level (width $\approx 1200\text{ px}$).Segment the dark aiming black:Convert to CIE $L^*a^*b^*$ and extract the lightness channel $L^*$.Apply multi-threshold Otsu segmentation combined with an adaptive black-hat morphological transform to isolate the dark disc regardless of ambient lighting gradients.Extract external contours and reject non-bullseye candidates:Area Check: Must encompass between $1.5\%$ and $50\%$ of the image area.Convexity: Convexity ratio $> 0.92$.Inertia / Aspect Ratio: Ratio of minor-to-major axis $b / a \in [0.45, 1.0]$ (tolerating tilt up to $\arccos(0.45) \approx 63^\circ$).2. Sub-Pixel Ellipse Fitting (Direct Least-Squares)Fit an algebraic conic section $A x^2 + B x y + C y^2 + D x + E y + F = 0$ constrained by $4AC - B^2 = 1$ (Fitzgibbon-Pilu-Fisher formulation) to the boundary points of the segmented bullseye.This returns continuous ellipse parameters:$$\mathbf{E} = \left( (x_c, y_c), (2a, 2b), \phi \right)$$where $(x_c, y_c)$ is the projected center, $a$ is the semi-major axis, $b$ is the semi-minor axis, and $\phi$ is the tilt orientation angle.3. Closed-Form Homography EstimationFrom the fitted ellipse parameters:The tilt angle (camera pitch off-normal) is:$$\theta = \arccos\left(\frac{b}{a}\right)$$The major axis $2a$ is completely uncompressed by perspective tilt and represents the true unforeshortened physical diameter in the camera plane.We define four synthetic virtual points on the fitted ellipse along its major and minor axes:$$\mathbf{p}_1, \mathbf{p}_2 = (x_c, y_c) \pm a \begin{bmatrix} \cos\phi \\ \sin\phi \end{bmatrix}$$$$\mathbf{p}_3, \mathbf{p}_4 = (x_c, y_c) \pm b \begin{bmatrix} -\sin\phi \\ \cos\phi \end{bmatrix}$$We define the corresponding canonical circular points where radius $R_{\text{norm}} = a$:$$\mathbf{q}_1, \mathbf{q}_2 = (X_c, Y_c) \pm R_{\text{norm}} \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$$$\mathbf{q}_3, \mathbf{q}_4 = (X_c, Y_c) \pm R_{\text{norm}} \begin{bmatrix} 0 \\ 1 \end{bmatrix}$$Solve for the $3 \times 3$ Homography matrix $H$:$$H = \mathrm{findHomography}(\{\mathbf{p}_i\}, \{\mathbf{q}_i\}, \mathrm{RANSAC})$$Apply cv2.warpPerspective using $H$ to transform the tilted phone photo into a perfectly flat, square canonical plane.Component C: Metric Auto-Calibration ContractOnce the target is flattened, the scale is locked to physical reality using official NRA specifications.Target TypeOfficial NRA Aiming BlackPhysical Bullseye Radius (Rnominal​)Canonical Frame Radius (Rcanon​)NRA B-2 (50 ft Slow Fire)Rings 7, 8, 9, 10$38.989\text{ mm}$ ($3.07''$ dia / 2)$155.96\text{ px}$ (@ 4.0 px/mm)NRA B-3 (50 ft Timed/Rapid)Rings 9, 10, X$38.862\text{ mm}$ ($3.06''$ dia / 2)$155.45\text{ px}$ (@ 4.0 px/mm)Exact Scaling FormulaFrom the de-tilted target, the observed semi-major axis $a$ (in raw pixels) corresponds to $R_{\text{nominal}}$:$$S_{\text{raw}} = \frac{a}{R_{\text{nominal}}} \quad [\text{raw pixels per mm}]$$To feed the High-Res U-Net at its trained resolution ($512 \times 512$ with $4.0\text{ px/mm}$), the required physical field of view is:$$\text{FOV} = \frac{512\text{ px}}{4.0\text{ px/mm}} = 128.0\text{ mm}$$The bounding box in the raw perspective-corrected image centered at the bullseye $(x_c, y_c)$ has a half-width of:$$W_{\text{crop\_raw}} = 64.0\text{ mm} \times S_{\text{raw}} = \frac{64.0}{R_{\text{nominal}}} \times a$$Extracting and resampling this region to $512 \times 512$ guarantees that every target presented to the neural network has an exact scale of $4.000\text{ pixels/mm}$, regardless of camera sensor resolution, zoom, or shooting distance.Component D: Synthetic Training Pipeline AugmentationsEven with front-end rectification, small residual distortions ($\pm 2\text{–}3^\circ$ tilt error or $\pm 5\text{ px}$ crop miscentering) will occur in edge cases. The neural network must be trained to be invariant to these residuals.Modify SyntheticTargetDataset in train_heatmap_model.py with an Albumentations / OpenCV projective pipeline:# Procedural Augmentations to add to SyntheticTargetDataset:
1. Random Affine Translation:
   Shift bullseye center by dx, dy ~ Uniform(-32 px, +32 px)

2. Scale Jitter:
   Scale pixels_per_mm by factor s ~ Uniform(0.92, 1.08)

3. Residual Perspective Perturbation:
   Apply random 4-point homography tilt with corners shifted up to +/- 12 px

4. Photometric Jitter:
   Random Brightness (+/- 25%), Contrast (+/- 20%), Gaussian Sensor Noise (sigma=4.0)

5. Background Color Jitter:
   Vary cardboard backing from light kraft tan to dark recycled corrugated brown
3. Architecture of the Modified CodebaseThe software architecture will be partitioned into three modular stages:target_scorer/
├── target_rectifier.py         <-- NEW: Auto-detection, ellipse fitting, homography rectification
├── train_heatmap_model.py      <-- UPDATED: Adds geometric & photometric data augmentations
├── infer_and_score.py          <-- UPDATED: Integrates target_rectifier before U-Net inference
└── tests/
    └── test_camera_angles.py   <-- NEW: Stress tests on tilted synthetic/real photos
Detailed Class Design: TargetRectifier (target_rectifier.py)class TargetRectifier:
    """
    Automated pre-inference geometric normalizer.
    Locates the target bullseye in raw photos, calculates the de-tilting
    homography, and extracts a metric-calibrated 512x512 canonical crop.
    """
    def __init__(self, target_type: str = "B-2", canonical_dim: int = 512, pixels_per_mm: float = 4.0):
        self.target_type = target_type.upper()
        self.canonical_dim = canonical_dim
        self.pixels_per_mm = pixels_per_mm
        self.nominal_radius_mm = 38.862 if self.target_type == "B-3" else 38.989

    def process_raw_image(self, raw_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Input: Raw phone photo (any resolution, any aspect ratio)
        Output:
          - canonical_crop: (512, 512, 3) image with 4.0 px/mm scale, centered bullseye
          - transform_meta: Homography matrix, tilt angles, detected scale, and confidence
        """:
        # 1. Optical gate & pre-scaling
        # 2. Bullseye segmentation & contour filtering
        # 3. Direct least-squares ellipse fit
        # 4. Homography matrix derivation
        # 5. Perspective warp & metric 512x512 crop extraction
        ...
4. Implementation Phasing & MilestonesSprintPhaseKey TasksAcceptance CriteriaSprint 1TargetRectifier EngineImplement Otsu/HSV contour extraction, Fitzgibbon ellipse fitting, and closed-form 4-point homography extraction.Correctly flattens targets tilted up to $45^\circ$; circularity of rectified bullseye $> 0.98$.Sprint 2Metric Scale CalibrationConnect detected ellipse major axis to nominal NRA dimensions ($38.99\text{ mm}$ for B-2, $38.86\text{ mm}$ for B-3).Known caliber plug gauges measure within $\pm 0.10\text{ mm}$ across distances ranging from 12" to 36".Sprint 3Training Pipeline AugmentationsAdd random homography, center translations ($\pm 32\text{ px}$), and scale jitter ($\pm 8\%$) to train_heatmap_model.py.Validation MRE maintains $< 0.15\text{ mm}$ under simulated residual tilt and offset.Sprint 4End-to-End IntegrationWire TargetRectifier into infer_and_score.py; add visual debug diagnostics showing the detected ellipse and rectified image.Full CLI execution: python infer_and_score.py --image phone_photo.jpg runs completely unattended.Sprint 5Field ValidationCollect 50 smartphone photos from 3 different phones across varying angles ($0^\circ, 15^\circ, 30^\circ, 45^\circ$) and lighting conditions.$> 98\%$ successful automatic target lock; scoring matches human visual inspection with plug gauges.5. Failure Modes & Edge-Case FallbacksSevere Occlusion / Torn Center:Problem: A tight cluster of 10 shots in the X-ring completely destroys the center, distorting the interior black.Fallback: The ellipse fitting algorithm samples edge points from the outer boundary of the black bullseye (the 7-ring line on B-2, 9-ring line on B-3). Shots inside the X/10 ring leave the outer circumference intact.Ambiguous Lighting / Shadows:Problem: A harsh shadow cuts halfway across the target face at an outdoor range.Fallback: If Otsu global thresholding fails, fall back to localized adaptive thresholding (cv2.adaptiveThreshold with a Gaussian kernel size of 1/8 the image width).Extreme Oblique Angle ($> 55^\circ$):Problem: When tilt exceeds $55^\circ$, perspective foreshortening severely compresses the minor axis, causing loss of spatial resolution on vertical edges.Fallback: The rectifier calculates the tilt angle $\theta = \arccos(b/a)$. If $\theta > 45^\circ$, the system triggers a warning: "Camera angle too steep. Please hold phone flatter over target."
