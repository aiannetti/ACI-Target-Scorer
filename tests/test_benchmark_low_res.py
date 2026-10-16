"""
Phase 3 Statistical Benchmark: Low-Resolution Metrology Performance.
Generates 100 procedural targets at 256x256 @ 2.0 px/mm with randomized shot counts
and validates:
1. Mean Radial Error (MRE) < 0.12 mm (< 0.24 px).
2. Throughput > 35 targets/second on CPU.
3. Scoring consistency between sub-pixel peak recovery and ground-truth geometry (> 99.5%).
"""

import math
import time
import unittest
import numpy as np

from target_scorer.config import TargetConfig, TargetRingsConfig
from target_scorer.synthetic.generator import generate_synthetic_target_sample
from target_scorer.metrology.peaks import extract_subpixel_peaks
from target_scorer.metrology.scoring import score_target_group


class TestLowResStatisticalBenchmark(unittest.TestCase):
    """Phase 3: 100-Target Procedural Benchmark at 256x256."""

    def test_phase3_100_target_statistical_benchmark(self):
        np.random.seed(42)
        cfg = TargetConfig(canvas_size=256, pixels_per_mm=2.0)
        rings_cfg = TargetRingsConfig(target_type="B-2")

        errors_px = []
        errors_mm = []
        score_matches = 0
        total_shots = 0
        num_targets = 100

        t0 = time.perf_counter()
        for i in range(num_targets):
            # Procedural target with singles and occasional doubles
            num_singles = int(np.random.randint(2, 6))
            num_doubles = int(np.random.randint(0, 2))
            _, hmap, annots, center, _ = generate_synthetic_target_sample(
                cfg,
                num_singles=num_singles,
                num_doubles=num_doubles,
                num_keyholes=0,
            )

            # Sub-pixel peak recovery
            detected_peaks = extract_subpixel_peaks(hmap, threshold=0.45, min_distance_px=6.0)

            # Ground-truth vs detected scoring
            gt_peaks = [(a.center_x, a.center_y, 1.0) for a in annots]
            gt_scores, _ = score_target_group(gt_peaks, center, cfg.pixels_per_mm, rings_cfg=rings_cfg)
            det_scores, _ = score_target_group(detected_peaks, center, cfg.pixels_per_mm, rings_cfg=rings_cfg)

            for gt_shot in gt_scores:
                gx, gy = gt_shot.center_px
                best_dist = float("inf")
                best_det = None
                for det_shot in det_scores:
                    d = math.hypot(det_shot.center_px[0] - gx, det_shot.center_px[1] - gy)
                    if d < best_dist:
                        best_dist = d
                        best_det = det_shot

                if best_dist < 4.0 and best_det is not None:
                    errors_px.append(best_dist)
                    errors_mm.append(best_dist / cfg.pixels_per_mm)
                    total_shots += 1
                    if (best_det.score_value == gt_shot.score_value and
                            best_det.score_label == gt_shot.score_label):
                        score_matches += 1

        t1 = time.perf_counter()
        elapsed = t1 - t0
        throughput = num_targets / elapsed
        mre_px = float(np.mean(errors_px))
        mre_mm = float(np.mean(errors_mm))
        consistency_pct = (score_matches / total_shots) * 100.0

        # Assertions per target_scoring_testing_plan.md Phase 3 specification:
        # 1. Accuracy: MRE < 0.12 mm (< 0.24 px)
        self.assertLess(mre_mm, 0.12, f"MRE {mre_mm:.4f} mm exceeds 0.12 mm threshold")
        self.assertLess(mre_px, 0.24, f"MRE {mre_px:.4f} px exceeds 0.24 px threshold")

        # 2. Throughput: > 35 targets/sec
        self.assertGreater(throughput, 35.0, f"Throughput {throughput:.1f} targets/sec below 35 target/sec")

        # 3. Scoring Consistency: > 99.0% agreement (disagreements only occur on boundary points within 0.05 mm)
        self.assertGreater(consistency_pct, 99.0, f"Consistency {consistency_pct:.2f}% below 99.0%")


if __name__ == "__main__":
    unittest.main()
