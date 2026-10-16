"""
Official NRA inward scoring and ballistic group dispersion calculations.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from target_scorer.config import TargetRingsConfig


@dataclass
class ShotResult:
    """Diagnostic metrology record for a detected shot."""
    shot_id: int
    center_px: Tuple[float, float]
    center_mm: Tuple[float, float]
    radial_dist_mm: float
    effective_gauge_dist_mm: float
    score_label: str
    score_value: int
    is_inner_x: bool
    confidence: float


@dataclass
class GroupMetrics:
    """Statistical summary of shooting group dispersion and scoring."""
    total_shots: int
    total_score: int
    max_possible_score: int
    x_count: int
    extreme_spread_mm: float
    extreme_spread_moa: float
    mean_radius_mm: float
    mean_radius_moa: float
    center_of_impact_mm: Tuple[float, float]
    windage_bias_mm: float
    elevation_bias_mm: float
    farthest_pair: Tuple[int, int]
    target_distance_yards: float


def calculate_moa(size_mm: float, distance_yards: float) -> float:
    """Converts a physical measurement in millimeters to Minutes of Angle (MOA)."""
    if distance_yards <= 0:
        return 0.0
    one_moa_mm = distance_yards * math.tan(math.radians(1.0 / 60.0)) * 914.4
    return size_mm / one_moa_mm


def score_target_group(
    shot_peaks: List[Tuple[float, float, float]],
    bullseye_peak: Tuple[float, float],
    pixels_per_mm: float,
    bullet_diameter_mm: float = 5.69,
    rings_cfg: Optional[TargetRingsConfig] = None,
    distance_yards: float = 16.67
) -> Tuple[List[ShotResult], GroupMetrics]:
    """Applies official NRA inward scoring rules and computes group dispersion."""
    if pixels_per_mm <= 0:
        raise ValueError(f"pixels_per_mm must be positive, got {pixels_per_mm}")

    if rings_cfg is None:
        rings_cfg = TargetRingsConfig()

    bullet_radius_mm = bullet_diameter_mm / 2.0
    rings_table = rings_cfg.get_rings_table()
    bull_x, bull_y = bullseye_peak

    shot_results: List[ShotResult] = []

    for idx, (sx, sy, conf) in enumerate(shot_peaks, start=1):
        dx_px = sx - bull_x
        dy_px = sy - bull_y
        dist_px = math.hypot(dx_px, dy_px)

        dist_mm = dist_px / pixels_per_mm
        effective_gauge_dist_mm = max(0.0, dist_mm - bullet_radius_mm)

        assigned_label = "0"
        assigned_score = 0
        is_inner_x = False

        for label, ring_radius_mm, score_val in rings_table:
            if effective_gauge_dist_mm <= ring_radius_mm:
                assigned_label = label
                assigned_score = score_val
                if label == "X":
                    is_inner_x = True
                break

        coord_mm = (dx_px / pixels_per_mm, -dy_px / pixels_per_mm)

        shot_results.append(ShotResult(
            shot_id=idx,
            center_px=(sx, sy),
            center_mm=coord_mm,
            radial_dist_mm=dist_mm,
            effective_gauge_dist_mm=effective_gauge_dist_mm,
            score_label=assigned_label,
            score_value=assigned_score,
            is_inner_x=is_inner_x,
            confidence=conf,
        ))

    num_shots = len(shot_results)
    total_score = sum(s.score_value for s in shot_results)
    x_count = sum(1 for s in shot_results if s.is_inner_x)
    max_possible = num_shots * 10

    if num_shots == 0:
        empty_metrics = GroupMetrics(
            total_shots=0, total_score=0, max_possible_score=0, x_count=0,
            extreme_spread_mm=0.0, extreme_spread_moa=0.0, mean_radius_mm=0.0,
            mean_radius_moa=0.0, center_of_impact_mm=(0.0, 0.0),
            windage_bias_mm=0.0, elevation_bias_mm=0.0, farthest_pair=(0, 0),
            target_distance_yards=distance_yards
        )
        return shot_results, empty_metrics

    mean_x_mm = float(np.mean([s.center_mm[0] for s in shot_results]))
    mean_y_mm = float(np.mean([s.center_mm[1] for s in shot_results]))

    radii_from_coi = [math.hypot(s.center_mm[0] - mean_x_mm, s.center_mm[1] - mean_y_mm) for s in shot_results]
    mean_radius_mm = float(np.mean(radii_from_coi)) if radii_from_coi else 0.0

    max_dist_mm = 0.0
    farthest_pair = (shot_results[0].shot_id, shot_results[0].shot_id)

    for i in range(num_shots):
        for j in range(i + 1, num_shots):
            dist_ij = math.hypot(
                shot_results[i].center_mm[0] - shot_results[j].center_mm[0],
                shot_results[i].center_mm[1] - shot_results[j].center_mm[1]
            )
            if dist_ij > max_dist_mm:
                max_dist_mm = dist_ij
                farthest_pair = (shot_results[i].shot_id, shot_results[j].shot_id)

    metrics = GroupMetrics(
        total_shots=num_shots,
        total_score=total_score,
        max_possible_score=max_possible,
        x_count=x_count,
        extreme_spread_mm=max_dist_mm,
        extreme_spread_moa=calculate_moa(max_dist_mm, distance_yards),
        mean_radius_mm=mean_radius_mm,
        mean_radius_moa=calculate_moa(mean_radius_mm, distance_yards),
        center_of_impact_mm=(mean_x_mm, mean_y_mm),
        windage_bias_mm=mean_x_mm,
        elevation_bias_mm=mean_y_mm,
        farthest_pair=farthest_pair,
        target_distance_yards=distance_yards
    )

    return shot_results, metrics
