"""
Configuration models and NRA target specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class TargetConfig:
    """Configuration parameters for synthetic target and heatmap generation."""
    canvas_size: int = 1024
    pixels_per_mm: float = 4.0
    bullet_diameter_mm: float = 5.69
    heatmap_sigma_px: float = 3.2
    ring_line_width_px: int = 2
    flute_direction: str = "vertical"
    grease_ring_width_mm: float = 0.45
    cardboard_base_bgr: Tuple[int, int, int] = (145, 180, 215)
    target_type: str = "B-2"             # Official NRA "B-2" or "B-3"
    include_handwriting: bool = True     # Simulate competitor handwriting (name, NRA#, relay)
    include_header_text: bool = True     # Simulate official printed NRA target headers
    include_scorer_marks: bool = True    # Simulate scorer circled shots, numbers & checkmarks
    include_nra_seal: bool = True        # Simulate authentic official printed circular NRA seal in lower margin
    void_color_mode: str = "mixed"       # "mixed" (45% scanner white, 35% dark, 20% tan), "scanner_white", "dark_void", "tan_backer"


@dataclass
class TrainConfig:
    """Hyperparameter and hardware configuration for heatmap training."""
    canvas_size: int = 512
    pixels_per_mm: float = 4.0
    bullet_diameter_mm: float = 5.69
    heatmap_sigma_px: float = 2.5
    train_samples_per_epoch: int = 400
    val_samples_per_epoch: int = 50
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    epochs: int = 15
    learning_rate: float = 3.5e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    peak_threshold: float = 0.45
    num_workers: int = 2
    pin_memory: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = True
    awing_omega: float = 14.0
    awing_theta: float = 0.5
    awing_epsilon: float = 1.0
    awing_alpha: float = 2.1


if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([TrainConfig])

# Guarantee pickle resolution of __main__.TrainConfig regardless of entry point
import sys
_main_mod = sys.modules.get("__main__")
if _main_mod is not None and not hasattr(_main_mod, "TrainConfig"):
    setattr(_main_mod, "TrainConfig", TrainConfig)


@dataclass
class ShotAnnotation:
    """Ground truth annotation record for an individual bullet impact."""
    center_x: float
    center_y: float
    shot_type: str                       # 'single', 'double_primary', 'double_secondary', 'keyhole', 'bullseye'
    caliber_radius_px: float
    yaw_angle_deg: float = 0.0
    aspect_ratio: float = 1.0


@dataclass
class TargetRingsConfig:
    """
    Official NRA precision pistol target specifications in millimeters.
    Supports:
      - 'B-2': 50 ft Slow Fire (7, 8, 9, 10 black; 6, 5, 4 light)
      - 'B-3': 50 ft Timed & Rapid Fire (9, 10, X black; 8, 7, 6 light)
    """
    target_type: str = "B-2"

    # Official NRA B-2 (50 ft Slow Fire, Rule 4.2): Radii in mm (inches * 25.4 / 2)
    # The innermost ring is the 10 ring (0.90 in dia). No X-ring at 50ft slow fire.
    b2_10_mm: float = 11.43         # 0.90 in dia (innermost ring)
    b2_9_mm: float = 19.558         # 1.54 in dia
    b2_8_mm: float = 28.321         # 2.23 in dia
    b2_7_mm: float = 38.989         # 3.07 in dia (black edge: 7, 8, 9, 10 are black)
    b2_6_mm: float = 52.832         # 4.16 in dia
    b2_5_mm: float = 70.612         # 5.56 in dia
    b2_4_mm: float = 93.091         # 7.33 in dia

    # Official NRA B-3 (50 ft Timed and Rapid Fire, Rule 4.3): Radii in mm
    # 9 and 10 rings black (3.06 in dia). Innermost ring is X ring (0.90 in dia).
    b3_x_mm: float = 11.43          # 0.90 in dia (X ring)
    b3_10_mm: float = 22.86         # 1.80 in dia (10 ring)
    b3_9_mm: float = 38.862         # 3.06 in dia (black edge)
    b3_8_mm: float = 56.642         # 4.46 in dia
    b3_7_mm: float = 77.978         # 6.14 in dia
    b3_6_mm: float = 105.664        # 8.32 in dia

    def get_rings_table(self) -> List[Tuple[str, float, int]]:
        """Returns sorted list of (label, radius_mm, score_value)."""
        if self.target_type.upper() == "B-3":
            return [
                ("X", self.b3_x_mm, 10),
                ("10", self.b3_10_mm, 10),
                ("9", self.b3_9_mm, 9),
                ("8", self.b3_8_mm, 8),
                ("7", self.b3_7_mm, 7),
                ("6", self.b3_6_mm, 6),
            ]
        return [
            ("10", self.b2_10_mm, 10),
            ("9", self.b2_9_mm, 9),
            ("8", self.b2_8_mm, 8),
            ("7", self.b2_7_mm, 7),
            ("6", self.b2_6_mm, 6),
            ("5", self.b2_5_mm, 5),
            ("4", self.b2_4_mm, 4),
        ]

    def get_black_radius_mm(self) -> float:
        """Returns the physical radius of the solid black bullseye."""
        return self.b3_9_mm if self.target_type.upper() == "B-3" else self.b2_7_mm
