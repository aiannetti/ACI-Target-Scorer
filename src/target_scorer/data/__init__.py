"""
Data management, real-world target ingestion, active learning, and hybrid datasets.
"""

from target_scorer.data.autolabel import autolabel_target_image
from target_scorer.data.hybrid_dataset import HybridTargetDataset
from target_scorer.data.real_dataset import RealTargetDataset
from target_scorer.data.schema import AnnotatedShot, TargetAnnotation

__all__ = [
    "AnnotatedShot",
    "TargetAnnotation",
    "RealTargetDataset",
    "HybridTargetDataset",
    "autolabel_target_image",
]
