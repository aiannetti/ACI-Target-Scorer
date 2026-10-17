"""
Hybrid dataset blending real-world annotated targets with procedural synthetic targets.
Supports pure real training (Option B), pure synthetic training, or balanced hybrid sampling (Option A).
"""

import random
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import Dataset

from target_scorer.config import TrainConfig
from target_scorer.data.real_dataset import RealTargetDataset
from target_scorer.synthetic.dataset import SyntheticTargetDataset


class HybridTargetDataset(Dataset):
    """
    Blends real annotated target photographs with procedural synthetic targets.

    Modes:
    - Pure Real (Option B): real_ratio = 1.0 (rapidly adapts to real range textures/lighting)
    - Hybrid Blended (Option A): 0.0 < real_ratio < 1.0 (preserves broad geometric diversity while anchoring to real fibers)
    - Pure Synthetic: real_ratio = 0.0
    """

    def __init__(
        self,
        real_data_dir: Optional[Union[str, Path]],
        config: TrainConfig,
        size: int,
        real_ratio: float = 0.35,
        is_train: bool = True,
        manifest_file: Optional[str] = None,
    ):
        self.size = size
        self.config = config
        self.real_ratio = float(np_clip_ratio(real_ratio))
        self.is_train = is_train

        self.synthetic_dataset = SyntheticTargetDataset(
            size=size,
            config=config,
            is_train=is_train
        )

        self.real_dataset: Optional[RealTargetDataset] = None
        if real_data_dir is not None and Path(real_data_dir).exists():
            try:
                self.real_dataset = RealTargetDataset(
                    data_dir=real_data_dir,
                    config=config,
                    is_train=is_train,
                    manifest_file=manifest_file,
                    virtual_size=size,
                )
            except Exception as e:
                if self.real_ratio >= 1.0:
                    raise
                print(f"[!] Warning: Could not initialize RealTargetDataset ({e}). Falling back to 100% synthetic.")
                self.real_dataset = None
                self.real_ratio = 0.0

        if self.real_dataset is None:
            self.real_ratio = 0.0

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.real_dataset is not None and (self.real_ratio >= 1.0 or random.random() < self.real_ratio):
            return self.real_dataset[idx]
        return self.synthetic_dataset[idx]


def np_clip_ratio(val: float) -> float:
    return max(0.0, min(1.0, val))
