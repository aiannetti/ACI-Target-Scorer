"""
Target annotation schemas and serialization routines for real target datasets.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class AnnotatedShot:
    """Individual bullet impact annotation."""
    id: int
    center_x: float
    center_y: float
    score: int
    is_x: bool = False
    caliber_mm: float = 5.69
    shot_type: str = "single"  # 'single', 'double', 'keyhole'
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotatedShot":
        return cls(
            id=int(data.get("id", 1)),
            center_x=float(data.get("center_x", data.get("x", 0.0))),
            center_y=float(data.get("center_y", data.get("y", 0.0))),
            score=int(data.get("score", 0)),
            is_x=bool(data.get("is_x", False)),
            caliber_mm=float(data.get("caliber_mm", 5.69)),
            shot_type=str(data.get("shot_type", "single")),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class TargetAnnotation:
    """Complete target card ground-truth annotation."""
    target_id: str
    image_filename: str
    target_type: str  # 'B-2' or 'B-3'
    bullseye_center: Tuple[float, float]
    shots: List[AnnotatedShot] = field(default_factory=list)
    canvas_size: int = 1024
    pixels_per_mm: float = 4.0
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_score(self) -> int:
        return sum(s.score for s in self.shots)

    @property
    def x_count(self) -> int:
        return sum(1 for s in self.shots if s.is_x)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes annotation to dictionary."""
        return {
            "version": self.version,
            "target_id": self.target_id,
            "image_filename": self.image_filename,
            "canvas_size": self.canvas_size,
            "pixels_per_mm": self.pixels_per_mm,
            "target_type": self.target_type,
            "bullseye_center": list(self.bullseye_center),
            "shots": [asdict(s) for s in self.shots],
            "metadata": self.metadata,
        }

    def to_json(self, destination: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        """Serializes annotation to JSON string or saves to file destination."""
        payload = json.dumps(self.to_dict(), indent=indent)
        if destination is not None:
            dest_path = Path(destination)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "w", encoding="utf-8") as f:
                f.write(payload)
        return payload

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetAnnotation":
        """Deserializes TargetAnnotation from dictionary."""
        shots = [
            AnnotatedShot.from_dict(s) if isinstance(s, dict) else s
            for s in data.get("shots", [])
        ]

        center = data.get("bullseye_center", [data.get("canvas_size", 1024) / 2.0] * 2)

        return cls(
            target_id=str(data.get("target_id", "unnamed_target")),
            image_filename=str(data.get("image_filename", "")),
            target_type=str(data.get("target_type", "B-2")),
            bullseye_center=(float(center[0]), float(center[1])),
            shots=shots,
            canvas_size=int(data.get("canvas_size", 1024)),
            pixels_per_mm=float(data.get("pixels_per_mm", 4.0)),
            version=str(data.get("version", "1.0")),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, source: Union[str, Path]) -> "TargetAnnotation":
        """Loads TargetAnnotation from a JSON file path or raw JSON string."""
        source_str = str(source).strip()
        if source_str.startswith("{"):
            data = json.loads(source_str)
        else:
            path = Path(source)
            if path.exists() and path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(source_str)
        return cls.from_dict(data)

    def get_coords_array(self) -> np.ndarray:
        """Returns (N, 2) float32 NumPy array of continuous shot center coordinates."""
        if not self.shots:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array([[s.center_x, s.center_y] for s in self.shots], dtype=np.float32)

    def to_numpy_coords(self) -> np.ndarray:
        """Returns (N, 2) float32 NumPy array of continuous shot center coordinates."""
        return self.get_coords_array()
