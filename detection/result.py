"""
Detection result types shared by runtime detectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DetectionResult:
    """Single target detection result."""

    center_px: tuple[float, float]
    bbox: tuple[int, int, int, int]
    angle_deg: float = 0.0
    label: str = ""
    confidence: float = 1.0
    contour: Optional[np.ndarray] = None

    @property
    def center_int(self) -> tuple[int, int]:
        return int(self.center_px[0]), int(self.center_px[1])

