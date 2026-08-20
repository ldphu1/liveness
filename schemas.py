from dataclasses import dataclass
from typing import Optional

@dataclass
class FaceFeatures:
    yaw: float
    pitch: float

    left_ear: float
    right_ear: float
    avg_ear: float

    smile_ratio: float

    face_area_ratio: float

    bbox: tuple[int, int, int, int]

    quality_ok: bool

@dataclass
class HeadMovementState:
    yaw_baseline: Optional[float] = None
    pitch_baseline: Optional[float] = None

    detected_action: Optional[str] = None
    detected_confidence: float = 0.0

    hold_frames: int = 0

