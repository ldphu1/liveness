import time
from typing import Optional

import cv2
import numpy as np
from collections import deque
import mediapipe as mp

from schemas import HeadMovementState, FaceFeatures
from config import *
from metrics import _calculate_bbox, _calculate_ear, _calculate_smile_ratio, _calculate_head_pose, _calculate_face_area_ratio
from detector import FaceDetector


def _analyze_single_frame(image: np.ndarray, _detector) -> dict:
    if image is None:
        return {
            "status": "INVALID_IMAGE",
            "face_count": 0,
        }

    if image.ndim != 3:
        return {
            "status": "INVALID_IMAGE",
            "face_count": 0,
        }

    height, width = image.shape[:2]

    if width <= 0 or height <= 0:
        return {
            "status": "INVALID_IMAGE",
            "face_count": 0,
        }

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = _detector.detect(mp_image)

    face_count = len(result.face_landmarks)

    if face_count == 0:
        return {
            "status": "NO_FACE",
            "face_count": 0,
        }

    # Liveness một người:
    # nếu có >1 face thì reject frame
    if face_count > 1:
        return {
            "status": "MULTIPLE_FACES",
            "face_count": face_count,
        }

    landmarks = result.face_landmarks[0]

    # -------------------------------------------------------------------------
    # BBox
    # -------------------------------------------------------------------------
    bbox = _calculate_bbox(landmarks, width, height)

    face_area_ratio = _calculate_face_area_ratio(bbox, width, height)

    # -------------------------------------------------------------------------
    # Eye aspect ratio
    # -------------------------------------------------------------------------
    left_ear = _calculate_ear(landmarks, LEFT_EYE, width, height)

    right_ear = _calculate_ear(landmarks, RIGHT_EYE, width, height)

    avg_ear = (left_ear + right_ear) / 2.0

    # -------------------------------------------------------------------------
    # Smile
    # -------------------------------------------------------------------------

    smile_ratio = _calculate_smile_ratio(landmarks, width, height)

    # -------------------------------------------------------------------------
    # Head pose
    # -------------------------------------------------------------------------

    if (result.facial_transformation_matrixes and len(result.facial_transformation_matrixes)) > 0:
        yaw, pitch = _calculate_head_pose( result.facial_transformation_matrixes[0])
    else:
        yaw = 0.0
        pitch = 0.0

    # -------------------------------------------------------------------------
    # Quality
    # -------------------------------------------------------------------------

    # blur_score = _calculate_blur(image)
    blur_score = 0.0

    # quality_ok = (face_area_ratio >= MIN_FACE_AREA_RATIO and blur_score >= MIN_LAPLACIAN_VARIANCE)
    quality_ok = True

    features = FaceFeatures(
        yaw=yaw,
        pitch=pitch,

        left_ear=left_ear,
        right_ear=right_ear,
        avg_ear=avg_ear,

        smile_ratio=smile_ratio,

        face_area_ratio=face_area_ratio,

        bbox=bbox,

        quality_ok=quality_ok,
    )

    return {
        "status": "SUCCESS",
        "face_count": face_count,
        "features": features,
        "blur_score": blur_score,
    }


# =============================================================================
# BLINK DETECTION
# =============================================================================
def _detect_blink(results: list[dict]) -> Optional[float]:
    if len(results) < 3:
        return None

    ears = [r["features"].avg_ear for r in results]
    min_ear = min(ears)
    max_ear = max(ears)

    if min_ear <= EAR_CLOSED_THRESHOLD and max_ear >= EAR_OPEN_THRESHOLD:
        confidence = float(np.clip((EAR_OPEN_THRESHOLD - min_ear) / 0.1, 0.6, 1.0))
        return confidence

    return None

# =============================================================================
# SMILE DETECTION
# =============================================================================
def _detect_smile(results: list[dict]) -> Optional[float]:
    if len(results) < 3:
        return None

    ratios = [r["features"].smile_ratio for r in results]
    current_ratio = float(np.median(ratios[-2:]))
    start_ratio = float(np.median(ratios[:2]))
    delta = current_ratio - start_ratio

    if (current_ratio >= SMILE_RATIO_THRESHOLD and delta >= SMILE_DELTA_THRESHOLD) or current_ratio >= (SMILE_RATIO_THRESHOLD + 0.04):
        confidence = float(np.clip(current_ratio / SMILE_RATIO_THRESHOLD, 0.5, 1.0))
        return confidence

    return None



class ActiveLivenessAnalyzer:
    def __init__(self, _detector: FaceDetector, buffer_size: int = TEMPORAL_BUFFER_SIZE):
        self.buffer_size = buffer_size

        # Chỉ lưu kết quả/features, k lưu raw frame
        self.feature_buffer = deque(maxlen=buffer_size)

        self.head_state = HeadMovementState()
        self._detector = _detector

    # -------------------------------------------------------------------------
    # RESET
    # -------------------------------------------------------------------------
    def reset(self):
        self.feature_buffer.clear()

        self.head_state = HeadMovementState()

    # -------------------------------------------------------------------------
    # ADD FEATURE
    # -------------------------------------------------------------------------
    def _append_result(self, result: dict):

        if result["status"] != "SUCCESS":
            return

        features = result["features"]

        if not features.quality_ok:
            return

        self.feature_buffer.append(
            result
        )

    # =============================================================================
    # HEAD MOVEMENT
    # =============================================================================
    def _detect_head_action(self) -> tuple[Optional[str], float]:
        results = list(self.feature_buffer)
        if len(results) < 3:
            return None, 0.0

        yaws = np.asarray([r["features"].yaw for r in results], dtype=np.float32)
        pitches = np.asarray([r["features"].pitch for r in results], dtype=np.float32)

        # Baseline tính theo 3 frame đầu tiên
        if self.head_state.yaw_baseline is None:
            self.head_state.yaw_baseline = float(np.median(yaws[:min(3, len(yaws))]))
        if self.head_state.pitch_baseline is None:
            self.head_state.pitch_baseline = float(np.median(pitches[:min(3, len(pitches))]))

        # Lấy giá trị góc mượt mà ở các frame gần nhất
        yaw_current = float(np.median(yaws[-3:]))
        pitch_current = float(np.median(pitches[-3:]))

        yaw_delta = yaw_current - self.head_state.yaw_baseline
        pitch_delta = pitch_current - self.head_state.pitch_baseline

        action = None
        confidence = 0.0

        threshold = HEAD_ACTION_THRESHOLD
        abs_yaw = abs(yaw_delta)
        abs_pitch = abs(pitch_delta)

        # Chỉ kích hoạt nếu vượt ngưỡng và chọn trục có biên độ chuyển động lớn hơn
        if max(abs_yaw, abs_pitch) >= threshold:
            if abs_pitch > abs_yaw:
                # Trục dọc
                if pitch_delta <= -threshold:
                    action = "turn_up"
                    confidence = min(abs_pitch / (threshold * 2.0), 1.0)
                elif pitch_delta >= threshold:
                    action = "turn_down"
                    confidence = min(abs_pitch / (threshold * 2.0), 1.0)
            else:
                # Trục ngang
                if yaw_delta >= threshold:
                    action = "turn_left"
                    confidence = min(abs_yaw / (threshold * 2.0), 1.0)
                elif yaw_delta <= -threshold:
                    action = "turn_right"
                    confidence = min(abs_yaw / (threshold * 2.0), 1.0)

        if action is not None:
            self.head_state.detected_action = action
            self.head_state.detected_confidence = float(confidence)
            self.head_state.hold_frames = HEAD_RESULT_HOLD_FRAMES
            return action, float(confidence)

        if self.head_state.hold_frames > 0:
            self.head_state.hold_frames -= 1
            return self.head_state.detected_action, self.head_state.detected_confidence

        self.head_state.detected_action = None
        return None, 0.0

    #api cho từng frame
    def process_frame(self, frame: np.ndarray) -> dict:
        start_time = time.perf_counter()

        result = _analyze_single_frame(frame, self._detector)

        # -------------------------------------------------------------------------
        # INVALID / NO FACE / MULTIPLE FACE
        # -------------------------------------------------------------------------

        if result["status"] != "SUCCESS":
            return {
                "detected_action": None,
                "confidence": 0.0,
                "face_detected": (result["face_count"] > 0),
                "face_count": result["face_count"],
                "bbox": None,
                "processing_ms": round(
                    (time.perf_counter() - start_time)
                    * 1000.0,
                    2,
                ),
            }

        features = result["features"]

        # -------------------------------------------------------------------------
        # QUALITY FAIL
        # -------------------------------------------------------------------------
        if not features.quality_ok:

            return {
                "detected_action": None,
                "confidence": 0.0,
                "face_detected": True,
                "face_count": result["face_count"],
                "bbox": list(features.bbox),
                "processing_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
            }

        # -------------------------------------------------------------------------
        # ADD TO TEMPORAL BUFFER
        # -------------------------------------------------------------------------

        self._append_result(result)

        # -------------------------------------------------------------------------
        # DETECTION
        # -------------------------------------------------------------------------
        buffer_results = list(self.feature_buffer)

        head_action, head_confidence = self._detect_head_action()

        if head_action is not None:
            action = head_action
            confidence = head_confidence
        else:
            # -----------------------------------------------------------------
            # 2. CHỈ KIỂM TRA BLINK/SMILE KHI ĐẦU Ở TRẠNG THÁI ỔN ĐỊNH
            # -----------------------------------------------------------------
            blink_confidence = _detect_blink(buffer_results)
            if blink_confidence is not None:
                action = "blink"
                confidence = blink_confidence
            else:
                smile_confidence = _detect_smile(buffer_results)
                if smile_confidence is not None:
                    action = "smile"
                    confidence = smile_confidence
                else:
                    action = None
                    confidence = 0.0

        return {
            "detected_action": action,
            "confidence": round(float(confidence), 3),
            "face_detected": True,
            "face_count": result["face_count"],
            "bbox": list(features.bbox),
            "processing_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
        }
