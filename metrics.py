import cv2
import numpy as np
from config import *

# =============================================================================
# Euclid distance and normalize landmark point
# =============================================================================

def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _get_point(landmarks, index: int, width: int, height: int) -> np.ndarray:
    lm = landmarks[index]

    return np.array([lm.x * width,lm.y * height,],dtype=np.float32)


# =============================================================================
# Eye aspect ratio `
# =============================================================================

def _calculate_ear(landmarks, eye_indices: dict, width: int, height: int) -> float:

    outer = _get_point(
        landmarks,
        eye_indices["outer"],
        width,
        height,
    )

    upper_1 = _get_point(
        landmarks,
        eye_indices["upper_1"],
        width,
        height,
    )

    upper_2 = _get_point(
        landmarks,
        eye_indices["upper_2"],
        width,
        height,
    )

    inner = _get_point(
        landmarks,
        eye_indices["inner"],
        width,
        height,
    )

    lower_1 = _get_point(
        landmarks,
        eye_indices["lower_1"],
        width,
        height,
    )

    lower_2 = _get_point(
        landmarks,
        eye_indices["lower_2"],
        width,
        height,
    )

    horizontal = _distance(outer, inner)

    if horizontal <= 1e-6:
        return 0.0

    vertical_1 = _distance(upper_1, lower_2)
    vertical_2 = _distance(upper_2, lower_1)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# =============================================================================
# SMILE
# =============================================================================

def _calculate_smile_ratio(landmarks, width: int, height: int) -> float:
    mouth_left = _get_point(
        landmarks,
        MOUTH_LEFT,
        width,
        height,
    )

    mouth_right = _get_point(
        landmarks,
        MOUTH_RIGHT,
        width,
        height,
    )

    face_left = _get_point(
        landmarks,
        FACE_LEFT,
        width,
        height,
    )

    face_right = _get_point(
        landmarks,
        FACE_RIGHT,
        width,
        height,
    )

    mouth_width = _distance(
        mouth_left,
        mouth_right,
    )

    face_width = _distance(
        face_left,
        face_right,
    )

    if face_width <= 1e-6:
        return 0.0

    return mouth_width / face_width


# =============================================================================
# HEAD POSE
# =============================================================================
def _calculate_head_pose(matrix) -> tuple[float, float]:
    matrix = np.asarray(matrix, dtype=np.float32)
    rotation = matrix[:3, :3]

    yaw = np.degrees(np.arctan2(rotation[0, 2], rotation[2, 2]))

    pitch = np.degrees(np.arctan2(-rotation[1, 2], np.sqrt(rotation[0, 2]**2 + rotation[2, 2]**2)))

    return float(yaw), float(pitch)

# =============================================================================
# FACE BBOX
# =============================================================================
def _calculate_bbox(landmarks, width: int, height: int) -> tuple[int, int, int, int]:

    xs = np.array([lm.x for lm in landmarks], dtype=np.float32)

    ys = np.array([lm.y for lm in landmarks], dtype=np.float32)

    x1 = int(np.clip(xs.min() * width, 0, width,))

    y1 = int(np.clip(ys.min() * height,0, height,))

    x2 = int(np.clip(xs.max() * width,0, width,))

    y2 = int(np.clip(ys.max() * height,0, height,))

    return x1, y1, x2, y2


# =============================================================================
# FACE AREA
# =============================================================================
def _calculate_face_area_ratio(bbox, width: int, height: int) -> float:

    x1, y1, x2, y2 = bbox

    face_width = max(0, x2 - x1)
    face_height = max(0, y2 - y1)

    face_area = face_width * face_height

    image_area = width * height

    if image_area <= 0:
        return 0.0

    return face_area / image_area


# =============================================================================
# BLUR(dùng để loại bỏ frame mờ)
# =============================================================================

def _calculate_blur(image: np.ndarray) -> float:

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# =============================================================================
# SELECT FACE
# =============================================================================
# def _select_largest_face(face_landmarks_list):
#
#     best_index = -1
#     best_area = -1.0
#
#     for index, landmarks in enumerate(face_landmarks_list):
#
#         xs = [lm.x for lm in landmarks]
#         ys = [lm.y for lm in landmarks]
#
#         area = (max(xs) - min(xs)) * (max(ys) - min(ys))
#
#         if area > best_area:
#             best_area = area
#             best_index = index
#
#     return best_index


# =============================================================================
# ANALYZE SINGLE FRAME
# =============================================================================
