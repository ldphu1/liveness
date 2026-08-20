# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH = "face_landmarker_v2_with_blendshapes.task"

# Face
MAX_NUM_FACES = 2

# Blink
EAR_OPEN_THRESHOLD = 0.22
EAR_CLOSED_THRESHOLD = 0.19

# Smile
SMILE_RATIO_THRESHOLD = 0.38
SMILE_DELTA_THRESHOLD = 0.02

# Face quality
MIN_FACE_AREA_RATIO = 0.05

# Blurs
MIN_LAPLACIAN_VARIANCE = 20.0

# Left eye
LEFT_EYE = {
    "outer": 33,
    "upper_1": 160,
    "upper_2": 158,
    "inner": 133,
    "lower_1": 153,
    "lower_2": 144,
}

# Right eye
RIGHT_EYE = {
    "outer": 362,
    "upper_1": 385,
    "upper_2": 387,
    "inner": 263,
    "lower_1": 373,
    "lower_2": 380,
}

# Mouth
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Face width
FACE_LEFT = 234
FACE_RIGHT = 454

# =============================================================================
# TEMPORAL BUFFER
# =============================================================================

TEMPORAL_BUFFER_SIZE = 12

# Head movement
HEAD_ACTION_THRESHOLD = 8.0

# Sau khi detect head action thì giữ kết quả trong vài frame
HEAD_RESULT_HOLD_FRAMES = 5