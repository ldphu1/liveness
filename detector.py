from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config



class FaceDetector:
    def __init__(self, model_path: str = config.MODEL_PATH, max_faces: int = config.MAX_NUM_FACES):
        """
        Khởi tạo MediaPipe FaceLandmarker detector.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=max_faces,
            running_mode=vision.RunningMode.VIDEO
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def detect_video(self, mp_image, timestamp_ms: int):
        return self._detector.detect_for_video(mp_image, timestamp_ms)