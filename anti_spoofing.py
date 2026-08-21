from collections import deque
import cv2
import numpy as np
import onnxruntime as ort


class MiniFASNetONNX:
    def __init__(
            self,
            model_path: str = "models/anti_spoofing/best_model_quantized.onnx",
            scale: float = 2.7,
            input_size: tuple[int, int] = (128, 128),
            history_len: int = 7,
            spoof_threshold: float = 0.85,
    ):
        """
        Khởi tạo MiniFASNet INT8 ONNX với deque.
        """
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 2

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        self.scale = scale
        self.input_size = input_size
        self.spoof_threshold = spoof_threshold

        # Deque
        self.score_buffer = deque(maxlen=history_len)
    def reset(self):
        self.score_buffer.clear()

    def _crop_with_scale(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Crop mặt kèm tỉ lệ scale padding để lấy bối cảnh xung quanh."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]

        box_w = x2 - x1
        box_h = y2 - y1

        cx = x1 + box_w / 2
        cy = y1 + box_h / 2

        max_side = max(box_w, box_h) * self.scale

        new_x1 = int(cx - max_side / 2)
        new_y1 = int(cy - max_side / 2)
        new_x2 = int(cx + max_side / 2)
        new_y2 = int(cy + max_side / 2)

        # Padding nếu bbox vượt biên ảnh
        pad_x1 = max(0, -new_x1)
        pad_y1 = max(0, -new_y1)
        pad_x2 = max(0, new_x2 - w)
        pad_y2 = max(0, new_y2 - h)

        crop_x1 = max(0, new_x1)
        crop_y1 = max(0, new_y1)
        crop_x2 = min(w, new_x2)
        crop_y2 = min(h, new_y2)

        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

        if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                pad_y1, pad_y2, pad_x1, pad_x2,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )

        return cv2.resize(cropped, self.input_size)

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """BGR, NCHW, Float32"""
        img = crop.astype(np.float32)

        # NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, frame: np.ndarray, bbox: tuple[int, int, int, int], method: str = "mean") -> tuple[bool, float]:
        crop = self._crop_with_scale(frame, bbox)
        input_tensor = self._preprocess(crop)

        outputs = self.session.run(None, {self.input_name: input_tensor})[0]

        current_frame_real = 1.0 if outputs[0][0] > outputs[0][1] else 0.0

        self.score_buffer.append(current_frame_real)

        if method == "mean":
            # Yêu cầu ít nhất 5/7 frame là Real
            smoothed_score = float(np.mean(self.score_buffer))
            is_real = smoothed_score >= 0.65

        elif method == "median":
            # vote theo frame
            smoothed_score = float(np.median(self.score_buffer))
            is_real = smoothed_score == 1.0

        return is_real, smoothed_score


if __name__ == "__main__":
    img = cv2.imread("test/images.jpg")

    model = MiniFASNetONNX()

    is_real, smoothed_score = model.predict(
        img,
        (10, 10, 300, 300)
    )

    print(is_real, smoothed_score)