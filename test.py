import os
import time
import warnings

warnings.filterwarnings("ignore")

try:
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
except Exception:
    pass

import cv2

from analyzer import ActiveLivenessAnalyzer
from detector import FaceDetector

# =============================================================================
# CONFIGURATION
# =============================================================================

CAMERA_INDEX = 0
DRAW_BBOX = True
MIRROR_VIEW = False


# =============================================================================
# MAIN
# =============================================================================

def run_inference():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam.")
        return

    # Khởi tạo Analyzer
    analyzer = ActiveLivenessAnalyzer(_detector=FaceDetector()._detector)

    print(">>> Nhấn 'q' trên cửa sổ hiển thị để thoát.")
    print(">>> Nhấn 'r' để reset trạng thái liveness.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(time.time() * 1000)

        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)

        result = analyzer.process_frame(frame, timestamp_ms)

        face_detected = result.get("face_detected", False)
        face_count = result.get("face_count", 0)
        action = result.get("detected_action", None)
        confidence = result.get("confidence", 0.0)
        bbox = result.get("bbox", None)
        proc_time = result.get("processing_ms", 0.0)

        # Thông tin từ module Passive Anti-Spoofing
        is_real = result.get("is_real", True)
        liveness_score = result.get("liveness_score", 1.0)
        status = result.get("status", "SUCCESS")

        # ---------------------------------------------------------------------
        # FACE DETECTED
        # ---------------------------------------------------------------------
        if face_detected and bbox:
            x1, y1, x2, y2 = bbox

            # Kiểm tra trạng thái Giả mạo
            if not is_real or status == "SPOOF_DETECTED":
                box_color = (0, 0, 255)
                spoof_text = f"SPOOF DETECTED"
                cv2.putText(frame, spoof_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                box_color = (0, 255, 128)

                # Hiển thị Action
                if action:
                    action_text = f"Action: {action.upper()} ({confidence:.0%})"
                    action_color = (0, 255, 0)
                else:
                    action_text = "Action: WAITING..."
                    action_color = (0, 215, 255)

                cv2.putText(frame, action_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2)

                liveness_text = f"Liveness: REAL ({liveness_score:.1%})"
                cv2.putText(frame, liveness_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 128), 2)

            if DRAW_BBOX:
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # Tổng latency
            info_text = f"Faces: {face_count} | Latency: {proc_time:.1f}ms"
            cv2.putText(frame, info_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # ---------------------------------------------------------------------
        # NO VALID FACE / MULTIPLE FACES
        # ---------------------------------------------------------------------
        else:
            if face_count > 1:
                warn_text = "Status: MULTIPLE FACES DETECTED"
            else:
                warn_text = "Status: NO VALID FACE"

            cv2.putText(frame, warn_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            info_text = f"Faces: {face_count} | Latency: {proc_time:.1f}ms"
            cv2.putText(frame, info_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # ---------------------------------------------------------------------
        # TEMPORAL BUFFER STATUS
        # ---------------------------------------------------------------------
        buffer_size = len(analyzer.feature_buffer)
        max_buffer_size = analyzer.feature_buffer.maxlen
        buffer_text = f"Buffer: {buffer_size} / {max_buffer_size}"
        cv2.putText(frame, buffer_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Active & Passive Liveness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # Reset session liveness
        if key == ord("r"):
            analyzer.reset()
            print(">>> Liveness state reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()