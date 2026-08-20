import os
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


    analyzer = ActiveLivenessAnalyzer(_detector=FaceDetector()._detector)

    print(">>> Nhấn 'q' trên cửa sổ hiển thị để thoát.")
    print(">>> Nhấn 'r' để reset trạng thái liveness.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ---------------------------------------------------------------------
        # Mirror
        # ---------------------------------------------------------------------

        if MIRROR_VIEW:
            frame = cv2.flip(frame, 1)

        result = analyzer.process_frame(frame)

        face_detected = result.get("face_detected", False,)

        face_count = result.get( "face_count", 0)

        action = result.get( "detected_action", None)

        confidence = result.get( "confidence", 0.0)

        bbox = result.get("bbox", None)

        proc_time = result.get("processing_ms", 0.0)

        # ---------------------------------------------------------------------
        # FACE DETECTED
        # ---------------------------------------------------------------------
        if face_detected:
            # -------------------------------------------------------------
            # Bounding box
            # -------------------------------------------------------------

            if DRAW_BBOX and bbox:
                x1, y1, x2, y2 = bbox

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 128), 2)

            # -------------------------------------------------------------
            # Action
            # -------------------------------------------------------------
            if action:

                action_text = (f"Action: {action.upper()}")
                action_color = (0, 255, 0)

            else:
                action_text = ("Action: WAITING...")
                action_color = (0, 215, 255)

            cv2.putText(frame, f"{action_text} ({confidence:.0%})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, action_color, 2)

            # -------------------------------------------------------------
            # Face count + latency
            # -------------------------------------------------------------

            info_text = (f"Faces: {face_count} | " f"Latency: {proc_time:.1f}ms")

            cv2.putText(frame, info_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        # ---------------------------------------------------------------------
        # NO VALID FACE
        # ---------------------------------------------------------------------
        else:
            cv2.putText(frame, "Status: NO VALID FACE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            info_text = (f"Faces: {face_count} | " f"Latency: {proc_time:.1f}ms")

            cv2.putText(frame, info_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
        # ---------------------------------------------------------------------
        # Show temporal buffer status
        # ---------------------------------------------------------------------
        buffer_size = len(analyzer.feature_buffer)

        max_buffer_size = (analyzer.feature_buffer.maxlen)

        buffer_text = (f"Features: {buffer_size} / {max_buffer_size}")

        cv2.putText(frame, buffer_text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Active Liveness Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Reset liveness session
        if key == ord("r"):

            analyzer.reset()

            print(">>> Liveness state reset.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()