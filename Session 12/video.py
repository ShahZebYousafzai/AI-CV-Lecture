import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components import containers as mp_containers
from ultralytics import YOLO

def run_pose_webcam(use_mediapipe=True):
    """
    Real-time pose estimation using either MediaPipe Tasks API or YOLOv8-pose.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)
    counter = CurlCounter()

    if use_mediapipe:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        landmarker = PoseLandmarker.create_from_options(options)
    else:
        yolo_model = YOLO('yolov8n-pose.pt')

    frame_idx = 0
    print("Webcam started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if use_mediapipe:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # VIDEO mode requires a monotonically increasing timestamp in ms
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) or frame_idx * 33
            result = landmarker.detect_for_video(mp_frame, timestamp_ms)

            if result.pose_landmarks:
                frame = draw_landmarks_on_image(frame, result)
                lms = result.pose_landmarks[0]
                shoulder = get_landmark_coords(lms, 'LEFT_SHOULDER', frame.shape)
                elbow    = get_landmark_coords(lms, 'LEFT_ELBOW',    frame.shape)
                wrist    = get_landmark_coords(lms, 'LEFT_WRIST',    frame.shape)
                angle = calculate_angle(shoulder, elbow, wrist)
                counter.update(angle)
                cv2.putText(frame, f'Angle: {angle:.0f}',
                            tuple(elbow), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Reps: {counter.counter}',
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(frame, f'Stage: {counter.stage}',
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)
        else:
            yolo_results = yolo_model(frame, verbose=False)
            frame = yolo_results[0].plot()

        cv2.imshow('Pose Estimation — Press Q to quit', frame)
        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if use_mediapipe:
        landmarker.close()
    print(f"Session ended. Total reps counted: {counter.counter}")


if __name__ == "__main__":
# Uncomment to run:
    run_pose_webcam(use_mediapipe=True)   # MediaPipe with curl counter
# run_pose_webcam(use_mediapipe=False)  # YOLOv8-pose

print("ℹ️ Webcam function defined. Uncomment the last line to run it.")