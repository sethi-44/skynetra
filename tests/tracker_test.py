import cv2
import time

from detectors.yolo_face_detector import FaceDetectorTRT
from trackers.byte_tracker_wrapper import create_tracker

def main():
    source = input("Enter video path or press ENTER for webcam: ").strip()
    cap = cv2.VideoCapture(0 if source == "" else source)

    if not cap.isOpened():
        print("❌ Could not open source")
        return

    detector = FaceDetectorTRT(model_path=r"C:\Users\harsh\OneDrive\Desktop\skynetra\models\yolov9t-face-lindevs", device='cuda')
    tracker = create_tracker()  # YOLOX ByteTrack

    

    prev_time = time.time()

    print("🚀 Running YOLOv9 + YOLOX ByteTrack (ESC to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # -------------------------------
        # Detection
        # -------------------------------
        dets = detector.boxes_to_yolox(detector.detect(frame))

        # -------------------------------
        # Tracking (YOLOX contract)
        # -------------------------------
        tracks = tracker.update(
            dets,
            img_info=(h, w),
            img_size=(h, w)
        )

        # -------------------------------
        # Visualization
        # -------------------------------
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.tlbr)
            track_id = t.track_id

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        cv2.imshow("Face Tracking (YOLOv9 + YOLOX ByteTrack)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
