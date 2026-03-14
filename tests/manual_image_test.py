import cv2
import os
from pathlib import Path

from detectors.yolo_face_detector import FaceDetectorTRT

IMAGE_DIR = r"C:\Users\harsh\OneDrive\Desktop\skynetra\tests\images"


def main():
    detector = FaceDetectorTRT(model_path=r"C:\Users\harsh\OneDrive\Desktop\skynetra\models\yolov9t-face-lindevs", device='cuda')

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Loaded {len(image_files)} images")

    for idx, name in enumerate(image_files, 1):
        path = os.path.join(IMAGE_DIR, name)
        frame = cv2.imread(path)
        if frame is None:
            continue

        boxes = detector.detect(frame)

        # draw detections
        for box in boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"{idx}/{len(image_files)} | Faces: {len(boxes)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Manual Face Detector Test", frame)

        print(f"[{idx}/{len(image_files)}] {name}  →  faces: {len(boxes)}")
        print("Press:")
        print("  SPACE / ENTER → next image")
        print("  ESC           → quit")

        key = cv2.waitKey(0)

        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
