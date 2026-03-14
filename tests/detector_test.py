import cv2
import time
from detectors.yolo_face_detector import FaceDetectorTRT

detector = FaceDetectorTRT(model_path=r"C:\Users\harsh\OneDrive\Desktop\skynetra\models\yolov9t-face-lindevs", device='cuda')

cap = cv2.VideoCapture(0)  # webcam, change to video path if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()
    boxes = detector.detect(frame)
    fps = 1.0 / (time.time() - start)

    # draw boxes
    for box in boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLOv9 Face (TensorRT)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
