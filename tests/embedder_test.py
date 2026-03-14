import cv2
import os
import torch

from detectors.yolo_face_detector import FaceDetector
from utils.trt_mobilefacenet import TRTMobileFaceNet
from utils.face_utils import safe_crop_np

IMAGE_DIR = r"C:\Users\harsh\OneDrive\Desktop\skynetra\tests\images"

DETECTOR_MODEL = r"C:\Users\harsh\OneDrive\Desktop\skynetra\models\yolov9t-face-lindevs"
EMBEDDER_MODEL = r"C:\Users\harsh\OneDrive\Desktop\skynetra\models\mobilefacenet_fp16"


def main():

    detector = FaceDetector(model_path=DETECTOR_MODEL, device='cpu')
    embedder = TRTMobileFaceNet(model_path=EMBEDDER_MODEL, device='cpu')

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

        frame_rgb = frame[..., ::-1]

        faces = []

        # Draw detections
        for box in boxes.xyxy.cpu().numpy().astype(int):

            x1, y1, x2, y2 = box

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face = safe_crop_np(frame_rgb, (x1, y1, x2, y2))

            if face is None:
                continue

            faces.append(face)

        # -------------------
        # Embedder test
        # -------------------

        if faces:

            processed = [embedder.preprocess_face(f) for f in faces]

            results = embedder.embed_faces(
                processed,
                list(range(len(processed)))
            )

            y_offset = 80

            for i, (_, emb) in enumerate(results):

                emb = emb.float()

                has_nan = torch.isnan(emb).any().item() or torch.isinf(emb).any().item()

                norm = emb.norm().item()

                print(
                    f"[{idx}] Face {i}: "
                    f"NaN={has_nan} | "
                    f"norm={norm:.3f}"
                )

                color = (0, 255, 0)
                label = f"Face {i} OK"

                if has_nan or norm < 0.1 or norm > 10.0:
                    color = (0, 0, 255)
                    label = f"Face {i} BAD EMB"

                cv2.putText(
                    frame,
                    label,
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                y_offset += 30

            # -------------------
            # Determinism test
            # -------------------

            face_np = faces[0]
            face_proc = embedder.preprocess_face(face_np)

            embs = []

            for _ in range(3):

                emb = embedder.embed_faces([face_proc], [0])[0][1]

                emb = emb.float().clone()
                emb = emb / emb.norm().clamp(min=1e-6)

                embs.append(emb)

            for i in range(1, len(embs)):

                cos = torch.dot(embs[0], embs[i]).item()

                print(f"Determinism cosine {i}: {cos:.4f}")

                if cos < 0.95:

                    cv2.putText(
                        frame,
                        "NON-DETERMINISTIC EMBEDDING",
                        (20, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

        # Overlay info
        cv2.putText(
            frame,
            f"{idx}/{len(image_files)} | Faces: {len(boxes)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Detector + Embedder Test", frame)

        print(f"[{idx}/{len(image_files)}] {name} → faces: {len(boxes)}")

        key = cv2.waitKey(0)

        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()