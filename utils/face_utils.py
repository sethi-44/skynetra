import cv2
import torch
from ultralytics.engine.results import Boxes
import numpy as np
def crop_face(frame, box, size=160):
    x1, y1, x2, y2 = map(int, box)
    h, w, _ = frame.shape

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face = cv2.resize(face, (size, size))
    face = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
    return face

def empty_boxes(device):
    return Boxes(
        torch.empty((0, 6), device=device),
        orig_shape=(1, 1)
    )
# def safe_crop_np(frame_rgb, box):
#     x1, y1, x2, y2 = map(int, box)
#     h, w = frame_rgb.shape[:2]
#     x1, y1 = max(0, x1), max(0, y1)
#     x2, y2 = min(w, x2), min(h, y2)
#     face = frame_rgb[y1:y2, x1:x2]
#     return None if face.size == 0 else face
# utils/face_utils.py
def safe_crop_np(frame_rgb, box):
    x1, y1, x2, y2 = box
    h, w, _ = frame_rgb.shape

    # Clamp to valid slice bounds.
    # Lower bounds: 0.  Upper bounds: h/w (exclusive, matching slice semantics).
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # Reject tiny or inverted boxes
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None

    face = frame_rgb[y1:y2, x1:x2]

    # Cast to float32 before computing stats — uint8 std() can overflow
    # during intermediate variance calculation.
    face_f = face.astype(np.float32)

    if face_f.mean() < 5.0:
        return None
    if face_f.std() < 2.0:
        return None

    return face


def preprocess_face(face_rgb):
    """Preprocess face image for MobileFaceNet embedding."""
    face = cv2.resize(face_rgb, (112, 112))
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = np.transpose(face, (2, 0, 1))
    return face


def safe_crop(frame, box):
    """Safely crop face from frame with boundary checks."""
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face = frame[y1:y2, x1:x2]
    return None if face.size == 0 else face


def select_face_box(boxes):
    """Select the first detected face box."""
    xy = boxes.xyxy[0].cpu().numpy()
    return map(int, xy[:4])