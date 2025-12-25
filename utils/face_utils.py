import cv2
import torch

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
