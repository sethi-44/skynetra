"""
Skynetra Identity Builder (MobileFaceNet ONNX)
---------------------------------------------

Create or extend a face identity using:
- video file
- webcam stream
- optional reference image

Embedding backend:
- MobileFaceNet (ONNX Runtime)

IMPORTANT:
- This MUST match the runtime embedding model.
"""

import os
import cv2
import torch
import numpy as np
import onnxruntime as ort

from detectors.yolo_face_detector import FaceDetector
from utils.hopfield_layer import HopfieldLayer
from utils.identities.store import IdentityStore


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MIN_FACE_EMBS = 5
PRINT_EVERY = 15
WEBCAM_INDEX = 0


# ------------------------------------------------------------
# ONNX face preprocessing
# ------------------------------------------------------------
def preprocess_face(face_rgb):
    """
    face_rgb: (H, W, 3) RGB uint8
    returns: (3, 112, 112) float32 normalized
    """
    face = cv2.resize(face_rgb, (112, 112))
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = np.transpose(face, (2, 0, 1))
    return face


def embed_face(face_rgb, ort_session, input_name, output_name):
    """
    Returns L2-normalized embedding (torch tensor, 128-D)
    """
    face = preprocess_face(face_rgb)
    face = np.expand_dims(face, axis=0)

    emb = ort_session.run(
        [output_name],
        {input_name: face}
    )[0][0]

    emb = emb / np.linalg.norm(emb)
    return torch.from_numpy(emb).float()


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def safe_crop(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


def select_face_box(boxes):
    xy = boxes.xyxy[0].cpu().numpy()
    return map(int, xy[:4])


# ------------------------------------------------------------
# Operator input
# ------------------------------------------------------------
print(
    "\nHello Operator.\n"
    "You are about to mint identity embeddings.\n"
    "These embeddings define reality.\n"
)

source = input("Source (video / webcam): ").strip().lower()
assert source in {"video", "webcam"}

if source == "video":
    video_path = input("Video path: ").strip()
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
else:
    video_path = WEBCAM_INDEX

name = input("Person name: ").strip().title()
description = input("Description or None: ").strip()
description = None if description.lower() == "none" else description

ref_image_path = input("Reference image or None: ").strip()
use_ref = ref_image_path.lower() != "none" and os.path.exists(ref_image_path)
ref_img = cv2.imread(ref_image_path) if use_ref else None


# ------------------------------------------------------------
# Device & ONNX Runtime
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[i] Device: {device}\n")

providers = []
if device == "cuda":
    providers.append("CUDAExecutionProvider")
providers.append("CPUExecutionProvider")

ort_session = ort.InferenceSession(
    "models/mobilefacenet.onnx",
    providers=providers
)

onnx_input = ort_session.get_inputs()[0].name
onnx_output = ort_session.get_outputs()[0].name

# Warm-up
dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)
_ = ort_session.run([onnx_output], {onnx_input: dummy})

print("[i] MobileFaceNet ONNX ready\n")

detector = FaceDetector("models/yolov8n-face-lindevs.pt", device)


# ------------------------------------------------------------
# Capture embeddings
# ------------------------------------------------------------
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Cannot open video/webcam")

video_embs = []
frame_count = 0

print("[i] Collecting face embeddings...\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % PRINT_EVERY == 0:
            print(f"  > scanned {frame_count} frames, collected {len(video_embs)} faces")

        boxes = detector.detect(frame)
        if len(boxes) == 0:
            continue

        face_box = select_face_box(boxes)
        face = safe_crop(frame, face_box)
        if face.size == 0:
            continue

        face_rgb = face[..., ::-1]
        emb = embed_face(face_rgb, ort_session, onnx_input, onnx_output)
        video_embs.append(emb)

        if source == "webcam":
            cv2.imshow("Capture (ESC to stop)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

except KeyboardInterrupt:
    print("\n[⚠] Interrupted by operator.\n")

cap.release()
cv2.destroyAllWindows()


# ------------------------------------------------------------
# Embedding refinement
# ------------------------------------------------------------
if len(video_embs) < MIN_FACE_EMBS:
    raise RuntimeError(
        f"❌ Only {len(video_embs)} usable frames "
        f"(need ≥ {MIN_FACE_EMBS})"
    )

print(f"\n[i] Refining {len(video_embs)} embeddings...\n")

mean_emb = torch.stack(video_embs).mean(dim=0)
pooled_emb = HopfieldLayer(
    video_embs, beta=2.0, device=device
).refine(mean_emb)

if use_ref:
    ref_rgb = ref_img[..., ::-1]
    ref_emb = embed_face(ref_rgb, ort_session, onnx_input, onnx_output)
    pooled_emb = (pooled_emb + ref_emb) / 2
    pooled_emb = pooled_emb / pooled_emb.norm()
    print("[i] Reference image fused.\n")


# ------------------------------------------------------------
# Identity store
# ------------------------------------------------------------
store = IdentityStore.from_path("identities", device=device)

print("\n[i] Existing identities:")
for i, info in enumerate(store.store):
    print(f"  [{i}] {info.name}")

mode = input("\nAction (new / extend): ").strip().lower()
assert mode in {"new", "extend"}

if mode == "new":
    if store.is_duplicate(pooled_emb):
        print("⚠ Duplicate identity detected — aborting.")
    else:
        idx = store.add_identity(
            embedding=pooled_emb,
            name=name,
            description=description,
            image=ref_image_path if use_ref else None,
        )
        store.save("identities")
        print(f"\n[✔] New identity saved: #{idx} — {name}")

else:
    idx = int(input("Identity index to extend: ").strip())
    assert 0 <= idx < len(store.store)

    store.extend_identity(idx, pooled_emb)
    store.save("identities")
    print(f"\n[✔] Added prototype embedding to '{store.store[idx].name}'")


# ------------------------------------------------------------
# Final stats
# ------------------------------------------------------------
print("\n[i] Database stats:")
print(store.stats())
print("\nMission complete.\n")
