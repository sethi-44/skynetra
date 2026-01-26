"""
Skynetra Identity Builder (TensorRT)
-----------------------------------

Create or extend a face identity using:
- video file
- webcam stream
- optional reference image

Embedding backend:
- MobileFaceNet (TensorRT FP16)

IMPORTANT:
- This MUST match the runtime embedding model & precision.
"""

import os
import cv2
import torch
import numpy as np

from detectors.yolo_face_detector import FaceDetector
from utils.hopfield_layer import HopfieldLayer
from utils.identities.store import IdentityStore
from utils.trt_mobilefacenet import TRTMobileFaceNet

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MIN_FACE_EMBS = 5
PRINT_EVERY = 15
WEBCAM_INDEX = 0
MIN_FACE_SIZE = 40
EMB_DIM = 256

# ------------------------------------------------------------
# Preprocessing (CPU, matches runtime)
# ------------------------------------------------------------
def preprocess_face(face_rgb):
    face = cv2.resize(face_rgb, (112, 112))
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = np.transpose(face, (2, 0, 1))
    return face


def safe_crop(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face = frame[y1:y2, x1:x2]
    return None if face.size == 0 else face


def select_face_box(boxes):
    xy = boxes.xyxy[0].cpu().numpy()
    return map(int, xy[:4])

# ------------------------------------------------------------
# TensorRT embedding
# ------------------------------------------------------------
@torch.no_grad()
def embed_faces_trt(trt_embedder, faces, device):
    """
    faces: list of (3,112,112) numpy arrays
    returns: FP32, L2-normalized embeddings
    """
    faces = torch.from_numpy(np.stack(faces)).to(
        device=device,
        dtype=torch.float16,
        non_blocking=True
    )

    embs = trt_embedder.infer(faces).float()
    embs = embs / embs.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return embs

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
# Device & Models
# ------------------------------------------------------------
assert torch.cuda.is_available(), "CUDA required"
device = "cuda"
print(f"\n[i] Device: {device}\n")

detector = FaceDetector("models/yolov8n-face-lindevs.pt", device)

trt_embedder = TRTMobileFaceNet(
    engine_path="models/mobilefacenet_fp16.trt",
    device=device
)

# Warm-up
with torch.no_grad():
    dummy = torch.zeros((4, 3, 112, 112), device=device, dtype=torch.float16)
    for _ in range(5):
        trt_embedder.infer(dummy)

print("[i] TensorRT MobileFaceNet ready\n")

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
        if face is None:
            continue

        h, w = face.shape[:2]
        if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
            continue

        face_rgb = face[..., ::-1]
        face_np = preprocess_face(face_rgb)
        video_embs.append(face_np)

        if source == "webcam":
            cv2.imshow("Capture (ESC to stop)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

except KeyboardInterrupt:
    print("\n[⚠] Interrupted by operator.\n")

cap.release()
cv2.destroyAllWindows()

if len(video_embs) < MIN_FACE_EMBS:
    raise RuntimeError(
        f"❌ Only {len(video_embs)} usable frames "
        f"(need ≥ {MIN_FACE_EMBS})"
    )

# ------------------------------------------------------------
# Embedding refinement (FP32)
# ------------------------------------------------------------
print(f"\n[i] Refining {len(video_embs)} embeddings...\n")

TRT_MAX_BATCH = 8

all_embs = []

for i in range(0, len(video_embs), TRT_MAX_BATCH):
    batch = video_embs[i:i + TRT_MAX_BATCH]
    batch_embs = embed_faces_trt(trt_embedder, batch, device)
    all_embs.append(batch_embs)

embs = torch.cat(all_embs, dim=0)


mean_emb = embs.mean(dim=0)
pooled_emb = HopfieldLayer(
    embs, beta=2.0, device=device
).refine(mean_emb)

if use_ref:
    ref_rgb = ref_img[..., ::-1]
    ref_face = preprocess_face(ref_rgb)
    ref_emb = embed_faces_trt(trt_embedder, [ref_face], device)[0]
    pooled_emb = (pooled_emb + ref_emb) / 2
    print("[i] Reference image fused.\n")

# ---- Final normalize (FP32) ----
pooled_emb = pooled_emb / pooled_emb.norm()

# ---- STORAGE PRECISION BOUNDARY ----
pooled_emb = pooled_emb.half()

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
    if store.is_duplicate(pooled_emb.float()):
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
