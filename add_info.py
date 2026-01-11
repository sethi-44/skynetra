"""
Skynetra Identity Builder
------------------------

Create or extend a face identity using:
- video file
- webcam stream
- optional reference image

Features:
- Hopfield refinement
- Duplicate prevention
- Prototype embedding accumulation
"""

import os
import cv2
import torch
import torch.nn as nn

from models.inception_resnet_v1 import InceptionResnetV1
from detectors.yolo_face_detector import FaceDetector
from utils.hopfield_layer import HopfieldLayer
from utils.identities.store import IdentityStore


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MIN_FACE_EMBS = 5          # minimum frames required
PRINT_EVERY = 15           # progress print interval
WEBCAM_INDEX = 0           # default webcam


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def embed_face(img, resnet, device):
    """
    Resize, normalize and embed a cropped face.
    Returns L2-normalized embedding.
    """
    img = cv2.resize(img, (160, 160))
    tensor = (
        torch.tensor(img)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device) / 255.0
    )
    with torch.no_grad():
        emb = resnet(tensor).squeeze(0)
        emb = emb / emb.norm()
    return emb


def safe_crop(frame, box):
    """
    Crop bounding box safely within image bounds.
    """
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return frame[y1:y2, x1:x2]


def select_face_box(boxes):
    """
    Select first detected face.
    Assumes YOLO Boxes object.
    """
    xy = boxes.xyxy[0].cpu().numpy()
    return map(int, xy[:4])


# ------------------------------------------------------------
# Input handling
# ------------------------------------------------------------
print(
    "\nHello Operator.\n"
    "You are about to mint identity embeddings.\n"
    "Mistakes will be remembered forever.\n"
)

source = input("Source (video / webcam): ").strip().lower()
assert source in {"video", "webcam"}, "Invalid source"

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
# Device & models
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[i] Device: {device}\n")

detector = FaceDetector("models/yolov8n-face-lindevs.pt", device)

resnet = InceptionResnetV1(pretrained=None, classify=False, device=device)
resnet.logits = nn.Linear(512, 8631)
state_dict = torch.load(
    "models/facenet_20180402_114759_vggface2.pth",
    map_location=device
)
resnet.load_state_dict(state_dict)
resnet.eval().to(device)


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

        emb = embed_face(face, resnet, device)
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
    ref_emb = embed_face(ref_img, resnet, device)
    pooled_emb = (pooled_emb + ref_emb) / 2
    pooled_emb = pooled_emb / pooled_emb.norm()
    print("[i] Reference image fused.\n")


# ------------------------------------------------------------
# Identity store operations
# ------------------------------------------------------------
store = IdentityStore.from_path("identities", device=device)

print("\n[i] Existing identities:")
for i, info in enumerate(store.store):
    print(f"  [{i}] {info.name}")

mode = input("\nAction (new / extend): ").strip().lower()
assert mode in {"new", "extend"}, "Invalid action"

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

else:  # extend existing
    idx = int(input("Identity index to extend: ").strip())
    assert 0 <= idx < len(store.store), "Invalid index"

    store.extend_identity(idx, pooled_emb)
    store.save("identities")
    print(f"\n[✔] Added prototype embedding to '{store.store[idx].name}'")


# ------------------------------------------------------------
# Final stats
# ------------------------------------------------------------
print("\n[i] Database stats:")
print(store.stats())
print("\nMission complete.\n")
