"""
Identity Creation Workflow
--------------------------

Functions for creating and managing face identities.
"""

import os
import cv2
import torch
import numpy as np

from detectors.yolo_face_detector import FaceDetector
from utils.hopfield_layer import HopfieldLayer
from utils.identities.store import IdentityStore
from embedder.embedder import MobileFaceNet
from utils.face_utils import preprocess_face, safe_crop, select_face_box

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MIN_FACE_EMBS = 5
PRINT_EVERY = 15
WEBCAM_INDEX = 0
MIN_FACE_SIZE = 40


def get_user_input():
    """Get user inputs with validation."""
    print(
        "\nHello Operator.\n"
        "You are about to mint identity embeddings.\n"
        "These embeddings define reality.\n"
    )

    # Source selection
    while True:
        source = input("Source (video / webcam): ").strip().lower()
        if source in {"video", "webcam"}:
            break
        print("Invalid choice. Please enter 'video' or 'webcam'.")

    video_path = None
    if source == "video":
        while True:
            video_path = input("Video path: ").strip()
            if os.path.exists(video_path):
                break
            print(f"File not found: {video_path}")
    else:
        video_path = WEBCAM_INDEX

    # Name and description
    name = input("Person name: ").strip().title()
    while not name:
        name = input("Name cannot be empty. Person name: ").strip().title()

    description = input("Description or None: ").strip()
    description = None if description.lower() == "none" else description

    # Reference image
    ref_image_path = input("Reference image or None: ").strip()
    use_ref = ref_image_path.lower() != "none" and os.path.exists(ref_image_path)
    if ref_image_path.lower() != "none" and not use_ref:
        print(f"Reference image not found: {ref_image_path}. Proceeding without reference.")
    ref_img = cv2.imread(ref_image_path) if use_ref else None

    return source, video_path, name, description, use_ref, ref_img, ref_image_path


def capture_embeddings(cap, detector, source):
    """Capture face embeddings from video or webcam."""
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
        print("\n[!] Interrupted by operator.\n")

    cap.release()
    cv2.destroyAllWindows()

    if len(video_embs) < MIN_FACE_EMBS:
        raise RuntimeError(
            f"[!] Only {len(video_embs)} usable frames "
            f"(need >= {MIN_FACE_EMBS})"
        )

    return video_embs


def handle_identity_store(store, pooled_emb, name, description, ref_image_path, use_ref):
    """Handle adding or extending identity in the store."""
    print("\n[i] Existing identities:")
    for i, info in enumerate(store.store):
        print(f"  [{i}] {info.name}")

    while True:
        mode = input("\nAction (new / extend): ").strip().lower()
        if mode in {"new", "extend"}:
            break
        print("Invalid choice. Please enter 'new' or 'extend'.")

    if mode == "new":
        if store.is_duplicate(pooled_emb.float()):
            print("[!] Duplicate identity detected - aborting.")
            return
        else:
            idx = store.add_identity(
                embedding=pooled_emb,
                name=name,
                description=description,
                image=ref_image_path if use_ref else None,
            )
            store.save("identities")
            print(f"\n[+] New identity saved: #{idx} - {name}")

    else:
        while True:
            try:
                idx = int(input("Identity index to extend: ").strip())
                if 0 <= idx < len(store.store):
                    break
                print(f"Invalid index. Must be between 0 and {len(store.store)-1}.")
            except ValueError:
                print("Please enter a valid number.")

        store.add_embedding(idx, pooled_emb)
        store.save("identities")
        print(f"\n[+] Added prototype embedding to '{store.store[idx].name}'")


def refine_embeddings(video_embs, embedder, device, use_ref=False, ref_img=None):
    """Refine face embeddings using Hopfield layer and optional reference image."""
    print(f"\n[i] Refining {len(video_embs)} embeddings...\n")

    tids = list(range(len(video_embs)))  # dummy tids
    results = embedder.embed_faces(video_embs, tids)

    # Extract embeddings
    embs = torch.stack([emb for _, emb in results])

    # Convert to float if needed
    if embs.dtype != torch.float32:
        embs = embs.float()

    mean_emb = embs.mean(dim=0)
    pooled_emb = HopfieldLayer(
        embs, beta=2.0, device=device
    ).refine(mean_emb)

    if use_ref:
        ref_rgb = ref_img[..., ::-1]
        ref_face = preprocess_face(ref_rgb)
        ref_results = embedder.embed_faces([ref_face], [0])
        ref_emb = ref_results[0][1]
        pooled_emb = (pooled_emb + ref_emb) / 2
        print("[i] Reference image fused.\n")

    # Final normalize
    pooled_emb = pooled_emb / pooled_emb.norm()

    # Storage precision boundary
    pooled_emb = pooled_emb.half()

    return pooled_emb