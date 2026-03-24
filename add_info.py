"""
SkyNetra Identity Builder
-------------------------
Create or extend a face identity in the gallery using:
  - a video file
  - a live webcam stream
  - an optional reference image for embedding fusion

Embedding backend: MobileFaceNet (TensorRT FP16 on CUDA, ONNX on CPU)

Usage
-----
    python add_info.py
"""

import cv2
import torch

from detectors.yolo_face_detector import FaceDetector
from utils.identities.store import IdentityStore
from embedder.embedder import MobileFaceNet
from utils.identity_creation import (
    get_user_input,
    capture_embeddings,
    handle_identity_store,
    refine_embeddings,
)

# Bug fix: MIN_FACE_EMBS was duplicated here and in identity_creation.py.
# The value here was never actually used — capture_embeddings reads from
# identity_creation.py's constant. Removed the duplicate entirely.


def main() -> None:
    """End-to-end identity enrolment pipeline."""

    # ── 1. Operator input ────────────────────────────────────────────────────
    source, video_path, name, description, use_ref, ref_img, ref_image_path = (
        get_user_input()
    )

    # ── 2. Hardware and model setup ──────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[i] Device: {device}")

    detector = FaceDetector("models/yolov9t-face-lindevs", device)
    embedder = MobileFaceNet(
        model_path="models/mobilefacenet_fp16",
        device=device,
    )
    backend = "TensorRT" if device == "cuda" else "ONNX"
    print(f"[i] {backend} MobileFaceNet loaded\n")

    # ── 3. Open capture source ───────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video/webcam: {video_path!r}\n"
            "Check the path or webcam index."
        )

    # ── 4. Collect face crops ─────────────────────────────────────────────────
    # capture_embeddings releases cap internally — do not release again here.
    video_embs = capture_embeddings(cap, detector, source)

    # ── 5. Produce refined gallery embedding ──────────────────────────────────
    pooled_emb = refine_embeddings(
        video_embs, embedder, device,
        use_ref=use_ref, ref_img=ref_img,
    )

    # ── 6. Enrol into the identity store ──────────────────────────────────────
    store = IdentityStore.from_path("identities", device=device)
    handle_identity_store(
        store, pooled_emb, name, description, ref_image_path, use_ref
    )

    # ── 7. Summary ────────────────────────────────────────────────────────────
    stats = store.stats()
    print(
        f"\n[i] Gallery summary:\n"
        f"    Identities : {stats['alive']} alive, {stats['dead']} dead\n"
        f"    Embeddings : {stats['total_embeddings']} total\n"
        f"    Avg / ID   : {stats['avg_emb_per_id']:.1f}\n"
        "\nMission complete.\n"
    )


if __name__ == "__main__":
    main()