"""
Skynetra Identity Builder
-------------------------

Create or extend a face identity using:
- video file
- webcam stream
- optional reference image

Embedding backend:
- MobileFaceNet (TensorRT FP16 or ONNX CPU)
"""

import torch
import cv2

from detectors.yolo_face_detector import FaceDetector
from utils.identities.store import IdentityStore
from embedder.embedder import MobileFaceNet
from utils.identity_creation import get_user_input, capture_embeddings, handle_identity_store, refine_embeddings

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MIN_FACE_EMBS = 5

def main():
    """Main function for identity creation."""
    # Get user inputs
    source, video_path, name, description, use_ref, ref_img, ref_image_path = get_user_input()

    # Setup device and models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[i] Device: {device}\n")

    detector = FaceDetector("models/yolov9t-face-lindevs", device)

    embedder = MobileFaceNet(
        model_path="models/mobilefacenet_fp16",
        device=device
    )

    print(f"[i] {'TensorRT' if device == 'cuda' else 'ONNX'} MobileFaceNet ready\n")

    # Capture setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video/webcam")

    # Capture embeddings
    video_embs = capture_embeddings(cap, detector, source)

    # Embedding refinement
    pooled_emb = refine_embeddings(video_embs, embedder, device, use_ref, ref_img)

    # Handle identity store
    store = IdentityStore.from_path("identities", device=device)
    handle_identity_store(store, pooled_emb, name, description, ref_image_path, use_ref)

    # Final stats
    print("\n[i] Database stats:")
    print(store.stats())
    print("\nMission complete.\n")

if __name__ == "__main__":
    main()
