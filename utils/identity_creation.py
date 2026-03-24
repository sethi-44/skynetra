"""
Identity Creation Workflow
--------------------------
Functions for capturing, refining, and enrolling face identity embeddings.
"""

import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from detectors.yolo_face_detector import FaceDetector
from utils.hopfield_layer import HopfieldLayer
from utils.identities.store import IdentityStore
from embedder.embedder import MobileFaceNet

# Bug fix: renamed from preprocess_face → preprocess_face_for_embedder in
# the fixed face_utils.py to avoid collision with MobileFaceNet's own method.
from utils.face_utils import preprocess_face_for_embedder, safe_crop, select_face_box

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MIN_FACE_EMBS  = 5      # minimum usable face crops required
PRINT_EVERY    = 15     # console progress interval (frames)
WEBCAM_INDEX   = 0      # default webcam device index
MIN_FACE_SIZE  = 40     # minimum face crop dimension in pixels
FRAME_SKIP     = 3      # process every Nth frame (reduces redundant inference)


# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------

def get_user_input() -> tuple:
    """
    Collect and validate all operator inputs.

    Returns
    -------
    (source, video_path, name, description, use_ref, ref_img, ref_image_path)
    """
    print(
        "\nSkyNetra Identity Builder\n"
        "-------------------------\n"
        "You are about to enrol a new identity.\n"
    )

    # Source selection
    while True:
        source = input("Source (video / webcam): ").strip().lower()
        if source in {"video", "webcam"}:
            break
        print("  Invalid — enter 'video' or 'webcam'.")

    if source == "video":
        while True:
            video_path = input("Video path: ").strip()
            if os.path.exists(video_path):
                break
            print(f"  File not found: {video_path!r}")
    else:
        video_path = WEBCAM_INDEX

    # Name
    while True:
        name = input("Person name: ").strip().title()
        if name:
            break
        print("  Name cannot be empty.")

    # Description
    raw_desc   = input("Description (or leave blank): ").strip()
    description = None if not raw_desc or raw_desc.lower() == "none" else raw_desc

    # Reference image (optional)
    raw_ref       = input("Reference image path (or blank to skip): ").strip()
    use_ref       = bool(raw_ref) and raw_ref.lower() != "none" and os.path.exists(raw_ref)
    ref_image_path = raw_ref if use_ref else None

    if raw_ref and raw_ref.lower() != "none" and not use_ref:
        print(f"  Reference image not found: {raw_ref!r} — proceeding without it.")

    ref_img = cv2.imread(raw_ref) if use_ref else None

    return source, video_path, name, description, use_ref, ref_img, ref_image_path


# ---------------------------------------------------------------------------
# Frame capture
# ---------------------------------------------------------------------------

def capture_embeddings(cap: cv2.VideoCapture, detector, source: str) -> list:
    """
    Scan a video/webcam and collect preprocessed face crops.

    Parameters
    ----------
    cap      : open VideoCapture (caller must not release before calling)
    detector : FaceDetector instance
    source   : "video" or "webcam"

    Returns
    -------
    List of preprocessed numpy face arrays ready for embed_faces().

    Raises
    ------
    RuntimeError if fewer than MIN_FACE_EMBS usable crops were found.

    Bug fix: cap is released inside this function — caller should not
    release it again. Documented explicitly.

    Bug fix: select_face_box can return None (empty detection); guard added
    before safe_crop to prevent TypeError on None unpack.

    Bug fix: FRAME_SKIP added — processes every Nth frame to avoid running
    900 near-identical YOLO inferences on a 30-second 30fps video.
    """
    video_embs: list = []
    frame_count      = 0

    print("[i] Collecting face embeddings...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames to reduce redundant inference
            if frame_count % FRAME_SKIP != 0:
                if source == "webcam":
                    cv2.imshow("Capture (ESC to stop)", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                continue

            if frame_count % (PRINT_EVERY * FRAME_SKIP) == 0:
                print(f"  > scanned {frame_count} frames, "
                      f"collected {len(video_embs)} faces")

            boxes = detector.detect(frame)
            if len(boxes) == 0:
                continue

            # Bug fix: select_face_box returns None for empty Boxes objects.
            face_box = select_face_box(boxes)
            if face_box is None:
                continue

            face = safe_crop(frame, face_box)
            if face is None:
                continue

            h, w = face.shape[:2]
            if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
                continue

            face_rgb = face[..., ::-1]

            # Bug fix: use renamed function preprocess_face_for_embedder
            face_np = preprocess_face_for_embedder(face_rgb)
            video_embs.append(face_np)

            if source == "webcam":
                cv2.imshow("Capture (ESC to stop)", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    except KeyboardInterrupt:
        print("\n[!] Interrupted by operator.\n")
    finally:
        # Always release here — caller must not release again
        cap.release()
        cv2.destroyAllWindows()

    if len(video_embs) < MIN_FACE_EMBS:
        raise RuntimeError(
            f"Only {len(video_embs)} usable face crops found "
            f"(need >= {MIN_FACE_EMBS}). "
            "Try better lighting, a closer shot, or a longer clip."
        )

    print(f"  > done — {len(video_embs)} usable crops from {frame_count} frames\n")
    return video_embs


# ---------------------------------------------------------------------------
# Embedding refinement
# ---------------------------------------------------------------------------

def refine_embeddings(
    video_embs: list,
    embedder,
    device: str,
    use_ref: bool = False,
    ref_img: np.ndarray = None,
) -> torch.Tensor:
    """
    Produce a single refined gallery embedding from raw face crops.

    Process
    -------
    1. Batch-embed all crops via MobileFaceNet.
    2. Normalise to unit sphere.
    3. Hopfield-refine toward the cluster attractor (beta = sqrt(D)).
    4. Optionally fuse with a reference image embedding (slerp approximation).
    5. Final unit normalisation.

    Returns
    -------
    pooled_emb : [D] float32 tensor (unit norm, ready for store.add_identity)

    Bug fix: beta=2.0 was hardcoded — far below sqrt(256)=16 — making the
    Hopfield layer essentially average all embeddings instead of converging
    to an attractor. Removed; HopfieldLayer now auto-sets beta=sqrt(D).

    Bug fix: returned pooled_emb as float16. store._prepare_embedding_for_storage
    handles the FP16 cast itself. Returning FP16 here caused .float() calls in
    handle_identity_store to silently work but broke the contract that all
    gallery embeddings are normalised before storage. Now returns FP32.

    Bug fix: reference fusion used (a + b) / 2 without re-normalising before
    addition. Average of two unit vectors is not unit-norm; the unnormalised
    intermediate distorted the direction before the final normalise. Now uses
    F.normalize on the sum — equivalent to spherical midpoint for similar vectors.
    """
    print(f"[i] Embedding {len(video_embs)} face crops...\n")

    tids    = list(range(len(video_embs)))   # dummy track IDs
    results = embedder.embed_faces(video_embs, tids)

    # Stack and normalise all embeddings
    embs = torch.stack([emb for _, emb in results]).float()
    embs = F.normalize(embs, dim=1)

    # Hopfield pooling — beta auto-set to sqrt(D) inside HopfieldLayer
    mean_init  = F.normalize(embs.mean(dim=0), dim=-1)
    hop        = HopfieldLayer(embs, device=device)   # beta=None → sqrt(D)
    pooled_emb = hop.refine(mean_init)

    # Optional: fuse with reference image embedding
    if use_ref and ref_img is not None:
        ref_rgb     = ref_img[..., ::-1]
        ref_face    = preprocess_face_for_embedder(ref_rgb)
        ref_results = embedder.embed_faces([ref_face], [0])
        ref_emb     = F.normalize(ref_results[0][1].float(), dim=-1)

        # Bug fix: normalise the sum (spherical midpoint) rather than
        # averaging unnormalised vectors.
        pooled_emb = F.normalize(pooled_emb + ref_emb, dim=-1)
        print("[i] Reference image fused.\n")

    # Final unit normalisation (defensive — refine should already return unit)
    pooled_emb = F.normalize(pooled_emb, dim=-1)

    print(f"[i] Gallery embedding ready  (norm={pooled_emb.norm().item():.4f})\n")

    # Return FP32 — store._prepare_embedding_for_storage casts to FP16 itself
    return pooled_emb.float()


# ---------------------------------------------------------------------------
# Identity store management
# ---------------------------------------------------------------------------

def handle_identity_store(
    store: IdentityStore,
    pooled_emb: torch.Tensor,
    name: str,
    description: str,
    ref_image_path: str,
    use_ref: bool,
) -> None:
    """
    Add a new identity or extend an existing one with the refined embedding.

    Bug fix: called store.is_duplicate(pooled_emb.float()) after pooled_emb
    was already FP16. Now pooled_emb is always FP32 (cast removed from
    refine_embeddings) so the explicit .float() here is a safe no-op.

    Bug fix: store.is_duplicate compares against ALL stored embeddings
    including dead (alive=False) ones in the original store.py. The fixed
    store.py's _similarity normalises the query, so this now works correctly.
    """
    if store.store:
        print("[i] Existing identities:")
        for i, info in enumerate(store.store):
            status = "" if info.alive else " [dead]"
            print(f"  [{i}] {info.name}{status}")
    else:
        print("[i] Store is empty — this will be the first identity.")

    while True:
        mode = input("\nAction (new / extend): ").strip().lower()
        if mode in {"new", "extend"}:
            break
        print("  Invalid — enter 'new' or 'extend'.")

    if mode == "new":
        if store.is_duplicate(pooled_emb):
            print(
                "[!] A very similar identity already exists in the store.\n"
                "    Use 'extend' to add more embeddings to it, or verify\n"
                "    you are not enrolling the same person twice."
            )
            return

        idx = store.add_identity(
            embedding=pooled_emb,
            name=name,
            description=description,
            image=ref_image_path if use_ref else None,
        )
        store.save("identities")
        print(f"\n[+] New identity enrolled: #{idx} — {name}")

    else:
        # Extend existing
        alive_indices = [i for i, info in enumerate(store.store) if info.alive]
        if not alive_indices:
            print("[!] No alive identities to extend.")
            return

        while True:
            try:
                idx = int(input("Identity index to extend: ").strip())
                if idx in alive_indices:
                    break
                print(f"  Invalid — choose from: {alive_indices}")
            except ValueError:
                print("  Please enter a number.")

        store.add_embedding(idx, pooled_emb)
        store.save("identities")
        print(f"\n[+] Embedding added to '{store.store[idx].name}' "
              f"({len(store.store[idx].emb_rows)} total rows)")