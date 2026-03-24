"""
SkyNetra — main entry point
----------------------------
Runs the face detection, tracking, and re-identification pipeline.

Model paths are resolved relative to a configurable MODELS_DIR so the
project runs on any machine without editing source code.  Set the
environment variable SKYNETRA_MODELS to override the default location.

Usage
-----
    python main.py
    SKYNETRA_MODELS=/data/models python main.py
"""
# from flask import Flask, jsonify
# from flask_cors import CORS
# import threading

import os
import sys
import torch

from detectors.yolo_face_detector import FaceDetector
from trackers.byte_tracker_wrapper import create_tracker
from utils.identities.store import IdentityStore
from sampler.sampling import FrameSampler
from embedder.embedder import MobileFaceNet
from utils.quality_checker import FaceQualityChecker, TemporalConsistencyChecker
from utils.main_helpers import (
    setup_video_source,
    setup_identity_store,
    process_frame,
    cleanup,
    IdentityVoter,
)

# latest_state = {
#     "tracks": [],
#     "S": 0,
#     "gallery": [],
#     "frame": 0
# }

# app = Flask(__name__)
# CORS(app)

# @app.route("/state")
# def get_state():
#     return jsonify(latest_state)
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Bug fix: hardcoded absolute Windows path replaced with a portable resolution.
# Override with the SKYNETRA_MODELS environment variable if needed.
_DEFAULT_MODELS = os.path.join(os.path.dirname(__file__), "models")
MODELS_DIR = os.environ.get("SKYNETRA_MODELS", _DEFAULT_MODELS)

DETECTOR_MODEL  = os.path.join(MODELS_DIR, "yolov9t-face-lindevs")
EMBEDDER_MODEL  = os.path.join(MODELS_DIR, "mobilefacenet_fp16")
IDENTITIES_PATH = "identities"

EMB_DIM     = 256   # MobileFaceNet output dimension
MIN_SAMPLES = 10    # face crops required before re-ID fires (EmbeddingBuffer depth)


# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------

def _exists_any(base):
    for ext in [".engine", ".onnx", ".pt"]:
        if os.path.exists(base + ext):
            return True
    return False

def _check_models():
    missing = []

    if not _exists_any(os.path.join(MODELS_DIR, "yolov9t-face-lindevs")):
        missing.append("yolov9t-face-lindevs")

    if not _exists_any(os.path.join(MODELS_DIR, "mobilefacenet_fp16")):
        missing.append("mobilefacenet_fp16")

    if missing:
        for m in missing:
            print(f"[ERROR] Model not found: {m}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _check_models()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] Device: {device}")

    # ── Model initialisation ─────────────────────────────────────────────────
    sampler          = FrameSampler()
    detector         = FaceDetector(model_path=DETECTOR_MODEL, device=device)
    tracker          = create_tracker()
    quality_checker  = FaceQualityChecker()
    temporal_checker = TemporalConsistencyChecker()
    embedder         = MobileFaceNet(model_path=EMBEDDER_MODEL, device=device)

    # ── Video source ─────────────────────────────────────────────────────────
    cap, video_writer = setup_video_source()

    # ── Identity gallery ─────────────────────────────────────────────────────
    store = IdentityStore.from_path(IDENTITIES_PATH, device=device)
    id_names, gallery, hop = setup_identity_store(store, device, EMB_DIM)

    n_ids = len(id_names)
    if n_ids == 0:
        print("[i] Gallery is empty — tracker will run without re-ID.\n"
              "    Run add_info.py to enrol identities first.")
    else:
        print(f"[i] Gallery loaded: {n_ids} identit{'y' if n_ids == 1 else 'ies'}")

    # ── Per-frame state ───────────────────────────────────────────────────────
    identity_memory        = {}   # tid → EmbeddingBuffer
    identity_memory_pooled = {}   # tid → last pooled embedding (debug/future use)
    identity_voters        = {}   # tid → IdentityVoter  (stability: prevents fast flipping)
    track_info             = {}   # tid → display info dict

    print("\n[i] SkyNetra running  (press ESC to quit)\n")

    # Bug fix: wrap main loop in try/finally so cleanup() always runs even if
    # process_frame raises an uncaught exception (CUDA OOM, shape error, etc.).
    # Without this the VideoCapture and VideoWriter handles leak, the output
    # file ends up corrupted, and the webcam stays locked.
    try:
        # def run_api():
        #     app.run(host="0.0.0.0", port=5000)

        # threading.Thread(target=run_api, daemon=True).start()
        while True:
            if not process_frame(
                cap, sampler, detector, tracker,
                quality_checker, temporal_checker, embedder,
                identity_memory, identity_memory_pooled, identity_voters, track_info,
                id_names, gallery, hop,
                device, EMB_DIM, MIN_SAMPLES, video_writer,
            ):
                break
    except KeyboardInterrupt:
        print("\n[i] Interrupted by operator.")
    finally:
        cleanup(cap, video_writer)
        print("[i] Shutdown complete.")


if __name__ == "__main__":
    main()