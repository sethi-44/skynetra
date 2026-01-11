import cv2
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

# ------------------------------------------------------
# Model / Utils Imports
# ------------------------------------------------------
from models.inception_resnet_v1 import InceptionResnetV1
from detectors.yolo_face_detector import FaceDetector
from detectors.async_gpu_detector import AsyncGPUDetector
from trackers.byte_tracker_wrapper import create_tracker
from utils.sampling import FrameSampler
from utils.face_utils import crop_face, empty_boxes
from utils.embedding_ops import (
    normalize_embeddings,
    pool_embeddings,
    refine_identity,
    identify_person,
)
from utils.visualize import draw_tracks

# --- Hopfield & Identity persistence ---
from utils.hopfield_layer import HopfieldLayer
from utils.identities.store import IdentityStore


def run():
    """
    Main Skynetra pipeline.

    Architecture:
    - YOLO face detector (async, sparse)
    - ByteTrack tracker (every frame)
    - Sampler controls detector invocation
    - FaceNet embeddings + Hopfield identity refinement
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------
    # Models
    # --------------------------------------------------
    base_detector = FaceDetector(
        "models/yolov8n-face-lindevs.pt", device
    )
    face_detector = AsyncGPUDetector(base_detector, device)

    tracker = create_tracker()
    sampler = FrameSampler()

    # Face embedding model
    resnet = InceptionResnetV1(
        pretrained=None, classify=False, device=device
    )
    resnet.logits = nn.Linear(512, 8631)
    state_dict = torch.load(
        "models/facenet_20180402_114759_vggface2.pth",
        map_location=device,
    )
    resnet.load_state_dict(state_dict)
    resnet.eval().to(device)

    # --------------------------------------------------
    # Identity Gallery (persistent)
    # --------------------------------------------------
    store = IdentityStore.from_path("identities", device=device)

    if len(store.store) > 0:
        id_names = [info.name for info in store.store]
        gallery = store.embeddings.to(device)
        gallery = gallery / gallery.norm(dim=1, keepdim=True)
        hop = HopfieldLayer(gallery, device=device)
    else:
        id_names = []
        gallery = torch.empty((0, 512), device=device)
        hop = None

    # --------------------------------------------------
    # Runtime Buffers
    # --------------------------------------------------
    identity_memory = {}         # tid → rolling embeddings
    identity_memory_pooled = {} # tid → final embedding
    identity_labels = {}         # tid → label string

    executor = ThreadPoolExecutor(max_workers=1)
    pending = None              # async detector future
    last_boxes = None           # last detector result
    detector_fresh = False      # true ONLY when detector just finished

    # --------------------------------------------------
    # Video IO
    # --------------------------------------------------
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    min_samples = 10

    cv2.namedWindow("Skynetra Tracking", cv2.WINDOW_NORMAL)

    tracks = []

    print("🚀 Skynetra pipeline running... press ESC to quit")

    # ==================================================
    # Main Loop
    # ==================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Video ended or camera unavailable.")
            break

        # ----------------------------------------------
        # 1. Sampler decides whether detector MAY run
        # ----------------------------------------------
        run_det, reason = sampler.should_run_detector(tracks)
        if run_det and pending is None:
            pending = executor.submit(
                face_detector.infer, frame.copy()
            )
            print(f"🔍 Detector triggered: {reason}")
        else:
            print(f"⏭️ Detector skipped: {reason}")

        # ----------------------------------------------
        # 2. Collect async detector result (if ready)
        # ----------------------------------------------
        if pending and pending.done():
            try:
                last_boxes = pending.result()
                detector_fresh = True
            except Exception as e:
                print("⚠️ Detector error:", e)
                last_boxes = None
            pending = None

        # ----------------------------------------------
        # 3. Tracker update (EVERY frame)
        # ----------------------------------------------
        boxes = (
            last_boxes if last_boxes is not None
            else empty_boxes(device)
        )

        if boxes.data.is_cuda:
            boxes.data = boxes.data.cpu()

        tracks = tracker.update(boxes)

        # Acknowledge detection exactly once
        if detector_fresh:
            sampler.record_detection(tracks)
            detector_fresh = False

        # ----------------------------------------------
        # 4. Face embedding & identity logic
        # ----------------------------------------------
        faces, tids = [], []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for t in tracks:
            if t[6] > 0:  # not confirmed
                continue

            face = crop_face(frame_rgb, t[0:4])
            if face is not None:
                faces.append(face)
                tids.append(t[4])

        if faces:
            faces = torch.stack(faces).to(device)
            with torch.no_grad():
                embs = normalize_embeddings(resnet(faces))

            for tid, emb in zip(tids, embs):
                buf = identity_memory.setdefault(tid, [])
                buf.append(emb.cpu())

                if len(buf) > min_samples:
                    buf.pop(0)

                # Pool + identify once per track
                if tid not in identity_memory_pooled and len(buf) >= min_samples:
                    pooled = pool_embeddings(buf, device=device)
                    refined, _, energy, delta = refine_identity(pooled, hop)
                    name, score = identify_person(
                        refined, gallery, id_names, delta
                    )

                    identity_memory_pooled[tid] = refined.cpu()
                    identity_labels[tid] = f"{name} ({score:.2f})"

                    sampler.update_id_confidence(tid, score)
                    sampler.update_id_energy(tid, energy)
                    sampler.update_id_embedding(tid, refined.cpu())

        # ----------------------------------------------
        # 5. Visualization
        # ----------------------------------------------
        frame = draw_tracks(frame, tracks, identity_labels)
        cv2.imshow("Skynetra Tracking", frame)

        # ----------------------------------------------
        # 6. Controls
        # ----------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()
    print("🧹 Clean shutdown, bye.")


if __name__ == "__main__":
    run()
