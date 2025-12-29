import cv2
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor

# --- Model / Utils Imports ---
from models.inception_resnet_v1 import InceptionResnetV1
from detectors.yolo_face_detector import FaceDetector
from detectors.async_gpu_detector import AsyncGPUDetector
from trackers.byte_tracker_wrapper import create_tracker
from utils.sampling import FrameSampler
from utils.global_motion_estimator import GlobalMotionEstimator
from utils.face_utils import crop_face, empty_boxes
from utils.embedding_ops import normalize_embeddings, pool_embeddings, refine_identity, identify_person
from utils.visualize import draw_tracks

# --- Hopfield ---
from utils.hopfield_layer import HopfieldLayer   


def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------
    # Models
    # ------------------------------------------------------
    base_detector = FaceDetector("models/yolov8n-face-lindevs.pt", device)
    face_detector = AsyncGPUDetector(base_detector, device)

    tracker = create_tracker()
    sampler = FrameSampler()
    gme = GlobalMotionEstimator()

    # --- Face embedding model ---
    resnet = InceptionResnetV1(pretrained=None, classify=False, device=device)
    resnet.logits = nn.Linear(512, 8631)
    state_dict = torch.load("models/facenet_20180402_114759_vggface2.pth", map_location=device)
    resnet.load_state_dict(state_dict)
    resnet.eval().to(device)

    # --- Known identities (gallery) ---
    id_names = ["Person_A", "Person_B"]
    gallery = torch.stack([torch.randn(512), torch.randn(512)]).to(device)
    gallery = gallery / gallery.norm(dim=1, keepdim=True)

    hop = HopfieldLayer(gallery, beta=2.0,device=device)

    # ------------------------------------------------------
    # Buffers
    # ------------------------------------------------------
    identity_memory = {}
    identity_memory_pooled = {}
    identity_labels = {}

    executor = ThreadPoolExecutor(max_workers=1)
    pending = None
    latest_boxes = None

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap = cv2.VideoCapture(r"C:\Users\harsh\Downloads\6518202-uhd_2160_3840_24fps.mp4")
    min_samples = 5

    print("🚀 Skynetra pipeline running... press ESC to quit")

    cv2.namedWindow("Skynetra Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Skynetra Tracking",
                          cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

    tracks = []

    # ======================================================
    # Main Loop
    # ======================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of video or cannot open file.")
            break

        # --- Global motion estimation (unused for now) ---
        # _ = gme.compute_global_motion(frame)

        # --- Retrieve async result if detector finished ---
        if pending and pending.done():
            try:
                latest_boxes = pending.result()
            except Exception as e:
                print("⚠️ async detector error:", e)
                latest_boxes = None
            pending = None

        # --- Decide detection ---
        run_det, reason = sampler.should_run_detector(tracks)
        if run_det and pending is None:
            pending = executor.submit(face_detector.infer, frame.copy())
            sampler.prev_active_ids = {t[4] for t in tracks if t[6] == 0} 
            print(f"🔍 Detector triggered: {reason}")
        else:
            print(f"⏭️ Detector skipped: {reason}")    

        # --- Tracker update ---
        boxes = latest_boxes if latest_boxes is not None else empty_boxes(device)
        if boxes.data.is_cuda:
            boxes.data = boxes.data.cpu()

        tracks = tracker.update(boxes)

        # --------------------------------------------------
        # Embedding + Hopfield memory + Identity label
        # --------------------------------------------------
        faces, tids = [], []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for t in tracks:
            if t[6] > 0:
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

                # Rolling buffer size
                if len(buf) > min_samples:
                    buf.pop(0)

                # Hopfield + label only when pooled first time
                if tid not in identity_memory_pooled and len(buf) >= min_samples:
                    pooled = pool_embeddings(buf, device=device)
                    refined, energy = refine_identity(pooled, hop)
                    best_name, best_score = identify_person(refined, gallery, id_names)

                    identity_memory_pooled[tid] = refined.cpu()
                    identity_labels[tid] = f"{best_name} ({best_score:.2f})"
                    # ---- ⬇ ADD THESE THREE LINES ⬇ ----
                    sampler.update_id_confidence(tid, best_score)
                    sampler.update_id_energy(tid, energy)
                    sampler.update_id_embedding(tid, refined.cpu())

        # --- Visualization ---
        frame = draw_tracks(frame, tracks, identity_labels)
        cv2.imshow("Skynetra Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("👋 Exiting loop (ESC pressed)")
            break

    # Shutdown
    cap.release()
    cv2.destroyAllWindows()
    print("🧹 Clean shutdown, bye.")


if __name__ == "__main__":
    run()
