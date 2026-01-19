import time
import cv2
import torch
import numpy as np
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# ------------------------------------------------------
# Model / Utils Imports
# ------------------------------------------------------
from detectors.yolo_face_detector import FaceDetector
from detectors.async_gpu_detector import AsyncGPUDetector
from trackers.byte_tracker_wrapper import create_tracker
from utils.sampling import FrameSampler
from utils.face_utils import empty_boxes
from utils.embedding_ops import pool_embeddings, refine_identity,identify_person
from utils.visualize import draw_tracks
from utils.hopfield_layer import HopfieldLayer
from utils.identities.store import IdentityStore


# ------------------------------------------------------
# ONNX preprocessing (CPU, unavoidable)
# ------------------------------------------------------
def preprocess_face_onnx(face_rgb):
    face = cv2.resize(face_rgb, (112, 112))
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = np.transpose(face, (2, 0, 1))
    return face


def safe_crop_np(frame_rgb, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame_rgb.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face = frame_rgb[y1:y2, x1:x2]
    return None if face.size == 0 else face


# ------------------------------------------------------
# GPU rolling buffer
# ------------------------------------------------------
def update_gpu_buffer(buf, emb, max_len):
    if buf is None:
        return emb.unsqueeze(0)
    buf = torch.cat([buf, emb.unsqueeze(0)], dim=0)
    if buf.shape[0] > max_len:
        buf = buf[-max_len:]
    return buf


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def run():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame_count = 0
    start_total = time.perf_counter()
    t_det, t_track, t_emb = [], [], []

    # --------------------------------------------------
    # Detector / Tracker
    # --------------------------------------------------
    base_detector = FaceDetector("models/yolov8n-face-lindevs.pt", device)
    face_detector = AsyncGPUDetector(base_detector, device)

    tracker = create_tracker()
    sampler = FrameSampler()

    # --------------------------------------------------
    # ONNX Runtime GPU (NUMPY OUTPUT – SUPPORTED)
    # --------------------------------------------------
    ort_session = ort.InferenceSession(
        "models/mobilefacenet.onnx",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    onnx_input = ort_session.get_inputs()[0].name
    onnx_output = ort_session.get_outputs()[0].name

    # Warm-up
    dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)
    _ = ort_session.run([onnx_output], {onnx_input: dummy})

    # --------------------------------------------------
    # Identity Store (128-D)
    # --------------------------------------------------
    store = IdentityStore.from_path("identities", device=device)

    if store.store:
        id_names = [info.name for info in store.store]
        gallery = store.embeddings.to(device)
        gallery = gallery / gallery.norm(dim=1, keepdim=True)
        hop = HopfieldLayer(gallery, device=device)
    else:
        id_names, gallery, hop = [], torch.empty((0, 128), device=device), None

    identity_memory = {}
    identity_memory_pooled = {}
    identity_labels = {}

    # --------------------------------------------------
    # Async detector queue
    # --------------------------------------------------
    executor = ThreadPoolExecutor(max_workers=2)
    pending = deque(maxlen=2)
    last_boxes = None
    detector_fresh = False

    # --------------------------------------------------
    # Video
    # --------------------------------------------------
    source = input("Source (video / webcam): ").strip().lower()
    assert source in {"video", "webcam"}

    save_output = False
    video_writer = None

    if source == "video":
        video_path = input("Video path: ").strip()
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read video")

        save_output = input("Save output video? (y/n): ").strip().lower() == "y"
        if save_output:
            fps = int(input("Enter FPS for output video: "))
            h, w = frame.shape[:2]
            video_writer = cv2.VideoWriter(
                "skynetra_tracking.mp4",
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    min_samples = 10
    tracks = []

    print("🚀 Skynetra running — ESC to quit")

    # ==================================================
    # Main Loop
    # ==================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t0 = time.perf_counter()

        # Detector scheduling
        run_det, _ = sampler.should_run_detector(tracks)
        if run_det and len(pending) < 2:
            pending.append(executor.submit(face_detector.infer, frame.copy()))

        if pending and pending[0].done():
            try:
                last_boxes = pending.popleft().result()
                detector_fresh = True
            except Exception:
                last_boxes = None

        t1 = time.perf_counter()

        # Tracker (CPU)
        boxes = last_boxes if last_boxes is not None else empty_boxes(device)
        if boxes.data.is_cuda:
            boxes.data = boxes.data.cpu()

        tracks = tracker.update(boxes)

        if detector_fresh:
            sampler.record_detection(tracks)
            detector_fresh = False

        t2 = time.perf_counter()

        # --------------------------------------------------
        # Face embedding (ONNX GPU → NumPy → GPU ONCE)
        # --------------------------------------------------
        frame_rgb = frame[..., ::-1]

        for t in tracks:
            if t[6] > 0:
                continue

            x1, y1, x2, y2 = t[0:4]
            if (x2 - x1) < 40 or (y2 - y1) < 40:
                continue

            face = safe_crop_np(frame_rgb, (x1, y1, x2, y2))
            if face is None:
                continue

            face_np = preprocess_face_onnx(face)
            face_np = np.expand_dims(face_np, axis=0)

            # ---- ONNX inference (GPU, returns NumPy) ----
            emb_np = ort_session.run(
                [onnx_output],
                {onnx_input: face_np}
            )[0][0]

            # ---- normalize on CPU (cheap) ----
            emb_np /= np.linalg.norm(emb_np)

            # ---- CPU → GPU ONCE ----
            emb = torch.from_numpy(emb_np).to(device, non_blocking=True)

            tid = t[4]

            identity_memory[tid] = update_gpu_buffer(
                identity_memory.get(tid),
                emb,
                min_samples
            )

            if tid not in identity_memory_pooled:
                buf = identity_memory[tid]
                if buf.shape[0] >= min_samples:
                    pooled = pool_embeddings(buf,device=device)
                    refined, _, energy, delta= refine_identity(pooled, hop)

                    name, score = identify_person(
                        refined=refined,
                        gallery=gallery,
                        id_names=id_names,
                        delta=float(delta),   # or delta if you expose it
                        threshold=0.7,
                        delta_threshold=0.2,
                    )

                    identity_memory_pooled[tid] = refined
                    identity_labels[tid] = f"{name} ({score:.2f})"

                    sampler.update_id_confidence(tid, score)
                    sampler.update_id_energy(tid, float(energy))
                    sampler.update_id_embedding(tid, refined.detach().cpu())

        t3 = time.perf_counter()

        # --------------------------------------------------
        # Visualization
        # --------------------------------------------------
        frame = draw_tracks(frame, tracks, identity_labels)
        if save_output:
            video_writer.write(frame)
        # cv2.imshow("Skynetra Tracking", frame)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

        t_det.append((t1 - t0) * 1000)
        t_track.append((t2 - t1) * 1000)
        t_emb.append((t3 - t2) * 1000)

    # --------------------------------------------------
    # Cleanup + stats
    # --------------------------------------------------
    cap.release()
    if save_output:
        video_writer.release()
    # cv2.destroyAllWindows()    

    total = time.perf_counter() - start_total
    fps = frame_count / total

    print("\n📊 Benchmark")
    print(f"Frames: {frame_count}")
    print(f"Throughput FPS: {fps:.2f}")
    print(f"Detector avg: {sum(t_det)/len(t_det):.2f} ms")
    print(f"Tracker avg:  {sum(t_track)/len(t_track):.2f} ms")
    print(f"Embedding avg:{sum(t_emb)/len(t_emb):.2f} ms")
    print("🧹 Clean shutdown")


if __name__ == "__main__":
    run()
