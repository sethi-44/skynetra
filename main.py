import time
import cv2
from scipy.datasets import face
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
from utils.trt_mobilefacenet import TRTMobileFaceNet
from utils.quality_checker import TemporalConsistencyChecker,FaceQualityChecker

TRT_MAX_BATCH = 8
MIN_SAMPLES = 10
# ------------------------------------------------------
# Preprocessing
# ------------------------------------------------------
def preprocess_face(face_rgb):
    face = cv2.resize(face_rgb, (112, 112))
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = np.transpose(face, (2, 0, 1))
    return face

def filter_detections_pre_track(boxes, min_size=40, min_conf=0.4):
    if boxes is None or len(boxes) == 0:
        return boxes

    b = boxes.data.cpu().numpy()
    keep = []

    for i, box in enumerate(b):
        x1, y1, x2, y2, conf = box[:5]

        if conf < min_conf:
            continue
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue

        keep.append(i)

    if not keep:
        return empty_boxes("cpu")

    boxes.data = boxes.data[keep]
    return boxes


def sanitize_boxes(boxes):
    """
    Remove boxes with invalid geometry that crash ByteTrack.
    """
    if boxes is None or len(boxes) == 0:
        return boxes

    b = boxes.data.cpu().numpy()
    keep = []

    for i, box in enumerate(b):
        x1, y1, x2, y2 = box[:4]
        w = x2 - x1
        h = y2 - y1

        if w <= 1 or h <= 1:
            continue
        if not np.isfinite([x1, y1, x2, y2]).all():
            continue

        keep.append(i)

    if not keep:
        return empty_boxes("cpu")

    boxes.data = boxes.data[keep]
    return boxes

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
# TensorRT embedding (BLOCKING, BATCHED)
# ------------------------------------------------------
@torch.no_grad()
def embed_faces_trt(trt_embedder, faces, tids, device):
    faces = torch.from_numpy(np.stack(faces)).to(
        device=device,
        dtype=torch.float16,
        non_blocking=True
    )

    outputs = []
    for i in range(0, faces.shape[0], TRT_MAX_BATCH):
        f = faces[i:i + TRT_MAX_BATCH]
        t = tids[i:i + TRT_MAX_BATCH]

        embs = trt_embedder.infer(f).float()

        # ✅ EXPLICIT L2 NORMALIZATION
        embs = embs / embs.norm(dim=1, keepdim=True).clamp(min=1e-6)

        outputs.extend(zip(t, embs))

    return outputs

def pool_embeddings_weighted(embs, weights, device):
    """
    Quality-weighted Hopfield pooling.
    embs: Tensor [N, D]
    weights: Tensor [N]
    """
    embs = embs.to(device)
    w = weights.to(device)

    w = w / w.sum().clamp(min=1e-6)

    mean_init = (embs * w.unsqueeze(1)).sum(dim=0)
    mean_init = mean_init / mean_init.norm().clamp(min=1e-6)

    hop = HopfieldLayer(embs, device=device)
    return hop.refine(mean_init)

# ------------------------------------------------------
# Fixed-size rolling buffer (preallocated)
# ------------------------------------------------------
class EmbeddingBuffer:
    def __init__(self, max_len, dim, device):
        self.emb = torch.empty((max_len, dim), device=device)
        self.w   = torch.empty((max_len,), device=device)
        self.max_len = max_len
        self.ptr = 0
        self.count = 0

    def add(self, emb, weight=1.0):
        self.emb[self.ptr] = emb
        self.w[self.ptr] = weight
        self.ptr = (self.ptr + 1) % self.max_len
        self.count = min(self.count + 1, self.max_len)

    def full(self):
        return self.count >= self.max_len

    def get_weighted(self):
        if self.count < self.max_len:
            embs = self.emb[:self.count]
            w = self.w[:self.count]
        else:
            idx = torch.arange(
                self.ptr, self.ptr + self.max_len, device=self.emb.device
            ) % self.max_len
            embs = self.emb[idx]
            w = self.w[idx]

        w = w / w.sum().clamp(min=1e-6)
        pooled = (embs * w.unsqueeze(1)).sum(dim=0)
        return pooled / pooled.norm().clamp(min=1e-6)
    def get_all(self):
        if self.count < self.max_len:
            return self.emb[:self.count], self.w[:self.count]
        idx = torch.arange(
            self.ptr, self.ptr + self.max_len, device=self.emb.device
        ) % self.max_len
        return self.emb[idx], self.w[idx]




# ------------------------------------------------------
# Main
# ------------------------------------------------------
def run():
    cv2.setUseOptimized(True)
    cv2.setNumThreads(0)

    assert torch.cuda.is_available(), "CUDA required"
    device = "cuda"
    frame_count = 0
    
    t_det, t_track, t_emb = [], [], []

    # --------------------------------------------------
    # Face quality
    # --------------------------------------------------
    quality_checker = FaceQualityChecker(
        min_face_size=80,   # overrides MIN_FACE_SIZE effectively
    )

    temporal_quality = TemporalConsistencyChecker(
        memory=10,
        threshold=0.65
    )

    # --------------------------------------------------
    # Detector / Tracker
    # --------------------------------------------------
    base_detector = FaceDetector("models/yolov8n-face-lindevs.pt", device)
    face_detector = AsyncGPUDetector(base_detector, device)

    tracker = create_tracker()
    sampler = FrameSampler()

    # --------------------------------------------------
    # TensorRT embedder
    # --------------------------------------------------
    trt_embedder = TRTMobileFaceNet(
        engine_path="models/mobilefacenet_fp16.trt",
        device=device
    )

    # Warmup
    with torch.no_grad():
        dummy = torch.zeros((4, 3, 112, 112), device=device, dtype=torch.float16)
        for _ in range(5):
            trt_embedder.infer(dummy)


    # --------------------------------------------------
    # Identity Store (128-D)
    # --------------------------------------------------
    store = IdentityStore.from_path("identities", device=device)

    if store.store:
        id_names = [info.name for info in store.store]
        gallery = store.embeddings.to(device)
        gallery = gallery / gallery.norm(dim=1, keepdim=True).clamp(min=1e-6)
        hop = HopfieldLayer(gallery, device=device)
        EMB_DIM= gallery.shape[1]
    else:
        id_names, gallery, hop = [], torch.empty((0, 256), device=device), None
        EMB_DIM = 256

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
    tracks = []

    print("🚀 Skynetra running — ESC to quit")
    start_total = time.perf_counter()

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
        # Ensure CPU for ByteTrack
        if last_boxes is not None and last_boxes.data.is_cuda:
            last_boxes = last_boxes.to("cpu", non_blocking=True)

        # Decide what to feed the tracker
        if detector_fresh and last_boxes is not None:
            boxes = sanitize_boxes(last_boxes)
            boxes = filter_detections_pre_track(boxes)
        else:
            boxes = last_boxes if last_boxes is not None else empty_boxes("cpu")

        tracks = tracker.update(boxes)



        active_ids = {t[4] for t in tracks if t[6] == 0}
        temporal_quality.cleanup(active_ids)

        if detector_fresh:
            sampler.record_detection(tracks)
            detector_fresh = False

        t2 = time.perf_counter()

        frame_rgb = frame[..., ::-1]
        faces, tids, qualities = [], [], []

        for t in tracks:
            if t[6] > 0:
                continue

            x1, y1, x2, y2 = map(int, t[0:4])


            face = safe_crop_np(frame_rgb, (x1, y1, x2, y2))
            if face is None:
                continue

            
            face_bgr = face[..., ::-1]
            # ---- QUALITY ASSESSMENT ----
            report = quality_checker.assess(
                face_bgr=face_bgr,
                box=(x1, y1, x2, y2),
                frame_shape=frame.shape
            )

            q = report["quality"]

            # Hard floor: never embed trash
            if q < 0.4:
                continue

            faces.append(preprocess_face(face))
            tids.append(t[4])
            qualities.append(q)

        # ---------------- Embedding ----------------
        if faces:
            embed_result = embed_faces_trt(trt_embedder, faces, tids, device) 
            for (tid,emb),q in zip(embed_result, qualities):
                # ---- TEMPORAL QUALITY CHECK ----
                quality_ok = temporal_quality.update(tid, q)
                if not quality_ok:
                    continue

                if tid not in identity_memory:
                    identity_memory[tid] = EmbeddingBuffer(
                        MIN_SAMPLES,EMB_DIM,device
                    )  
                buf = identity_memory[tid]
                buf.add(emb,weight=q)  

                do_reid=buf.full()
                if do_reid and tid not in identity_memory_pooled:
                    emb_buf ,weights= buf.get_all()

                    pooled = pool_embeddings_weighted(emb_buf,weights,device=device)
                    refined, _, energy, delta= refine_identity(pooled, hop)

                    name, score = identify_person(
                        refined=refined,
                        gallery=gallery,
                        id_names=id_names,
                        delta=float(delta),  
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
        tracks = [t for t in tracks if t[6] == 0]

        frame = draw_tracks(frame, tracks, identity_labels)
        if save_output:
            video_writer.write(frame)
        cv2.imshow("Skynetra Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

        t_det.append((t1 - t0) * 1000)
        t_track.append((t2 - t1) * 1000)
        t_emb.append((t3 - t2) * 1000)

    # --------------------------------------------------
    # Cleanup + stats
    # --------------------------------------------------
    cap.release()
    if save_output:
        video_writer.release()
    cv2.destroyAllWindows()    

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
