import cv2
import torch
from typing import List, Tuple, Optional
import numpy as np
from detectors.yolo_face_detector import FaceDetector
from utils.identities.store import IdentityStore
from utils.hopfield_layer import HopfieldLayer
from embedder.embedder import TRTMobileFaceNet
from trackers.matching import ious
from utils.face_utils import safe_crop_np
from utils.visualize import draw_tracks
from utils.embedding_ops import EmbeddingBuffer, identify_person, refine_identity, pool_embeddings
from sampler.sampling import FrameSampler
def setup_video_source() -> Tuple[cv2.VideoCapture, Optional[cv2.VideoWriter]]:
    save_output = False
    video_writer = None
    source = input("Source (video / webcam): ").strip().lower()
    assert source in {"video", "webcam"}

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

    if not cap.isOpened():
        raise RuntimeError("Could not open source")

    return cap, video_writer


def setup_identity_store(store: IdentityStore, device: str, emb_dim: int) -> Tuple[List[str], torch.Tensor, Optional[HopfieldLayer]]:
    if store.store:
        id_names = [info.name for info in store.store]
        gallery = store.embeddings.to(device, dtype=torch.float32)
        gallery = gallery / gallery.norm(dim=1, keepdim=True).clamp(min=1e-6)
        hop = HopfieldLayer(gallery, device=device)
    else:
        id_names = []
        gallery = torch.empty((0, emb_dim), device=device)
        hop = None

    return id_names, gallery, hop


def process_frame(
    cap: cv2.VideoCapture,
    sampler: FrameSampler,
    detector: FaceDetector,
    tracker,
    embedder: TRTMobileFaceNet,
    identity_memory: dict,
    identity_memory_pooled: dict,
    track_info: dict,
    id_names: List[str],
    gallery: torch.Tensor,
    hop: Optional[HopfieldLayer],
    device: str,
    emb_dim: int,
    min_samples: int,
    video_writer: Optional[cv2.VideoWriter]
) -> bool:
    ret, frame = cap.read()
    if not ret:
        return False

    h, w = frame.shape[:2]

    should_detect,reason = sampler.should_run_detector(tracker)
    print(f"Frame {sampler.frame_idx}: should_detect={should_detect} reason={reason}")
    if should_detect:
    # Detection
        dets, dets_conf = detector.boxes_to_yolox(detector.detect(frame))
        sampler.record_detection()
    else:        
        dets = torch.empty((0, 6), device=device)
        dets_conf = torch.empty((0,), device=device)

    # Tracking
    tracks = tracker.update(dets, img_info=(h, w), img_size=(h, w))
    # DEBUG
    # print(f"  tracked_stracks: {len(tracker.tracked_stracks)} | "
    #     f"lost: {len(tracker.lost_stracks)} | "
    #     f"activated: {sum(t.is_activated for t in tracker.tracked_stracks)} | "
    #     f"dets: {dets.shape[0]} | "
    #     f"ids: {[t.track_id for t in tracker.tracked_stracks if t.is_activated]}")

    # Attach detector confidence to tracks
    active_tracks = [t for t in tracks if t.is_activated]
    if len(active_tracks) > 0 and dets.shape[0] > 0:
        track_boxes = np.asarray([t.tlbr for t in active_tracks], dtype=np.float64)
        det_boxes = dets[:, :4].astype(np.float64)
        det_scores = dets_conf.astype(np.float64)
        iou_mat = ious(track_boxes, det_boxes)
        for i, t in enumerate(active_tracks):
            j = int(np.argmax(iou_mat[i]))
            if iou_mat[i, j] > 0.3:
                t.det_conf = float(det_scores[j])
            else:
                t.det_conf = None
    else:
        for t in active_tracks:
            t.det_conf = None

    frame_rgb = frame[..., ::-1]
    faces, tids = [], []

    for t in tracks:
        if not t.is_activated:
            continue
        x1, y1, x2, y2 = map(int, t.tlbr)
        face = safe_crop_np(frame_rgb, (x1, y1, x2, y2))
        if face is None:
            continue
        faces.append(embedder.preprocess_face(face))
        tids.append(t.track_id)

    # Embedding
    if faces:
        embed_result = embedder.embed_faces(faces, tids)
        for tid, emb in embed_result:
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                print(f"❌ NaN/Inf embedding from TRT | track {tid}")
                continue
            if emb.norm() < 1e-6:
                print(f"❌ Zero embedding from TRT | track {tid}")
                continue

            if tid not in identity_memory:
                identity_memory[tid] = EmbeddingBuffer(min_samples, emb_dim, device)
            buf = identity_memory[tid]
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                continue
            buf.add(emb)

            do_reid = buf.full()
            if do_reid and tid not in identity_memory_pooled:
                emb_buf = buf.get_all()
                pooled = pool_embeddings(emb_buf, device=device)
                refined, energy_before, energy, delta = refine_identity(pooled, hop)

                name, score = identify_person(
                    refined=refined,
                    gallery=gallery,
                    id_names=id_names,
                    delta=float(delta),
                    threshold=0.7,
                    delta_threshold=0.2,
                )

                identity_memory_pooled[tid] = pooled
                track_info[tid] = {
                    "name": name,
                    "id_conf": score,
                    "E_before": float(energy_before),
                    "E_after": float(energy),
                    "dE": float(delta),
                    "det_conf": getattr(t, "det_conf", None),
                    "track_conf": float(t.score),
                }

    # Visualization
    frame = draw_tracks(frame, tracks, track_info)
    if video_writer:
        video_writer.write(frame)

    cv2.imshow("Skynetra Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        return False

    return True


def cleanup(cap: cv2.VideoCapture, video_writer: Optional[cv2.VideoWriter]):
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
