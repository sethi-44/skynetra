import cv2
import torch
from typing import List, Tuple, Optional
from collections import Counter
import numpy as np
from detectors.yolo_face_detector import FaceDetector
from utils.identities.store import IdentityStore
from utils.hopfield_layer import HopfieldLayer
from embedder.embedder import MobileFaceNet
from trackers.matching import ious
from utils.face_utils import safe_crop_np,preprocess_face_for_embedder
from utils.visualize import draw_tracks
from utils.embedding_ops import EmbeddingBuffer, identify_person, refine_identity, pool_embeddings,identify_person_cosine
from utils.quality_checker import FaceQualityChecker, TemporalConsistencyChecker
from sampler.sampling import FrameSampler

import numpy as np

sent_faces = {} 

def to_numpy(x, dtype=np.float64):
    """
    Converts torch.Tensor or numpy array to numpy array safely.
    """
    if x is None:
        return None

    # If it's already numpy
    if isinstance(x, np.ndarray):
        return x.astype(dtype)

    # If it's a torch tensor
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(dtype)
    except ImportError:
        pass

    # Fallback (just in case something weird comes in)
    return np.array(x, dtype=dtype)
# ---------------------------------------------------------------------------
# Identity voter — the primary fix for fast identity flipping
# ---------------------------------------------------------------------------

class IdentityVoter:
    """
    Accumulates re-ID results across multiple buffer windows and only
    commits a name change when that name wins a supermajority of recent
    votes weighted by calibrated confidence.

    Why flipping happens without this
    ----------------------------------
    The EmbeddingBuffer is circular with min_samples=10. Once full, every
    new embedding shifts the window by one frame and triggers a new re-ID
    that immediately overwrites track_info[tid].  One outlier embedding
    in the window is enough to flip the displayed name for 10 frames.

    How the voter fixes it
    ----------------------
    Each re-ID result (name, calibrated_conf) is appended to a rolling
    history of length `history_len`. The committed name only changes when
    a candidate clears both:
        - vote_share  >= vote_threshold   (fraction of recent windows)
        - mean_conf   >= conf_threshold   (average calibrated confidence)

    Once locked, a challenger needs lock_threshold share to unseat the
    current name — raising the bar further against momentary noise.

    Parameters
    ----------
    history_len     : rolling window of re-ID results to consider  (default 8)
    vote_threshold  : fraction of windows that must agree to lock in  (0.60)
    conf_threshold  : mean calibrated confidence required to commit    (0.50)
    lock_threshold  : share required to unseat a locked identity        (0.70)
    """

    def __init__(
        self,
        history_len:    int   = 8,
        vote_threshold: float = 0.60,
        conf_threshold: float = 0.50,
        lock_threshold: float = 0.70,
    ):
        self.history_len    = history_len
        self.vote_threshold = vote_threshold
        self.conf_threshold = conf_threshold
        self.lock_threshold = lock_threshold

        self._history: list = []   # [(name, calibrated_conf), ...]
        self.committed:  str   = ""
        self.locked:     bool  = False
        self.lock_conf:  float = 0.0

    def update(self, name: str, calibrated_conf: float) -> str:
        """
        Record a new window result and return the stable committed name.

        ABSTAIN/Unknown votes accumulate as uncertainty evidence but cannot
        flip a locked confident identity on their own — only a real-name
        supermajority can unseat a locked result.
        """
        self._history.append((name, float(calibrated_conf)))
        if len(self._history) > self.history_len:
            self._history.pop(0)

        total = len(self._history)
        vote_counts: Counter = Counter()
        conf_buckets: dict   = {}

        for n, c in self._history:
            vote_counts[n] += 1
            conf_buckets.setdefault(n, []).append(c)

        # Only real identity names compete; ABSTAIN/Unknown are uncertainty signals
        real = [n for n in vote_counts if n not in ("ABSTAIN", "Unknown", "")]

        if not real:
            abstain_share = vote_counts.get("ABSTAIN", 0) / total
            if self.locked and abstain_share >= self.lock_threshold:
                self.committed = "ABSTAIN"
                self.locked    = False
                self.lock_conf = 0.0
            return self.committed or "ABSTAIN"

        best      = max(real, key=lambda n: vote_counts[n])
        share     = vote_counts[best] / total
        mean_conf = float(np.mean(conf_buckets[best]))

        if not self.locked:
            if share >= self.vote_threshold and mean_conf >= self.conf_threshold:
                self.committed = best
                self.locked    = True
                self.lock_conf = mean_conf
        else:
            if best == self.committed:
                # Smooth the confidence of the locked identity
                self.lock_conf = 0.7 * self.lock_conf + 0.3 * mean_conf
            elif share >= self.lock_threshold and mean_conf >= self.conf_threshold:
                # Challenger cleared the higher bar — unseat
                self.committed = best
                self.lock_conf = mean_conf

        return self.committed if self.committed else best

    def reset(self) -> None:
        """Clear state when a track is reassigned or permanently lost."""
        self._history.clear()
        self.committed = ""
        self.locked    = False
        self.lock_conf = 0.0


def setup_video_source() -> Tuple[cv2.VideoCapture, Optional[cv2.VideoWriter]]:
    save_output = False
    video_writer = None
    source = input("Source (video / webcam): ").strip().lower()
    assert source in {"video", "webcam"}

    if source == "video":
        video_path = input("Video path: ").strip()
        cap = cv2.VideoCapture(video_path)

        # Bug fix: the old code read one frame to get dimensions then never
        # rewound, so the video always started one frame late. Read dimensions
        # directly from the capture properties instead.
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w == 0 or h == 0:
            raise RuntimeError("Could not read video dimensions")

        save_output = input("Save output video? (y/n): ").strip().lower() == "y"
        if save_output:
            fps = int(input("Enter FPS for output video: "))
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


def setup_identity_store(
    store: IdentityStore,
    device: str,
    emb_dim: int,
) -> Tuple[List[str], torch.Tensor, Optional[HopfieldLayer]]:
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
    quality_checker: FaceQualityChecker,
    temporal_checker: TemporalConsistencyChecker,
    embedder: MobileFaceNet,
    identity_memory: dict,
    identity_memory_pooled: dict,
    identity_voters: dict,
    track_info: dict,
    id_names: List[str],
    gallery: torch.Tensor,
    hop: Optional[HopfieldLayer],
    device: str,
    emb_dim: int,
    min_samples: int,
    video_writer: Optional[cv2.VideoWriter],
    latest_state: dict,  # for API
    latest_frame: dict,  # for API
) -> bool:
    ret, frame = cap.read()
    if not ret:
        return False

    h, w = frame.shape[:2]

    # ── Detection (adaptive sampler) ─────────────────────────────────────────
    # should_detect, reason, _ = sampler.should_run_detector(tracker)
    if True:
        dets, dets_conf = detector.boxes_to_yolox(detector.detect(frame))
        sampler.record_detection(tracker)
    else:
        dets = torch.empty((0, 6), device=device)
        dets_conf = torch.empty((0,), device=device)

    # ── Tracking ──────────────────────────────────────────────────────────────
    # Fix: on skipped detection frames, ByteTrack receives empty dets and
    # immediately ages out tracks, causing boxes to flicker off every non-detect
    # frame. The tracker's Kalman filter can predict positions without any
    # detections — we just need to call update() normally and then RE-ACTIVATE
    # any track that was active last frame but got deactivated purely because
    # there were no detections this frame (not because it's genuinely lost).
    tracks = tracker.update(dets, img_info=(h, w), img_size=(h, w))

    # Re-activate tracks that ByteTrack marked inactive only because dets were
    # empty this frame. A track is genuinely lost only after max_time_lost
    # consecutive missed frames — not after a single skipped detection.
    # We identify "spuriously deactivated" tracks as those in lost_stracks
    # whose last seen frame is THIS frame (deactivated right now, not before).
    # if not should_detect:
    #     for t in tracker.lost_stracks:
    #         if t.frame_id == tracker.frame_id and t.track_id in track_info:
    #             # Predict position forward one step using Kalman
    #             t.predict()
    #             # Re-mark as activated so visualiser picks it up
    #             t.is_activated = True
    #             tracks = list(tracks) + [t]

    # ── Attach YOLO confidence to each active track ───────────────────────────
    # Bug fix: dets_conf is a torch tensor — must convert to numpy before
    # calling .astype(). The old code called .astype() directly on the tensor.
    active_tracks = [t for t in tracks if t.is_activated]
    if len(active_tracks) > 0 and dets.shape[0] > 0:
        track_boxes = np.asarray([t.tlbr for t in active_tracks], dtype=np.float64)
        det_boxes   = to_numpy(dets[:, :4])
        det_scores  = to_numpy(dets_conf, dtype=np.float64)
        iou_mat     = ious(track_boxes, det_boxes)
        for i, t in enumerate(active_tracks):
            j = int(np.argmax(iou_mat[i]))
            t.det_conf = float(det_scores[j]) if iou_mat[i, j] > 0.3 else None
    else:
        for t in active_tracks:
            t.det_conf = None

    # Build a stable id→track lookup here, once, for the whole frame.
    # Bug fix: the old code rebuilt this inside the embedding loop on every
    # iteration, and used `t` from the outer for-loop (wrong track object).
    track_by_id = {t.track_id: t for t in tracks if t.is_activated}

    # ── Quality assessment + face crop ───────────────────────────────────────
    frame_rgb = frame[..., ::-1]
    faces: list          = []
    tids: list           = []
    quality_reports: dict = {}
    temporal_cache: dict  = {}

    for t in active_tracks:
        x1, y1, x2, y2 = map(int, t.tlbr)
        face = safe_crop_np(frame_rgb, (x1, y1, x2, y2))
        if face is None:
            continue

        report = quality_checker.assess(
            face_bgr=face[..., ::-1],
            box=(x1, y1, x2, y2),
            frame_shape=frame.shape,
        )
        quality_reports[t.track_id] = report

        # Update temporal consistency even for rejected frames so the
        # temporal score reflects actual observation history.
        temp = temporal_checker.update(t.track_id, report["quality"])
        temporal_cache[t.track_id] = temp

        # Feed quality signal to sampler every frame — not just on reid.
        sampler.update_quality(t.track_id, report["quality"])

        if report["quality"] < 0.5:
            track_info[t.track_id] = {
                "name": "REJECTED",
                "is_abstain": True,
                "id_conf": 0.0,
                "quality": report["quality"],
                "quality_level": report["level"],
                "quality_reasons": report["reasons"],
                "temporal_score": temp["temporal_score"],
                "temporal_level": temp["level"],
                "track_conf": float(t.score),
                "warning": "poor input quality",
            }
            continue
        
        # cv2.imshow(f"face_{t.track_id}", face)
        faces.append(embedder.preprocess_face(face))
        tids.append(t.track_id)

    # ── Embedding + re-identification ─────────────────────────────────────────
    if faces:
        
        embed_result = embedder.embed_faces(faces, tids)

        for tid, emb in embed_result:
            # print(f"[DEBUG EMB] norm: {emb.norm().item():.3f}")
            q_report = quality_reports.get(tid)
            if q_report is None:
                continue

            # Bug fix: temporal_cache.get(tid) can return None if this tid
            # somehow bypassed the quality loop. Guard with a safe default.
            temp = temporal_cache.get(tid, {"temporal_score": 0.0, "level": "LOW"})

            # Bug fix: duplicate NaN/Inf check removed — the second one (below
            # buf.add) was unreachable because we already `continue`d above.
            if torch.isnan(emb).any() or torch.isinf(emb).any():
                print(f"[WARN] NaN/Inf embedding | track {tid} — skipping")
                continue
            if emb.norm() < 1e-6:
                print(f"[WARN] Zero-norm embedding | track {tid} — skipping")
                continue

            if tid not in identity_memory:
                identity_memory[tid] = EmbeddingBuffer(min_samples, emb_dim, device)
            buf = identity_memory[tid]
            buf.add(emb)

            if not buf.full():
                continue

            # ── Re-ID ────────────────────────────────────────────────────────
            # Stability fix: only re-run identify_person when the buffer has
            # received NEW embeddings since the last re-ID — not on every frame
            # once the buffer is full.  Track this with a per-tid counter.
            buf._reid_ptr = getattr(buf, "_reid_ptr", -1)
            if buf.ptr == buf._reid_ptr:
                # Buffer write pointer hasn't moved — no new embedding since
                # last re-ID.  Skip to avoid re-firing on stale data.
                continue
            buf._reid_ptr = buf.ptr

            emb_buf  = buf.get_all()
            pooled   = pool_embeddings(emb_buf, device=device)
            # refined, energy_before, energy, delta = refine_identity(pooled, hop)
            refined=pooled
            energy_before=1
            energy=1
            delta=1
            # Feed Hopfield energy and embedding drift to sampler.
            sampler.update_energy(tid, float(energy), float(delta))
            sampler.update_embedding(tid, refined)

            # print("\n[DEBUG INPUT TO IDENTIFY]")
            # print(f"Track ID: {tid}")
            # print(f"Quality: {q_report['quality']:.2f}")
            # print(f"Temporal: {temp['temporal_score']:.2f}")
            # print(f"Delta: {delta:.3f}")
            # if source == "video":

            name, score = identify_person_cosine(
                embedding=refined,
                gallery=gallery,
                id_names=id_names
               )
            # else:
            #     name, score = identify_person(
            #         embedding=refined,
            #         gallery=gallery,
            #         id_names=id_names,
            #         hop=hop,
            #         device=device,
            #     )

            # ── Trust / abstain logic ────────────────────────────────────────
            trust_low = (
                q_report["quality"]       < 0.6 or
                temp["temporal_score"]    < 0.6 or
                score                     < 0.7  or
                delta                     < 0.15
            )
            prev_abstain = track_info.get(tid, {}).get("is_abstain", False)

            if trust_low or (prev_abstain and score < 0.75):
                name = "ABSTAIN"

            # ── Calibrated confidence ────────────────────────────────────────
            calibrated_conf = score * q_report["quality"] * temp["temporal_score"]

            # ── Identity voter — the core stability fix ──────────────────────
            # Feed this window's result into the voter.  The voter accumulates
            # a rolling history and only commits a name change when a candidate
            # wins a supermajority of recent windows with sufficient confidence.
            # This prevents a single bad buffer window from flipping the display.
            if tid not in identity_voters:
                identity_voters[tid] = IdentityVoter()
            voter        = identity_voters[tid]
            stable_name  = voter.update(name, calibrated_conf)
            stable_conf  = voter.lock_conf if voter.locked else calibrated_conf

            # ── Causal warning string ────────────────────────────────────────
            warning_parts = []
            if q_report["quality"]    < 0.6:  warning_parts.append("low quality")
            if temp["temporal_score"] < 0.6:  warning_parts.append("unstable tracking")
            if score                  < 0.7:  warning_parts.append("weak match")
            if delta                  < 0.15: warning_parts.append("weak refinement")
            if not voter.locked:              warning_parts.append("gathering evidence")
            warning = " / ".join(warning_parts) or None

            t_obj = track_by_id.get(tid)

            track_info[tid] = {
                "name":           stable_name,
                "is_abstain":     stable_name in ("ABSTAIN", "Unknown", ""),
                "id_conf":        float(stable_conf),

                "E_before":       float(energy_before),
                "E_after":        float(energy),
                "dE":             float(delta),

                "det_conf":       getattr(t_obj, "det_conf", None),
                "track_conf":     float(t_obj.score) if t_obj else 0.0,

                "quality":        q_report["quality"],
                "quality_level":  q_report["level"],
                "quality_reasons": q_report["reasons"],

                "temporal_score": temp["temporal_score"],
                "temporal_level": temp["level"],

                "warning":        warning,
            }

            identity_memory_pooled[tid] = pooled

    # ── Prune voters for tracks that have disappeared ─────────────────────────
    active_ids = {t.track_id for t in tracks if t.is_activated}
    for lost_tid in list(identity_voters.keys()):
        if lost_tid not in active_ids:
            identity_voters[lost_tid].reset()
            del identity_voters[lost_tid]


    # print("\n[DEBUG FINAL]")
    # print(f"Raw name: {name}")
    # print(f"Stable name: {stable_name}")
    # print(f"Confidence: {stable_conf:.3f}")
    # print(f"Warning: {warning}")
    # ── Visualisation ─────────────────────────────────────────────────────────
    # Overlay sampler state: trigger reason + system confidence mode.
    # mode        = FrameSampler.confidence_mode(sampler.last_S)
    # mode_color  = {"STABLE": (0, 200, 0), "WATCH": (0, 200, 255), "UNSTABLE": (0, 60, 255)}[mode]
    # cv2.putText(frame, reason, (10, 28),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1, cv2.LINE_AA)
    # cv2.putText(frame, f"Mode: {mode}", (10, 52),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.52, mode_color, 1, cv2.LINE_AA)

    frame = draw_tracks(frame, tracks, track_info)

    if video_writer:
        video_writer.write(frame)

    cv2.imshow("Skynetra Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        return False
    
    #New code to prepare API response with track info and latest frame
    import base64

    face_images = {}

    for t in active_tracks:
        tid = t.track_id
        x1, y1, x2, y2 = map(int, t.tlbr)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop is None or face_crop.size == 0:
            continue

        _, buffer = cv2.imencode('.jpg', face_crop)
        face_base64 = base64.b64encode(buffer).decode('utf-8')

        face_images[tid] = face_base64
        
       
    tracks_output = []

    for tid, info in track_info.items():

        # 🔹 Step 1: compute condition
        needs_confirmation = (
            info.get("id_conf", 0) < 0.6 or
            info.get("temporal_score", 0) < 0.6
        )

        # 🔹 Step 2: decide whether to send face
        send_face = False

        if needs_confirmation:
            if tid not in sent_faces:
                send_face = True
                sent_faces[tid] = True

        # 🔹 Step 3: get face
        face = face_images.get(tid) if send_face else None

        # 🔹 Step 4: build dict
        tracks_output.append({
            "id": tid,
            "name": info.get("name"),

            "confidence": round(
                info.get("id_conf", 0) *
                info.get("quality", 0) *
                info.get("temporal_score", 0),
                2
            ),

            "raw_conf": round(info.get("id_conf", 0), 2),
            "quality": round(info.get("quality", 0), 2),
            "temporal": round(info.get("temporal_score", 0), 2),

            "trust": (
                "HIGH" if info.get("id_conf", 0) > 0.8 and info.get("quality", 0) > 0.8
                else "MEDIUM" if info.get("id_conf", 0) > 0.6
                else "LOW"
            ),

            "needs_confirmation": needs_confirmation,
            "face": face
        })

    # 🔥 final assignment
    latest_state["tracks"] = tracks_output
    latest_state["frame"] = latest_state.get("frame", 0) + 1

    latest_frame["frame"] = frame.copy()
    # print("Frame updated")
    return True


def cleanup(cap: cv2.VideoCapture, video_writer: Optional[cv2.VideoWriter]) -> None:
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()