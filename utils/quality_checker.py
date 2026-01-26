"""
Face Quality Assessment Module (Rewritten)
-----------------------------------------
Produces a continuous quality score (0–1) instead of hard rejection.
Designed for video-based face recognition pipelines.
"""

import cv2
import numpy as np
import torch


# ------------------------------------------------------------
# Face Quality Checker
# ------------------------------------------------------------
class FaceQualityChecker:
    def __init__(
        self,
        min_face_size=80,
        max_face_size=800,
        blur_norm_thresh=0.0015,          # normalized Laplacian variance
        brightness_range=(40, 220),
        aspect_ratio_range=(0.6, 1.4),
        min_edge_distance=20,
    ):
        self.min_face_size = min_face_size
        self.max_face_size = max_face_size
        self.blur_norm_thresh = blur_norm_thresh
        self.brightness_range = brightness_range
        self.aspect_ratio_range = aspect_ratio_range
        self.min_edge_distance = min_edge_distance

    # ----------------------------
    # Individual metrics (0–1)
    # ----------------------------
    def blur_score(self, gray, w, h):
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        norm = lap / max(w * h, 1)
        return np.clip(norm / self.blur_norm_thresh, 0.0, 1.0)

    def brightness_score(self, gray):
        mean = gray.mean()
        lo, hi = self.brightness_range
        if mean < lo:
            return mean / lo
        if mean > hi:
            return max(0.0, 1.0 - (mean - hi) / 50.0)
        return 1.0

    def size_score(self, w, h):
        size = max(w, h)
        if size < self.min_face_size:
            return size / self.min_face_size
        if size > self.max_face_size:
            return max(0.0, 1.0 - (size - self.max_face_size) / self.max_face_size)
        return 1.0

    def aspect_ratio_score(self, w, h):
        r = w / max(h, 1)
        lo, hi = self.aspect_ratio_range
        if r < lo:
            return r / lo
        if r > hi:
            return max(0.0, 1.0 - (r - hi) / hi)
        return 1.0

    def edge_distance_score(self, x1, y1, x2, y2, fw, fh):
        d = min(x1, y1, fw - x2, fh - y2)
        if d >= self.min_edge_distance:
            return 1.0
        return max(0.0, d / self.min_edge_distance)

    def occlusion_score(self, gray):
        edges = cv2.Canny(gray, 50, 150)
        ratio = np.count_nonzero(edges) / gray.size

        # Ideal range ~0.05–0.30, soft penalty outside
        if ratio < 0.05:
            return ratio / 0.05
        if ratio > 0.30:
            return max(0.0, 1.0 - (ratio - 0.30) / 0.20)
        return 1.0

    # ----------------------------
    # Main assessment
    # ----------------------------
    def assess(self, face_bgr, box, frame_shape):
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        fh, fw = frame_shape[:2]

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        scores = {
            "blur": self.blur_score(gray, w, h),
            "brightness": self.brightness_score(gray),
            "size": self.size_score(w, h),
            "aspect": self.aspect_ratio_score(w, h),
            "edge": self.edge_distance_score(x1, y1, x2, y2, fw, fh),
            "occlusion": self.occlusion_score(gray),
        }

        # Weighted aggregation (sums to 1.0)
        quality = (
            0.25 * scores["blur"] +
            0.20 * scores["brightness"] +
            0.20 * scores["size"] +
            0.15 * scores["aspect"] +
            0.10 * scores["edge"] +
            0.10 * scores["occlusion"]
        )

        return {
            "quality": float(np.clip(quality, 0.0, 1.0)),
            "scores": scores
        }

    # ----------------------------
    # Detection filtering
    # ----------------------------
    def filter_detections(self, frame, boxes, min_quality=0.6):
        if len(boxes) == 0:
            return [], []

        h, w = frame.shape[:2]
        kept, reports = [], []

        box_data = (
            boxes.xyxy[0].cpu().numpy()
            if hasattr(boxes, "xyxy")
            else boxes
        )

        for box in box_data:
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            report = self.assess(face, (x1, y1, x2, y2), frame.shape)

            if report["quality"] >= min_quality:
                kept.append(box)
                reports.append(report)

        return kept, reports


# ------------------------------------------------------------
# Temporal Quality Consistency
# ------------------------------------------------------------
class TemporalConsistencyChecker:
    def __init__(self, memory=10, threshold=0.65):
        self.memory = memory
        self.threshold = threshold
        self.history = {}  # tid → [quality]

    def update(self, tid, quality):
        buf = self.history.setdefault(tid, [])
        buf.append(float(quality))

        if len(buf) > self.memory:
            buf.pop(0)

        if len(buf) < 3:
            return False

        return np.mean(buf) >= self.threshold

    def cleanup(self, active_ids):
        self.history = {
            tid: hist for tid, hist in self.history.items()
            if tid in active_ids
        }
