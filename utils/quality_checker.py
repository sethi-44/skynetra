"""
Face Quality Assessment Module (Explainable Version)
--------------------------------------------------
Adds:
- Weight justification
- Reason generation
- Reliability levels
- Continuous temporal scoring
"""

import cv2
import numpy as np


# ------------------------------------------------------------
# Face Quality Checker
# ------------------------------------------------------------
class FaceQualityChecker:
    def __init__(
        self,
        min_face_size=80,
        max_face_size=800,
        blur_norm_thresh=0.0015,
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
        # Weight configuration
        # ----------------------------
        self.weights = {
            "blur": 0.25,
            "brightness": 0.20,
            "size": 0.20,
            "aspect": 0.15,
            "edge": 0.10,
            "occlusion": 0.10
        }

        # ----------------------------
        # Weight justification (EXPLAINABILITY)
        # ----------------------------
        self.weight_reasoning = {
            "blur": "Sharpness is critical for extracting facial features",
            "brightness": "Lighting affects visibility and contrast",
            "size": "Sufficient face resolution is required for recognition",
            "aspect": "Distorted face shapes reduce embedding quality",
            "edge": "Faces near borders are often incomplete",
            "occlusion": "Obstructions hide key identity features"
        }

    # ----------------------------
    # Individual metrics
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

    # def occlusion_score(self, gray):
    #     edges = cv2.Canny(gray, 50, 150)
    #     ratio = np.count_nonzero(edges) / gray.size

    #     if ratio < 0.05:
    #         return ratio / 0.05
    #     if ratio > 0.30:
    #         return max(0.0, 1.0 - (ratio - 0.30) / 0.20)
    #     return 1.0
    def occlusion_score(self, gray: np.ndarray) -> float:
        """
        Region-based occlusion score using pixel intensity variance.

        Divides the face into 4 horizontal zones and measures std-dev
        per zone. A uniform/flat patch (low std-dev) indicates occlusion
        (mask, hand, sunglasses, etc.). Zones are weighted by their
        importance to identity recognition.

        Returns:
            float in [0.0, 1.0] where 1.0 = fully visible, 0.0 = occluded
        """
        h, w = gray.shape

        zones = {
            # (y_start_frac, y_end_frac): weight
            "forehead":   ((0.00, 0.25), 0.15),
            "eyes":       ((0.25, 0.50), 0.35),
            "nose_mouth": ((0.50, 0.75), 0.35),
            "chin":       ((0.75, 1.00), 0.15),
        }

        # Minimum std-dev that counts as "textured / visible"
        # Empirically: plain skin ~12-18, occluded ~2-6, open face ~20-40
        TEXTURE_THRESHOLD = 15.0

        total_score = 0.0

        for (y0_frac, y1_frac), weight in zones.values():
            y0 = int(h * y0_frac)
            y1 = int(h * y1_frac)
            zone = gray[y0:y1, :]

            if zone.size == 0:
                # Degenerate crop — treat as invisible
                continue

            std = float(np.std(zone))
            zone_score = float(np.clip(std / TEXTURE_THRESHOLD, 0.0, 1.0))
            total_score += weight * zone_score

        return float(np.clip(total_score, 0.0, 1.0))

    # ----------------------------
    # Reliability level
    # ----------------------------
    def reliability_level(self, q):
        if q > 0.8:
            return "HIGH"
        elif q > 0.6:
            return "MEDIUM"
        return "LOW"

    # ----------------------------
    # Reason generator
    # ----------------------------
    def generate_reasons(self, scores):
        reasons = []

        if scores["blur"] < 0.5:
            reasons.append("Image is blurry")

        if scores["brightness"] < 0.5:
            reasons.append("Poor lighting conditions")

        if scores["size"] < 0.5:
            reasons.append("Face resolution too small")

        if scores["aspect"] < 0.5:
            reasons.append("Face angle or distortion detected")

        if scores["edge"] < 0.5:
            reasons.append("Face too close to frame boundary")

        if scores["occlusion"] < 0.5:
            reasons.append("Face partially occluded")

        if not reasons:
            reasons.append("Good quality face")

        return reasons

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

        # Weighted aggregation
        quality = sum(self.weights[k] * scores[k] for k in scores)

        quality = float(np.clip(quality, 0.0, 1.0))

        return {
            "quality": quality,
            "level": self.reliability_level(quality),
            "scores": scores,
            "reasons": self.generate_reasons(scores),
            "weight_justification": self.weight_reasoning
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
# Temporal Consistency Checker (Improved)
# ------------------------------------------------------------
class TemporalConsistencyChecker:
    def __init__(self, memory=10, threshold=0.65):
        self.memory = memory
        self.threshold = threshold
        self.history = {}

    def update(self, tid, quality):
        buf = self.history.setdefault(tid, [])
        buf.append(float(quality))

        if len(buf) > self.memory:
            buf.pop(0)

        if len(buf) == 0:
            return {"temporal_score": 0.0, "stable": False, "level": "LOW"}

        score = float(np.mean(buf))

        return {
            "temporal_score": score,
            "stable": score >= self.threshold,
            "level": "HIGH" if score > 0.8 else "MEDIUM" if score > 0.6 else "LOW"
        }

    def cleanup(self, active_ids):
        self.history = {
            tid: hist for tid, hist in self.history.items()
            if tid in active_ids
        }