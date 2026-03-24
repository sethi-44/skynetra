"""
Adaptive Frame Sampler for SkyNetra
====================================
Decides each frame whether to run the YOLO detector or skip it.

Decision model
--------------
Every frame, each active/lost track gets a per-track urgency score U ∈ [0,1]
built from six independently-smoothed signals:

    Signal              Weight   Meaning when high
    ──────────────────  ──────   ────────────────────────────────────────────
    Kalman covariance   0.25     Tracker has grown uncertain about position
    Detection conf      0.20     Last matched detection was weak
    Cosine drift        0.20     Embedding is drifting — identity unstable
    Hopfield dE         0.15     Refinement energy is high — bad latent state
    Quality score       0.15     Face was blurry/occluded last time we saw it
    Motion              0.05     Track moved fast relative to its own size

Scene urgency S = 0.7 * max(U_i) + 0.3 * mean(U_i), plus additive bonuses:
    +0.3  if any track is lost (ByteTrack lost_stracks)
    +0.3  if >1 active-ID changed since last detection (topology change)
    clipped to [0, 1]

Gate logic (in order, first match wins):
    1. Unconfirmed tracks exist           → detect immediately
    2. frames_since_detect < min_gap      → skip (hard cooldown)
    3. No tracks at all, gap ≥ max_gap/2  → detect (scene may be empty)
    4. S ≥ urgency_threshold              → detect (system is uncertain)
    5. frames_since_detect ≥ max_gap      → detect (hard ceiling)
    6. Otherwise                          → skip

Integration with main_helpers.py
---------------------------------
Replace the `if True:` block with:

    should_detect, reason, S = sampler.should_run_detector(tracker)
    if should_detect:
        dets, dets_conf = detector.boxes_to_yolox(detector.detect(frame))
        sampler.record_detection(tracker)
    else:
        dets  = torch.empty((0, 6), device=device)
        dets_conf = torch.empty((0,), device=device)

After identity is resolved for a track, feed signals back:

    sampler.update_embedding(tid, refined_emb)           # after refine_identity
    sampler.update_energy(tid, energy_after, delta_e)    # after refine_identity
    sampler.update_quality(tid, q_report["quality"])     # after quality_checker.assess
    # motion and covariance are updated automatically every frame

UI overlay (add to draw_tracks or process_frame):

    # Detector trigger reason — e.g. "Trigger: quality + lost (U=0.73)"
    cv2.putText(frame, reason, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # System confidence mode — uses sampler.last_S so cooldown frames still
    # show the correct mode from the most recently computed urgency score.
    mode = FrameSampler.confidence_mode(sampler.last_S)
    mode_color = {"STABLE": (0,200,0), "WATCH": (0,200,255), "UNSTABLE": (0,60,255)}[mode]
    cv2.putText(frame, f"Mode: {mode}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Per-track signal state — one instance per track ID
# ---------------------------------------------------------------------------
@dataclass
class TrackSignals:
    """All smoothed signals for one track."""

    # Kalman covariance trace (normalised to [0,1])
    cov_score: float = 0.0

    # 1 - detection_confidence  (high = weak detection)
    det_uncertainty: float = 0.5

    # Cosine drift from previous embedding (0 = stable, 1 = completely different)
    cos_drift: float = 0.0
    prev_emb: Optional[torch.Tensor] = None
    last_emb_frame: int = -1  # frame index of the last update_embedding() call

    # Hopfield energy after refinement (normalised)
    energy_score: float = 0.0

    # 1 - quality  (high = poor face quality)
    quality_uncertainty: float = 0.5

    # Normalised motion  (displacement / face_size, clipped to [0,1])
    motion_score: float = 0.0
    prev_tlbr: Optional[Tuple[int, int, int, int]] = None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class FrameSampler:
    """
    Adaptive detector scheduler for SkyNetra.

    Parameters
    ----------
    min_gap : int
        Minimum frames between detector calls (hard cooldown).
    max_gap : int
        Maximum frames between detector calls (hard ceiling).
    urgency_threshold : float
        Scene urgency [0,1] above which we trigger detection.
        0.5 is a reasonable starting point; lower = more sensitive.
    ema_alpha : float
        EMA smoothing for all per-track signals.
        Closer to 1.0 = faster response, more noise.
        Closer to 0.0 = slower response, smoother.
    cov_norm : float
        Kalman covariance trace value that maps to urgency = 1.0.
        Tune by printing np.trace(t.covariance) for your tracker.
    energy_norm : float
        Hopfield energy value that maps to urgency = 1.0.
    motion_norm_cap : float
        Displacement/face_size ratio that maps to urgency = 1.0.
    weights : dict | None
        Override the default signal weights. Must sum to 1.0.
    """

    # Default signal weights — must sum to 1.0
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "cov":     0.25,
        "det":     0.20,
        "cos":     0.20,
        "energy":  0.15,
        "quality": 0.15,  # raised: quality directly gates embedding reliability
        "motion":  0.05,  # lowered: noisy signal, ByteTrack handles motion well
    }

    def __init__(
        self,
        min_gap: int = 3,
        max_gap: int = 20,
        urgency_threshold: float = 0.50,
        ema_alpha: float = 0.40,
        cov_norm: float = 10.0,
        energy_norm: float = 0.5,
        motion_norm_cap: float = 0.8,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.urgency_threshold = urgency_threshold
        self.ema_alpha = ema_alpha
        self.cov_norm = cov_norm
        self.energy_norm = energy_norm
        self.motion_norm_cap = motion_norm_cap

        w = weights or self.DEFAULT_WEIGHTS
        assert abs(sum(w.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.weights = w

        # Frame counters
        self.frame_idx: int = 0
        self.last_detect_frame: int = -1

        # Last computed scene urgency — persists across frames so fast-path
        # returns (cooldown, unconfirmed) still yield a meaningful S for
        # confidence_mode() rather than a misleading 0.0.
        self.last_S: float = 0.0

        # Per-track signal state, keyed by track_id
        self._signals: Dict[int, TrackSignals] = {}

        # Topology tracking (set of active IDs at last detection)
        self._prev_active_ids: set = set()

    # ------------------------------------------------------------------
    # Public API — main loop
    # ------------------------------------------------------------------

    def should_run_detector(self, tracker) -> Tuple[bool, str, float]:
        """
        Call once per frame BEFORE detection. Increments frame_idx.

        Returns
        -------
        (should_detect, reason_string, scene_urgency_S)

        scene_urgency_S is always returned so callers can pass it to
        FrameSampler.confidence_mode(S) for UI display without storing
        any extra state themselves. Returns 0.0 when S was never computed
        (cooldown / empty-scene fast paths).
        """
        self.frame_idx += 1
        gap = self.frame_idx - self.last_detect_frame

        # ── 1. Unconfirmed tracks always need a follow-up detection ──────
        num_unconfirmed = sum(
            not t.is_activated for t in tracker.tracked_stracks
        )
        if num_unconfirmed > 0:
            return True, f"Trigger: unconfirmed({num_unconfirmed})", self.last_S

        # ── 2. Hard cooldown ─────────────────────────────────────────────
        if gap < self.min_gap:
            return False, f"cooldown({gap})", self.last_S

        # ── 3. Scene completely empty ────────────────────────────────────
        all_tracks = tracker.tracked_stracks + tracker.lost_stracks
        if len(all_tracks) == 0 and gap >= self.max_gap // 2:
            return True, "Trigger: empty scene", self.last_S

        # ── 4. Compute scene urgency and check threshold ─────────────────
        # If the gap is unusually long, EMA signals may be stale — decay
        # volatile signals so the system doesn't stay falsely confident.
        if gap > self.max_gap:
            for sig in self._signals.values():
                sig.motion_score *= 0.8
                sig.cos_drift    *= 0.8

        S, reason = self._scene_urgency(tracker, gap)
        self.last_S = S  # persist for fast-path frames and UI callers

        if S >= self.urgency_threshold:
            return True, f"{reason} (U={S:.2f})", S

        # ── 5. Hard max-gap ceiling ──────────────────────────────────────
        if gap >= self.max_gap:
            return True, f"Trigger: max gap ({gap}f)", S

        return False, f"skip (U={S:.2f}, gap={gap})", S

    def record_detection(self, tracker) -> None:
        """
        Call immediately after running the detector (when should_run_detector
        returned True). Updates last_detect_frame, syncs position cache, and
        prunes state for vanished tracks.
        """
        self.last_detect_frame = self.frame_idx

        # Snapshot active IDs for topology change detection
        active_ids = {
            t.track_id
            for t in tracker.tracked_stracks
            if t.is_activated
        }
        self._prev_active_ids = active_ids

        # Prune state for tracks that have fully disappeared
        alive = {t.track_id for t in tracker.tracked_stracks + tracker.lost_stracks}
        vanished = set(self._signals.keys()) - alive
        for tid in vanished:
            del self._signals[tid]

    # ------------------------------------------------------------------
    # Signal update API — call from main_helpers.py after each signal fires
    # ------------------------------------------------------------------

    def update_embedding(self, tid: int, refined_emb: torch.Tensor) -> None:
        """
        Feed in the Hopfield-refined embedding for a track after identity
        resolution. Computes cosine drift against the previous embedding.

        Call after: refine_identity()
        """
        sig = self._get_or_create(tid)

        # Normalise to unit sphere (safe)
        norm = refined_emb.norm()
        if norm < 1e-6:
            return
        emb = (refined_emb / norm).detach().cpu().float()

        if sig.prev_emb is not None:
            cos_sim = float(torch.dot(emb, sig.prev_emb).clamp(-1.0, 1.0))
            raw_drift = float(np.clip(1.0 - cos_sim, 0.0, 1.0))
            sig.cos_drift = self._ema(sig.cos_drift, raw_drift)

        sig.prev_emb = emb
        sig.last_emb_frame = self.frame_idx  # mark as freshly updated this frame

    def update_energy(self, tid: int, energy_after: float, delta_e: float) -> None:
        """
        Feed in Hopfield energy metrics after refine_identity().

        Parameters
        ----------
        energy_after : float  — E_after from refine_identity
        delta_e      : float  — E_before - E_after (positive = good refinement)
        """
        sig = self._get_or_create(tid)
        # Soft exponential scaling — stable across a wide range of energy values.
        # Linear clipping (energy / norm) collapses to 0 or 1 if the energy
        # distribution shifts; 1 - exp(-x) saturates gracefully instead.
        raw = float(1.0 - np.exp(-energy_after / max(self.energy_norm, 1e-6)))
        # If delta_e is tiny, refinement did nothing → more uncertain
        refinement_weakness = float(np.clip(1.0 - delta_e / max(self.energy_norm, 1e-6), 0.0, 1.0))
        combined = 0.6 * raw + 0.4 * refinement_weakness
        sig.energy_score = self._ema(sig.energy_score, combined)

    def update_quality(self, tid: int, quality: float) -> None:
        """
        Feed in the FaceQualityChecker quality score [0,1] for a track.

        Call after: quality_checker.assess()
        """
        sig = self._get_or_create(tid)
        raw = float(np.clip(1.0 - quality, 0.0, 1.0))
        sig.quality_uncertainty = self._ema(sig.quality_uncertainty, raw)

    # ------------------------------------------------------------------
    # Internal: urgency computation
    # ------------------------------------------------------------------

    def _scene_urgency(self, tracker, gap: int) -> Tuple[float, str]:
        """
        Compute scene-level urgency S ∈ [0, 1].
        Returns (S, dominant_reason_string).
        """
        active_tracks = [t for t in tracker.tracked_stracks if t.is_activated]
        lost_tracks   = tracker.lost_stracks

        # ── Update covariance + detection-conf signals from tracker state ─
        for t in active_tracks:
            sig = self._get_or_create(t.track_id)
            self._update_from_tracker(t, sig, tracker.frame_id)

        # ── Per-track urgency ─────────────────────────────────────────────
        track_urgencies: list[float] = []
        dominant_signals: list[str] = []

        weight_sum = sum(self.weights.values())  # normally 1.0; guard for future edits

        for t in active_tracks:
            sig = self._signals.get(t.track_id)
            if sig is None:
                # Unknown track: use a conservative baseline — not urgent by default.
                # 0.5 was arbitrary and could randomly tip the threshold; 0.3 is
                # a deliberate "we know nothing, assume mild uncertainty" value.
                track_urgencies.append(0.3)
                dominant_signals.append("unknown")
                continue

            # Decay cos_drift only when embedding is stale (>2 frames since last
            # update_embedding call). Active embedding flow = real drift signal;
            # don't suppress it. Stale drift = old news; let it decay.
            if self.frame_idx - sig.last_emb_frame > 2:
                sig.cos_drift *= 0.95

            weighted = {
                "cov":     self.weights["cov"]     * sig.cov_score,
                "det":     self.weights["det"]     * sig.det_uncertainty,
                "cos":     self.weights["cos"]     * sig.cos_drift,
                "energy":  self.weights["energy"]  * sig.energy_score,
                "quality": self.weights["quality"] * sig.quality_uncertainty,
                "motion":  self.weights["motion"]  * sig.motion_score,
            }
            # Normalise by weight sum: U stays in [0,1] even if weights are
            # tweaked and no longer sum to exactly 1.0.
            U = float(np.clip(sum(weighted.values()) / weight_sum, 0.0, 1.0))
            track_urgencies.append(U)
            dominant_signals.append(max(weighted, key=weighted.__getitem__))

        # dominant signal = the weighted contribution that drove the
        # highest-urgency track — surfaced in the reason string.
        if track_urgencies:
            S = 0.7 * max(track_urgencies) + 0.3 * float(np.mean(track_urgencies))
            peak_idx = int(np.argmax(track_urgencies))
            dominant = dominant_signals[peak_idx] if peak_idx < len(dominant_signals) else "urgency"
        else:
            S = 0.0
            dominant = "urgency"
        reason = f"urgency:{dominant}"

        # ── Bonus: lost tracks ────────────────────────────────────────────
        # Smooth exponential scaling: 1 lost ≈ 0.19, 3 ≈ 0.28, ∞ → 0.30.
        if len(lost_tracks) > 0:
            lost_bonus = 0.3 * float(1.0 - np.exp(-len(lost_tracks)))
            S = min(S + lost_bonus, 1.0)
            reason += "+lost"  # append — don't overwrite the dominant signal

        # ── Bonus: topology change ────────────────────────────────────────
        current_ids = {t.track_id for t in active_tracks}
        topology_delta = len(current_ids.symmetric_difference(self._prev_active_ids))
        if topology_delta > 1 and gap >= self.min_gap:
            S = min(S + 0.30, 1.0)
            reason += "+topology"  # append — compound, not replace

        # Format reason into clean human-readable form.
        # Raw internal form: "urgency:quality+lost+topology"
        # Rendered form:     "Trigger: quality + lost + topology"
        reason = reason.replace("urgency:", "")
        reason = " + ".join(reason.split("+"))
        reason = f"Trigger: {reason}"

        return S, reason

    def _update_from_tracker(self, t, sig: TrackSignals, frame_id: int) -> None:
        """Sync covariance and detection confidence from ByteTrack state."""
        # Kalman covariance — how uncertain is the tracker about this position
        if t.covariance is not None:
            try:
                trace = float(np.trace(t.covariance))
                raw_cov = float(np.clip(trace / self.cov_norm, 0.0, 1.0))
                sig.cov_score = self._ema(sig.cov_score, raw_cov)
            except Exception:
                pass

        # Detection uncertainty — prefer t.det_conf (YOLO detection score set
        # in main_helpers.py) over t.score (ByteTrack's internal tracker score).
        # t.score is tracker-smoothed and drifts away from detection reality;
        # t.det_conf is the raw YOLO confidence from the last matched detection.
        if hasattr(t, "det_conf") and t.det_conf is not None:
            raw_det = float(np.clip(1.0 - t.det_conf, 0.0, 1.0))
        else:
            raw_det = float(np.clip(1.0 - t.score, 0.0, 1.0))
        sig.det_uncertainty = self._ema(sig.det_uncertainty, raw_det)

        # Motion — compute from stored previous position, then immediately
        # update the baseline so the next frame measures frame-to-frame
        # displacement rather than detection-to-detection displacement.
        # Without this, skipped-detection frames accumulate displacement and
        # produce artificially high motion scores that trigger unnecessary detections.
        x1, y1, x2, y2 = map(int, t.tlbr)
        if sig.prev_tlbr is not None:
            px1, py1, px2, py2 = sig.prev_tlbr
            cx,  cy  = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
            disp = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
            face_size = max(x2 - x1, y2 - y1, 1)
            raw_motion = float(np.clip(disp / face_size / self.motion_norm_cap, 0.0, 1.0))
            sig.motion_score = self._ema(sig.motion_score, raw_motion)
        sig.prev_tlbr = (x1, y1, x2, y2)  # always advance baseline

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _ema(self, prev: float, new: float) -> float:
        """Exponential moving average. alpha = weight on the new value."""
        return self.ema_alpha * new + (1.0 - self.ema_alpha) * prev

    def _get_or_create(self, tid: int) -> TrackSignals:
        if tid not in self._signals:
            sig = TrackSignals()
            # prev_tlbr starts as None so the first call to _update_from_tracker
            # skips motion computation instead of measuring displacement from a
            # position that belonged to a different track that held this ID before.
            # This guards against motion spikes on ID reassignment by ByteTrack.
            sig.prev_tlbr = None
            self._signals[tid] = sig
        return self._signals[tid]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def confidence_mode(S: float) -> str:
        """
        Map scene urgency S to a human-readable system confidence mode.

        STABLE   — system is tracking confidently, detector rarely needed
        WATCH    — moderate uncertainty, system is monitoring closely
        UNSTABLE — high uncertainty, detection is being triggered often

        Usage in main_helpers.py:
            mode = FrameSampler.confidence_mode(last_S)
            cv2.putText(frame, f"Mode: {mode}", (10, 50), ...)
        """
        if S < 0.3:
            return "STABLE"
        elif S < 0.6:
            return "WATCH"
        else:
            return "UNSTABLE"

    def debug_state(self) -> dict:
        """
        Returns a dict of per-track signal breakdowns.
        Useful for logging or overlaying on the visualisation.
        """
        out = {}
        for tid, sig in self._signals.items():
            out[tid] = {
                "cov":     round(sig.cov_score, 3),
                "det":     round(sig.det_uncertainty, 3),
                "cos":     round(sig.cos_drift, 3),
                "energy":  round(sig.energy_score, 3),
                "quality": round(sig.quality_uncertainty, 3),
                "motion":  round(sig.motion_score, 3),
                "U": round(
                    self.weights["cov"]     * sig.cov_score          +
                    self.weights["det"]     * sig.det_uncertainty     +
                    self.weights["cos"]     * sig.cos_drift           +
                    self.weights["energy"]  * sig.energy_score        +
                    self.weights["quality"] * sig.quality_uncertainty +
                    self.weights["motion"]  * sig.motion_score,
                    3
                ),
            }
        return out