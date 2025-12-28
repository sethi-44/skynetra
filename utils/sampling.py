import numpy as np
import torch
import torch.nn.functional as F

class FrameSampler:
    def __init__(
        self,
        min_gap=2,
        max_gap=15,
        conf_thresh=0.5,
        motion_thresh=25,
        id_conf_thresh=0.7,
        id_conf_decay=0.995,
        id_conf_alpha=0.7,
        energy_thresh=0.45,
        energy_delta_thresh=0.12,
        energy_smooth=0.5,
        cos_thresh=0.82,          # <-- new: absolute cosine threshold
        cos_delta_thresh=0.08,    # <-- new: cosine drop allowed
        cos_smooth=0.5            # <-- new: smoothing for drift
    ):
        self.frame_idx = 0
        self.last_detect_frame = -1

        # tracker stuff
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.conf_thresh = conf_thresh
        self.motion_thresh = motion_thresh

        # identity confidence
        self.id_conf_thresh = id_conf_thresh
        self.id_conf_decay  = id_conf_decay
        self.id_conf_alpha  = id_conf_alpha
        self.prev_id_conf   = {}

        # hopfield energy
        self.energy_thresh       = energy_thresh
        self.energy_delta_thresh = energy_delta_thresh
        self.energy_smooth       = energy_smooth
        self.prev_energy         = {}
        self.prev_energy_trend   = {}

        # cosine drift
        self.cos_thresh       = cos_thresh
        self.cos_delta_thresh = cos_delta_thresh
        self.cos_smooth       = cos_smooth
        self.prev_emb         = {}   # tid → embedding
        self.prev_cos         = {}   # tid → smoothed similarity
        self.prev_cos_delta   = {}   # tid → smoothed drift

        # motion tracking
        self.prev_pos = {}
        self.prev_active_ids = set()


    # ------------------------------------------------------
    # MAIN LOGIC
    # ------------------------------------------------------
    def should_run_detector(self, tracks):
        self.frame_idx += 1
        frames_since_detect = self.frame_idx - self.last_detect_frame
        active_ids = {t[4] for t in tracks if t[6] == 0}

        if len(tracks) == 0:
            return True, "no_tracks"

        if active_ids != self.prev_active_ids:
            return True, "id_topology_change"

        live_confs = [t[5] for t in tracks if t[6]==0]
        if live_confs:
            avg_conf = np.mean(live_confs)
            if avg_conf < self.conf_thresh:
                return True, f"low_tracker_conf({avg_conf:.2f})"
        else:
            return True, "no_live_tracks"


        # ---------- PER-TRACK CHECKS ----------
        for t in tracks:
            if t[6] != 0:
                continue

            tid = t[4]
            x1, y1, x2, y2 = map(int, t[0:4])

            # --- motion ---
            prev = self.prev_pos.get(tid)
            if prev:
                px1, py1, px2, py2 = prev
                motion = abs(x1-px1) + abs(y1-py1)
                if motion > self.motion_thresh:
                    return True, f"motion({tid}:{motion:.1f})"

            # --- identity confidence drift ---
            prev_conf = self.prev_id_conf.get(tid)
            if prev_conf is not None:
                decayed_conf = prev_conf * (self.id_conf_decay ** frames_since_detect)
                if decayed_conf < self.id_conf_thresh:
                    return True, f"id_conf({tid}:{decayed_conf:.2f})"


            # --- hopfield absolute energy ---
            E = self.prev_energy.get(tid)
            if E and E > self.energy_thresh:
                return True, f"energy_high({tid}:{E:.3f})"

            # --- hopfield dE ---
            dE = self.prev_energy_trend.get(tid)
            if dE and dE > self.energy_delta_thresh:
                return True, f"energy_rising({tid}:{dE:.3f})"


            # --- cosine absolute ---
            cos_sim = self.prev_cos.get(tid)
            if cos_sim and cos_sim < self.cos_thresh:
                return True, f"low_cos_sim({tid}:{cos_sim:.2f})"

            # --- cosine drift ---
            cos_drop = self.prev_cos_delta.get(tid)
            if cos_drop and cos_drop > self.cos_delta_thresh:
                return True, f"cos_drift({tid}:{cos_drop:.2f})"


        # ------ forced refresh ------
        if frames_since_detect >= self.max_gap:
            return True, f"max_gap({frames_since_detect})"

        # ------ cooldown ------
        if frames_since_detect < self.min_gap:
            return False, f"cooldown({frames_since_detect})"

        return False, "stable"



    # ------------------------------------------------------
    # STATE UPDATES
    # ------------------------------------------------------
    def record_detection(self, tracks):
        self.last_detect_frame = self.frame_idx

        active = {t[4] for t in tracks if t[6]==0}
        self.prev_active_ids = active

        # prune memory of vanished IDs
        self.prev_id_conf   = {tid:self.prev_id_conf.get(tid)   for tid in active if tid in self.prev_id_conf}
        self.prev_energy    = {tid:self.prev_energy.get(tid)    for tid in active if tid in self.prev_energy}
        self.prev_energy_trend = {tid:self.prev_energy_trend.get(tid) for tid in active if tid in self.prev_energy_trend}
        self.prev_emb       = {tid:self.prev_emb.get(tid)       for tid in active if tid in self.prev_emb}
        self.prev_cos       = {tid:self.prev_cos.get(tid)       for tid in active if tid in self.prev_cos}
        self.prev_cos_delta = {tid:self.prev_cos_delta.get(tid) for tid in active if tid in self.prev_cos_delta}

        # update positions for motion
        self.prev_pos = {t[4]:tuple(map(int,t[0:4])) for t in tracks if t[6]==0}


    # ------------------------------------------------------
    # UPDATE HELPERS — call these in your main loop
    # ------------------------------------------------------
    def update_id_confidence(self, tid, new_conf):
        prev = self.prev_id_conf.get(tid)
        if prev is None:
            self.prev_id_conf[tid] = new_conf
        else:
            self.prev_id_conf[tid] = self.id_conf_alpha*new_conf + (1-self.id_conf_alpha)*prev


    def update_id_energy(self, tid, E):
        prevE = self.prev_energy.get(tid, E)
        delta = E - prevE

        smoothE = self.energy_smooth*prevE + (1-self.energy_smooth)*E
        prevDelta = self.prev_energy_trend.get(tid, delta)
        smoothDelta = self.energy_smooth*prevDelta + (1-self.energy_smooth)*delta

        self.prev_energy[tid] = smoothE
        self.prev_energy_trend[tid] = smoothDelta


    def update_id_embedding(self, tid, refined_emb):
        """
        refined_emb: tensor [512] on CPU
        """
        refined_emb = refined_emb / refined_emb.norm()

        prev = self.prev_emb.get(tid)
        if prev is None:
            self.prev_emb[tid] = refined_emb
            return

        # cosine similarity to previous
        cos = torch.dot(refined_emb, prev).item()
        delta = 1.0 - cos  # drop in similarity

        smooth_cos = self.cos_smooth * self.prev_cos.get(tid, cos) + (1-self.cos_smooth)*cos
        smooth_delta = self.cos_smooth * self.prev_cos_delta.get(tid, delta) + (1-self.cos_smooth)*delta

        self.prev_emb[tid] = refined_emb
        self.prev_cos[tid] = smooth_cos
        self.prev_cos_delta[tid] = smooth_delta
