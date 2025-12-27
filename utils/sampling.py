import numpy as np
class FrameSampler:
    def __init__(self, min_gap=2,max_gap=15,conf_thresh=0.5,motion_thresh=25):
        self.frame_idx = 0
        self.last_detect_frame=-1
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.conf_thresh = conf_thresh
        self.motion_thresh = motion_thresh
        self.prev_pos={}
        self.prev_active_ids = set()

    def should_run_detector(self,tracks):
        self.frame_idx += 1
        active_ids = {t[4] for t in tracks if t[6] == 0}
        if len(tracks)==0: 
            return True
        if active_ids != self.prev_active_ids:
            return True
        avg_conf=np.mean([t[5] for t in tracks if t[6]==0]) if len(active_ids)>0 else 0
        if avg_conf < self.conf_thresh:
            return True
        for t in tracks:
            tid = t[4]
            x1, y1, x2, y2 = map(int, t[0:4])

            if tid in self.prev_pos:
                px1, py1, px2, py2 = self.prev_pos[tid]
                # Manhattan distance as motion measure
                motion = abs(x1 - px1) + abs(y1 - py1)
                if motion > self.motion_thresh:
                    return True
        if (self.frame_idx - self.last_detect_frame) >= self.max_gap:
            return True
        if (self.frame_idx - self.last_detect_frame) < self.min_gap:
            return False
        return False
    def record_detection(self,tracks):
        self.last_detect_frame = self.frame_idx
        self.prev_active_ids = {t[4] for t in tracks if t[6] == 0}
        self.prev_pos = {}
        for t in tracks:
            tid = t[4]
            x1, y1, x2, y2 = map(int, t[0:4])
            self.prev_pos[tid] = (x1, y1, x2, y2)    