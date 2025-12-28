import cv2
import numpy as np


class GlobalMotionEstimator:
    def __init__(self, max_corners=200, quality=0.01, min_distance=7):
        self.prev_gray = None
        self.prev_pts = None

        # feature detection params
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality,
            minDistance=min_distance,
            blockSize=7
        )

        # optical flow params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def compute_global_motion(self, frame):
        """
        Returns average pixel displacement caused by camera motion.
        Higher value => camera shake / movement.
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # first frame: just initialize
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return 0.0

        # no points to track → reinitialize
        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            self.prev_gray = gray
            return 0.0

        # track features between frames
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        # valid tracked points only
        good_prev = self.prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]

        if len(good_prev) == 0:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
            return 0.0

        # displacement vector magnitude
        diffs = good_next - good_prev
        motion = np.sqrt((diffs[:, 0] ** 2) + (diffs[:, 1] ** 2))
        global_motion = float(np.median(motion))  # median is robust to outliers

        # update memory
        self.prev_gray = gray
        self.prev_pts = good_next.reshape(-1, 1, 2)

        return global_motion
