from ultralytics.trackers.byte_tracker import BYTETracker

class TrackerConfig:
    track_high_thresh = 0.6
    track_low_thresh = 0.1
    new_track_thresh = 0.7
    track_buffer = 30
    match_thresh = 0.8
    fuse_score=False

def create_tracker():
    return BYTETracker(TrackerConfig(), frame_rate=30)
