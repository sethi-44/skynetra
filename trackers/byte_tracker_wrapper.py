from trackers.byte_tracker import BYTETracker

class ByteTrackConfig:
    track_thresh = 0.35
    track_buffer = 45
    match_thresh = 0.75
    mot20 = False

def create_tracker():
    return BYTETracker(ByteTrackConfig(), frame_rate=30)