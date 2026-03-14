import torch
from detectors.yolo_face_detector import FaceDetector
from trackers.byte_tracker_wrapper import create_tracker
from utils.identities.store import IdentityStore
from sampler.sampling import FrameSampler
from embedder.embedder import MobileFaceNet
from utils.main_helpers import setup_video_source, setup_identity_store, process_frame, cleanup 

def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    EMB_DIM = 256  # MobileFaceNet embedding dimension

    sampler=FrameSampler()
    detector = FaceDetector(model_path=r"C:\Users\harsh\OneDrive\Desktop\skynetra\models\yolov9t-face-lindevs", device=device)
    tracker = create_tracker()
    embedder = MobileFaceNet(model_path=r"C:\Users\harsh\OneDrive\Desktop\skynetra\models\mobilefacenet_fp16", device=device)

    cap, video_writer = setup_video_source()

    store = IdentityStore.from_path("identities", device=device)

    id_names, gallery, hop = setup_identity_store(store, device, EMB_DIM)

    identity_memory = {}
    identity_memory_pooled = {}
    track_info = {}
    MIN_SAMPLES = 10

    print("Running SkyNetra (ESC to quit)")

    while True:
        if not process_frame(cap, sampler,detector, tracker, embedder, identity_memory, identity_memory_pooled, track_info, id_names, gallery, hop, device, EMB_DIM, MIN_SAMPLES, video_writer):
            break

    cleanup(cap, video_writer)    


if __name__ == "__main__":
    main()
