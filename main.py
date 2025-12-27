import cv2
import torch
import torch.nn as nn

from models.inception_resnet_v1 import InceptionResnetV1
from detectors.yolo_face_detector import FaceDetector
from trackers.byte_tracker_wrapper import create_tracker
from utils.sampling import FrameSampler
from utils.face_utils import crop_face, empty_boxes
from utils.embedding_ops import normalize_embeddings, mean_pool_embeddings, identify
from utils.visualize import draw_tracks



def run():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load Models ---
    face_detector = FaceDetector("models/yolov8n-face-lindevs.pt", device)
    tracker = create_tracker()
    sampler = FrameSampler()

    resnet = InceptionResnetV1(
        pretrained=None,
        classify=False,
        device=device,
    )

    # add classifier layer so pretrained weights load cleanly
    resnet.logits = nn.Linear(512, 8631)

    state_dict = torch.load("models/facenet_20180402_114759_vggface2.pth", map_location=device)
    resnet.load_state_dict(state_dict)  # strict=True works now

    resnet.eval().to(device)

    # known identities (placeholder)
    known_identities = {"Person_A": torch.randn(512), "Person_B": torch.randn(512)}
    for k in known_identities:
        known_identities[k] = known_identities[k] / known_identities[k].norm()

    identity_memory = {}
    identity_memory_pooled = {}
    identity_labels = {}

    # --- Video Processing ---
    # cap = cv2.VideoCapture(r"C:\Users\harsh\Downloads\6518202-uhd_2160_3840_24fps.mp4")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    min_samples = 5

    print("🚀 Skynetra pipeline running... press ESC to quit")
    cv2.namedWindow("Skynetra Tracking", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Skynetra Tracking", cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    tracks = []



    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of video or cannot open file.")
            break

        # Detection
        if sampler.should_run_detector(tracks):
            boxes = face_detector.detect(frame)
            sampler.record_detection(tracks)
        else:
            boxes = empty_boxes(device)

        # Tracker
        if boxes.data.is_cuda:
            boxes.data = boxes.data.cpu()




        tracks = tracker.update(boxes)

        faces, tids = [], []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for t in tracks:
            if t[6] > 0:
                continue
            face = crop_face(frame_rgb, t[0:4])
            if face is not None:
                faces.append(face)
                tids.append(t[4])

        if faces:
            faces = torch.stack(faces).to(device)
            with torch.no_grad():
                embs = normalize_embeddings(resnet(faces))

            for tid, emb in zip(tids, embs):
                buf = identity_memory.setdefault(tid, [])
                buf.append(emb.cpu())
                if len(buf) > min_samples:
                    buf.pop(0)

                if tid not in identity_memory_pooled and len(buf) >= min_samples:
                    pooled = mean_pool_embeddings(buf)
                    identity_memory_pooled[tid] = pooled
                    name, score = identify(pooled, known_identities)
                    identity_labels[tid] = f"{name} ({score:.2f})"
        # print(
        #     "detections:", 0 if boxes is None else len(boxes),
        #     "| tracks:", len(tracks)
        # )
            

        frame = draw_tracks(frame, tracks, identity_labels)
        cv2.imshow("Skynetra Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("👋 Exiting loop (ESC pressed)")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🧹 Clean shutdown, bye.")


if __name__ == "__main__":
    run()
