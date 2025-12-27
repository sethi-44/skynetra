import cv2
import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
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
    from detectors.async_gpu_detector import AsyncGPUDetector
    base_detector = FaceDetector("models/yolov8n-face-lindevs.pt", device)
    face_detector = AsyncGPUDetector(base_detector, device)

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
    # Configure asynchronous detection only on CPU to avoid CUDA context issues
    executor = ThreadPoolExecutor(max_workers=1) 
    pending = None
    latest_boxes = None


    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ End of video or cannot open file.")
            break
        # 1) COLLECT if ready
        if pending and pending.done():
            try:
                latest_boxes = pending.result()   # <-- USE returned value
            except Exception as e:
                print("⚠️ async detector error:", e)
                latest_boxes = None
            pending = None

        # 2) DECIDE if we schedule new inference
        if sampler.should_run_detector(tracks) or len(tracker.tracked_stracks)==0:
            if pending is None:
                pending = executor.submit(face_detector.infer, frame.copy())

        # 3) CHOOSE BOXES FOR TRACKING
        boxes = latest_boxes if latest_boxes is not None else empty_boxes(device)

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
