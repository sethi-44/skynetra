import torch

class AsyncGPUDetector:
    def __init__(self, face_detector, device):
        self.detector = face_detector
        self.device = device
        self.stream = torch.cuda.Stream()
        self.latest_boxes = None

    def infer(self, frame):
        # Run YOLO in its own CUDA stream
        # print("[DETECTOR] running on GPU stream")

        with torch.cuda.stream(self.stream):
            with torch.no_grad():
                result = self.detector.model(frame[..., ::-1], verbose=False)[0]
                boxes = result.boxes

        # Sync with default stream before returning
        torch.cuda.current_stream().wait_stream(self.stream)

        self.latest_boxes = boxes
        return boxes  # optional
