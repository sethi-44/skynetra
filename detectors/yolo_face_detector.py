import torch
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path, device="cuda", conf=0.05):
        self.device = device
        self.conf = conf
        self.model = YOLO(model_path).to(device)

    def detect(self, frame):
        result = self.model(frame, conf=self.conf, verbose=False)[0]
        return result.boxes
