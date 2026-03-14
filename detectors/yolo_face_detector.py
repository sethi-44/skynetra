import cv2
import torch
import numpy as np
from ultralytics import YOLO

class FaceDetector:
    def __init__(
        self,
        model_path,
        device='cuda',
        imgsz=1280,
        conf=0.45,
        iou=0.6,
        warmup_iters=5
    ):
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device

        # Determine model path based on device
        if device == 'cpu':
            # Use ONNX model for CPU
            full_model_path = model_path + '.onnx'
        else:
            # Use TensorRT for GPU
            full_model_path = model_path + '.engine'

        # Load model
        self.model = YOLO(full_model_path, task="detect")

        # Warmup (CRITICAL for TensorRT)
        self._warmup(warmup_iters)

    def _warmup(self, iters):
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(iters):
            self.model(
                dummy,
                conf=self.conf,
                iou=self.iou,
                verbose=False
            )

    @torch.no_grad()
    def detect(self, frame):
        """
        frame: BGR numpy array (H, W, 3)
        returns: ultralytics Boxes object
        """

        # Ensure contiguous memory (important for TRT)
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        result = self.model(
            frame,
            imgsz=self.imgsz,   # force TRT-compatible size
            conf=self.conf,
            iou=self.iou,
            verbose=False
        )[0]

        return result.boxes
    
    
# -------------------------------
# Helper: Ultralytics Boxes → YOLOX ByteTrack
# -------------------------------

    @staticmethod
    def boxes_to_yolox(boxes):
        """
        Convert Ultralytics Boxes to YOLOX ByteTrack format
        output: np.ndarray (N,5) float64 -> x1,y1,x2,y2,score
        """
        if boxes is None or len(boxes) == 0:
            return (
                np.empty((0, 5), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
            )

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
        conf = boxes.conf.cpu().numpy().astype(np.float64)

        return np.concatenate([xyxy, conf[:, None]], axis=1),conf
