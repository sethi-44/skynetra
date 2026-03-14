import tensorrt as trt
import torch
import numpy as np
import cv2

try:
    import onnxruntime as ort
    ONNX_RUNTIME_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ONNX Runtime not available: {e}")
    print("CPU mode will not work. Please install onnxruntime: pip install onnxruntime")
    ONNX_RUNTIME_AVAILABLE = False

from pathlib import Path


class MobileFaceNet:

    MAX_BATCH = 8

    def __init__(self, model_path: str, device: str = "cuda"):

        self.device = device
        model_path = Path(model_path)

        if device == "cpu":
            if not ONNX_RUNTIME_AVAILABLE:
                raise ImportError(
                    "ONNX Runtime is required for CPU mode but not available. "
                    "Please install it with: pip install onnxruntime"
                )

            model_path = model_path.with_suffix(".onnx")

            self.session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"]
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            output_shape = self.session.get_outputs()[0].shape
            self.embed_dim = output_shape[-1] if output_shape[-1] > 0 else 256

        else:

            model_path = model_path.with_suffix(".trt")

            self.logger = trt.Logger(trt.Logger.ERROR)
            self.runtime = trt.Runtime(self.logger)

            with open(model_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()

            self.input_name = None
            self.output_name = None

            for i in range(self.engine.num_io_tensors):

                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)

                if mode == trt.TensorIOMode.INPUT:
                    self.input_name = name
                else:
                    self.output_name = name

            assert self.input_name is not None
            assert self.output_name is not None

            shape = self.engine.get_tensor_shape(self.output_name)

            assert shape[-1] > 0
            self.embed_dim = shape[-1]

        self._warmup()

    # -------------------------------------------------------------
    # Warmup
    # -------------------------------------------------------------

    def _warmup(self):

        if self.device == "cpu":

            dummy = np.zeros(
                (self.MAX_BATCH, 3, 112, 112),  # Use MAX_BATCH for ONNX warmup
                dtype=np.float32
            )

            for _ in range(5):
                self.infer(dummy)

        else:

            dummy = torch.zeros(
                (self.MAX_BATCH, 3, 112, 112),
                device="cuda",
                dtype=torch.float16
            )

            for _ in range(5):
                self.infer(dummy)

    # -------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------

    @torch.no_grad()
    def infer(self, x):

        if self.device == "cpu":

            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()

            if x.dtype != np.float32:
                x = x.astype(np.float32)

            outputs = self.session.run(
                [self.output_name],
                {self.input_name: x}
            )

            return outputs[0]

        else:

            assert isinstance(x, torch.Tensor)
            assert x.is_cuda
            assert x.dtype == torch.float16
            assert x.shape[0] <= self.MAX_BATCH

            B = x.shape[0]

            self.context.set_input_shape(
                self.input_name,
                tuple(x.shape)
            )

            out = torch.empty(
                (B, self.embed_dim),
                device="cuda",
                dtype=torch.float16
            )

            self.context.set_tensor_address(
                self.input_name,
                x.data_ptr()
            )

            self.context.set_tensor_address(
                self.output_name,
                out.data_ptr()
            )

            stream = torch.cuda.current_stream()

            self.context.execute_async_v3(stream.cuda_stream)

            # Important for deterministic embeddings
            stream.synchronize()

            return out

    # -------------------------------------------------------------
    # Batch embedding
    # -------------------------------------------------------------

    @torch.no_grad()
    def embed_faces(self, faces: list, tids: list):

        outputs = []

        if self.device == "cpu":
            # ONNX: Process faces in batches for better performance
            faces_array = np.stack(faces).astype(np.float32)
            
            embs = self.infer(faces_array)
            
            # L2 normalize each embedding
            for i, (tid, emb) in enumerate(zip(tids, embs)):
                emb_tensor = torch.from_numpy(emb)
                norm = emb_tensor.norm()
                norm = max(norm, 1e-6)
                emb_tensor = emb_tensor / norm
                outputs.append((tid, emb_tensor))

        else:

            faces_tensor = torch.tensor(
                np.stack(faces),
                device="cuda",
                dtype=torch.float16
            )

            for i in range(0, faces_tensor.shape[0], self.MAX_BATCH):

                batch = faces_tensor[i:i + self.MAX_BATCH]
                batch_tids = tids[i:i + self.MAX_BATCH]

                embs = self.infer(batch).float()

                embs = embs / embs.norm(
                    dim=1,
                    keepdim=True
                ).clamp(min=1e-6)

                embs_cpu = embs.cpu()

                outputs.extend(zip(batch_tids, embs_cpu))

        return outputs

    # -------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------

    @staticmethod
    def preprocess_face(face_bgr: np.ndarray) -> np.ndarray:

        face_rgb = cv2.cvtColor(
            face_bgr,
            cv2.COLOR_BGR2RGB
        )

        face = cv2.resize(face_rgb, (112, 112))

        face = face.astype(np.float32)

        face = (face - 127.5) / 128.0

        face = np.transpose(face, (2, 0, 1))

        return face