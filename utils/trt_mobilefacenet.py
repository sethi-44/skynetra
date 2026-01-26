# utils/trt_mobilefacenet.py

import tensorrt as trt
import torch
import os

class TRTMobileFaceNet:
    def __init__(self, engine_path: str, device="cuda"):
        assert device == "cuda"
        assert os.path.exists(engine_path), f"Engine not found: {engine_path}"

        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Tensor names (from trtexec logs)
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

        self.device = device

        self._warmup()

    def _warmup(self):
        dummy = torch.zeros(
            (4, 3, 112, 112),
            device="cuda",
            dtype=torch.float16
        )
        for _ in range(5):
            _ = self.infer(dummy)

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 112, 112) FP16 CUDA tensor
        returns: (B, 256) FP16 CUDA tensor
        """

        assert x.is_cuda
        assert x.dtype == torch.float16

        B = x.shape[0]

        # Set dynamic shape
        self.context.set_input_shape(self.input_name, x.shape)

        # Allocate output
        out = torch.empty(
            (B, 256),
            device="cuda",
            dtype=torch.float16
        )

        # Bind device pointers (TRT 10 style)
        self.context.set_tensor_address(
            self.input_name, x.data_ptr()
        )
        self.context.set_tensor_address(
            self.output_name, out.data_ptr()
        )

        # Execute (ONLY stream handle allowed)
        stream = torch.cuda.current_stream().cuda_stream
        self.context.execute_async_v3(stream)

        return out
