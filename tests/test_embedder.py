"""
tests/test_embedder.py

Run with:
    pytest tests/test_embedder.py -v

CUDA-only tests are skipped automatically if no GPU is present.
TensorRT / ONNX Runtime tests are skipped if the libraries are not installed.
"""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helper: build a minimal fake ONNX session so CPU tests never touch disk
# ---------------------------------------------------------------------------

def _make_fake_ort_session(embed_dim: int = 128):
    inp  = MagicMock(); inp.name = "input";  inp.shape = [None, 3, 112, 112]
    out  = MagicMock(); out.name = "output"; out.shape = [None, embed_dim]
    sess = MagicMock()
    sess.get_inputs.return_value  = [inp]
    sess.get_outputs.return_value = [out]

    def _run(output_names, feed):
        x     = feed["input"]                   # (N, 3, 112, 112) float32
        B     = x.shape[0]
        embs  = np.random.randn(B, embed_dim).astype(np.float32)
        # return raw (un-normalised) embeddings; the embedder normalises them
        return [embs]

    sess.run.side_effect = _run
    return sess


# ---------------------------------------------------------------------------
# Preprocessing tests  (no model / GPU needed)
# ---------------------------------------------------------------------------

class TestPreprocessFace:
    def test_output_shape(self):
        from embedder import preprocess_face
        face = np.random.randint(0, 256, (80, 60, 3), dtype=np.uint8)
        out  = preprocess_face(face)
        assert out.shape == (3, 112, 112)

    def test_dtype(self):
        from embedder import preprocess_face
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        out  = preprocess_face(face)
        assert out.dtype == np.float32

    def test_value_range(self):
        from embedder import preprocess_face
        # White pixel (255,255,255) should map to (255-127.5)/128 ≈ 0.996
        face = np.full((112, 112, 3), 255, dtype=np.uint8)
        out  = preprocess_face(face)
        assert out.min() >= -1.01 and out.max() <= 1.01

    def test_black_pixel_is_negative(self):
        from embedder import preprocess_face
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        out  = preprocess_face(face)
        assert (out < 0).all()

    def test_channel_order_chw(self):
        from embedder import preprocess_face
        face = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        out  = preprocess_face(face)
        assert out.shape[0] == 3   # C first


class TestPreprocessBatch:
    def test_empty_raises(self):
        from embedder import preprocess_batch
        with pytest.raises(ValueError, match="empty"):
            preprocess_batch([])

    def test_shape(self):
        from embedder import preprocess_batch
        faces = [np.zeros((50, 40, 3), dtype=np.uint8) for _ in range(4)]
        arr   = preprocess_batch(faces)
        assert arr.shape == (4, 3, 112, 112)

    def test_dtype(self):
        from embedder import preprocess_batch
        faces = [np.zeros((112, 112, 3), dtype=np.uint8)]
        assert preprocess_batch(faces).dtype == np.float32


# ---------------------------------------------------------------------------
# L2-normalisation helper
# ---------------------------------------------------------------------------

class TestL2Normalize:
    def test_1d_unit(self):
        from embedder import _l2_normalize
        x    = torch.tensor([3.0, 4.0])
        normed = _l2_normalize(x)
        assert abs(normed.norm().item() - 1.0) < 1e-5

    def test_2d_rows_unit(self):
        from embedder import _l2_normalize
        x    = torch.randn(5, 128)
        normed = _l2_normalize(x)
        norms = normed.norm(dim=1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)

    def test_zero_vector_stable(self):
        from embedder import _l2_normalize
        x = torch.zeros(64)
        out = _l2_normalize(x)          # should not raise / produce NaN
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# CPU backend (ONNX) — mocked so no real model file is needed
# ---------------------------------------------------------------------------

FAKE_ORT_MODULE = types.ModuleType("onnxruntime")
FAKE_ORT_MODULE.InferenceSession = MagicMock  # replaced per-test


@patch("embedder._ORT_AVAILABLE", True)
class TestCpuBackend:
    """Tests for _CpuBackend with a mocked onnxruntime."""

    def _make_backend(self, embed_dim: int = 128, tmp_path=None):
        import tempfile, pathlib
        if tmp_path is None:
            tmp_path = pathlib.Path(tempfile.mkdtemp())
        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()

        with patch("embedder.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = _make_fake_ort_session(embed_dim)
            from embedder import _CpuBackend
            return _CpuBackend(tmp_path / "model"), embed_dim

    def test_embed_dim_detected(self):
        backend, dim = self._make_backend(128)
        assert backend.embed_dim == 128

    def test_run_output_shape(self):
        backend, dim = self._make_backend(256)
        x   = np.random.randn(3, 3, 112, 112).astype(np.float32)
        out = backend.run(x)
        assert out.shape == (3, 256)

    def test_run_float32_conversion(self):
        """float64 input should be auto-cast to float32."""
        backend, _ = self._make_backend()
        x   = np.random.randn(2, 3, 112, 112).astype(np.float64)
        out = backend.run(x)
        assert out.dtype == np.float32

    def test_missing_onnx_raises(self, tmp_path=None):
        import pathlib, tempfile
        p = pathlib.Path(tempfile.mkdtemp()) / "nonexistent"
        with patch("embedder.ort"):
            from embedder import _CpuBackend
            with pytest.raises(FileNotFoundError):
                _CpuBackend(p)

    def test_ort_unavailable_raises(self):
        with patch("embedder._ORT_AVAILABLE", False):
            from embedder import _CpuBackend
            import pathlib, tempfile
            p = pathlib.Path(tempfile.mkdtemp()) / "model"
            with pytest.raises(ImportError, match="onnxruntime"):
                _CpuBackend(p)


# ---------------------------------------------------------------------------
# MobileFaceNet (CPU mode) — integration-level with mocked backend
# ---------------------------------------------------------------------------

@patch("embedder._ORT_AVAILABLE", True)
class TestMobileFaceNetCPU:

    @staticmethod
    def _build(embed_dim: int = 128, warmup_iters: int = 0):
        import pathlib, tempfile
        tmp = pathlib.Path(tempfile.mkdtemp())
        (tmp / "model.onnx").touch()

        with patch("embedder.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = _make_fake_ort_session(embed_dim)
            from embedder import MobileFaceNet
            return MobileFaceNet(str(tmp / "model"), device="cpu", warmup_iters=warmup_iters)

    def test_repr(self):
        m = self._build()
        assert "cpu" in repr(m)
        assert "MobileFaceNet" in repr(m)

    def test_embed_dim_attribute(self):
        m = self._build(embed_dim=512)
        assert m.embed_dim == 512

    def test_embed_faces_output_count(self):
        m     = self._build()
        faces = [np.zeros((80, 60, 3), dtype=np.uint8) for _ in range(5)]
        tids  = ["a", "b", "c", "d", "e"]
        out   = m.embed_faces(faces, tids)
        assert len(out) == 5

    def test_embed_faces_tids_preserved(self):
        m     = self._build()
        faces = [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(3)]
        tids  = [10, 20, 30]
        out   = m.embed_faces(faces, tids)
        assert [t for t, _ in out] == tids

    def test_embed_faces_embeddings_normalised(self):
        m     = self._build()
        faces = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(4)]
        tids  = list(range(4))
        out   = m.embed_faces(faces, tids)
        for _, emb in out:
            assert abs(emb.norm().item() - 1.0) < 1e-4, "Embedding not unit-norm"

    def test_embed_faces_empty_returns_empty(self):
        m   = self._build()
        out = m.embed_faces([], [])
        assert out == []

    def test_embed_faces_mismatched_lengths_raises(self):
        m     = self._build()
        faces = [np.zeros((112, 112, 3), dtype=np.uint8)]
        with pytest.raises(ValueError, match="same length"):
            m.embed_faces(faces, [1, 2])

    def test_embed_single(self):
        m    = self._build(embed_dim=128)
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        emb  = m.embed_single(face)
        assert emb.shape == (128,)
        assert abs(emb.norm().item() - 1.0) < 1e-4

    def test_similarity_identical(self):
        m    = self._build()
        face = np.zeros((112, 112, 3), dtype=np.uint8)
        emb  = m.embed_single(face)
        sim  = m.similarity(emb, emb)
        assert abs(sim - 1.0) < 1e-4

    def test_similarity_range(self):
        m     = self._build()
        faces = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(2)]
        out   = m.embed_faces(faces, [0, 1])
        sim   = m.similarity(out[0][1], out[1][1])
        assert -1.01 <= sim <= 1.01

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="device"):
            import pathlib, tempfile
            from embedder import MobileFaceNet
            MobileFaceNet("x", device="tpu")

    def test_warmup_runs_without_error(self):
        # warmup_iters>0 should complete without raising
        m = self._build(warmup_iters=2)
        assert m is not None


# ---------------------------------------------------------------------------
# CUDA tests — skipped when no GPU / TRT available
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMobileFaceNetCUDA:
    """Smoke tests for the CUDA/TRT path.  Requires a real .trt engine file
    pointed to by the EMBEDDER_TRT_MODEL env var (skip otherwise)."""

    @pytest.fixture(autouse=True)
    def _check_model(self):
        import os
        self.model_path = os.environ.get("EMBEDDER_TRT_MODEL")
        if not self.model_path:
            pytest.skip("Set EMBEDDER_TRT_MODEL=/path/to/weights/mfn to run CUDA tests")

    def test_load_and_embed(self):
        from embedder import MobileFaceNet
        m     = MobileFaceNet(self.model_path, device="cuda", warmup_iters=1)
        faces = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(4)]
        tids  = list(range(4))
        out   = m.embed_faces(faces, tids)
        assert len(out) == 4
        for _, emb in out:
            assert abs(emb.norm().item() - 1.0) < 1e-3

    def test_large_batch_chunked(self):
        """Batches larger than MAX_BATCH should be chunked transparently."""
        from embedder import MobileFaceNet
        m     = MobileFaceNet(self.model_path, device="cuda", warmup_iters=0)
        n     = m._backend.MAX_BATCH * 2 + 3
        faces = [np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8) for _ in range(n)]
        out   = m.embed_faces(faces, list(range(n)))
        assert len(out) == n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])