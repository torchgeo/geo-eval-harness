"""Device validation must remain useful when GPU FAISS is not installed."""

import numpy as np
import pytest

from geo_eval_harness import knn


def test_missing_gpu_faiss_is_reported_before_cuda_initialization(monkeypatch):
    monkeypatch.setattr(knn, "gpu_faiss_available", lambda: False)

    def unexpected_context(_device):
        raise AssertionError("CUDA must not initialize before the backend check")

    monkeypatch.setattr(knn.torch.cuda, "device", unexpected_context)
    with pytest.raises(RuntimeError, match="GPU-enabled FAISS is unavailable"):
        knn.KNNClassifier(device="cuda:1").fit(np.ones((2, 3), dtype=np.float32), np.array([0, 1]))
