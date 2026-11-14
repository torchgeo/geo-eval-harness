"""CPU/CUDA parity for :class:`KNNClassifier`; requires CUDA and GPU-enabled FAISS."""

import numpy as np
import pytest
import torch

from geo_eval_harness.knn import KNNClassifier, gpu_faiss_available

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not gpu_faiss_available(),
    reason="requires torch CUDA and GPU-enabled FAISS for cpu/cuda parity",
)

N_TRAIN, N_TEST, DIM, K, N_CLASSES = 2000, 500, 64, 5, 10


@pytest.fixture
def features() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((N_TRAIN, DIM)).astype(np.float32)
    x_test = rng.standard_normal((N_TEST, DIM)).astype(np.float32)
    return x_train, x_test


def test_singlelabel_parity(features: tuple[np.ndarray, np.ndarray]) -> None:
    x_train, x_test = features
    rng = np.random.default_rng(1)
    y_train = rng.integers(0, N_CLASSES, size=N_TRAIN).astype(np.int64)

    cpu = KNNClassifier(n_neighbors=K, device="cpu").fit(x_train, y_train)
    cu = KNNClassifier(n_neighbors=K, device="cuda").fit(x_train, y_train)

    assert (cpu.predict(x_test) == cu.predict(x_test)).all()

    pp_cpu = cpu.predict_proba(x_test)
    pp_cu = cu.predict_proba(x_test)
    assert pp_cpu.shape == pp_cu.shape
    np.testing.assert_allclose(pp_cpu, pp_cu, atol=1e-5)


def test_multilabel_parity(features: tuple[np.ndarray, np.ndarray]) -> None:
    x_train, x_test = features
    rng = np.random.default_rng(2)
    y_train = (rng.random((N_TRAIN, N_CLASSES)) > 0.7).astype(np.int64)

    cpu = KNNClassifier(n_neighbors=K, device="cpu").fit(x_train, y_train)
    cu = KNNClassifier(n_neighbors=K, device="cuda").fit(x_train, y_train)

    p_cpu = cpu.predict(x_test)
    p_cu = cu.predict(x_test)
    assert p_cpu.shape == p_cu.shape
    assert (p_cpu == p_cu).all()

    np.testing.assert_allclose(cpu.predict_proba(x_test), cu.predict_proba(x_test), atol=1e-5)


@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires a nondefault CUDA device")
@pytest.mark.parametrize("multilabel", [False, True])
def test_nondefault_gpu_uses_its_own_stream(*, multilabel: bool) -> None:
    rng = np.random.default_rng(3)
    train = rng.normal(size=(20000, 378)).astype(np.float32)
    query = rng.normal(size=(4000, 378)).astype(np.float32)
    labels = (
        (rng.random((len(train), 19)) > 0.7).astype(np.int64)
        if multilabel
        else rng.integers(0, 19, len(train))
    )
    cpu = KNNClassifier(5, "cpu").fit(train, labels)
    with torch.cuda.device(0):
        gpu = KNNClassifier(5, "cuda:1").fit(train, labels)
        assert torch.cuda.current_device() == 0
        np.testing.assert_array_equal(gpu.predict(query), cpu.predict(query))
        np.testing.assert_allclose(gpu.predict_proba(query), cpu.predict_proba(query), atol=1e-5)
        assert torch.cuda.current_device() == 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires a nondefault CUDA device")
def test_unnumbered_cuda_agrees_with_faiss_default_device(
    features: tuple[np.ndarray, np.ndarray],
) -> None:
    train, query = features
    labels = np.random.default_rng(1).integers(0, N_CLASSES, len(train))
    expected = KNNClassifier(K, "cpu").fit(train, labels).predict(query)
    with torch.cuda.device(1):
        model = KNNClassifier(K, "cuda").fit(train, labels)
        assert model.device == "cuda:0"
        np.testing.assert_array_equal(model.predict(query), expected)
        assert torch.cuda.current_device() == 1
