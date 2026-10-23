"""Tests for extract_features in geo_eval_harness.utils."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from geo_eval_harness.utils import extract_features


class _IdentityModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)


def _make_loader(
    n: int = 8, c: int = 3, h: int = 4, *, batch_size: int = 4, multi_label: bool = False
) -> DataLoader:
    rng = torch.Generator().manual_seed(0)
    images = torch.rand(n, c, h, h, generator=rng)
    shape = (n, 5) if multi_label else (n,)
    labels = torch.randint(0, 2 if multi_label else 4, shape, generator=rng)
    dataset = [{"image": images[i], "label": labels[i]} for i in range(n)]
    return DataLoader(dataset, batch_size=batch_size, generator=rng)


@pytest.mark.parametrize("verbose", [False, True])
def test_basic_extraction(*, verbose: bool, capsys: pytest.CaptureFixture[str]) -> None:
    loader = _make_loader()
    model = _IdentityModel()
    X, y = extract_features(
        model, loader, device="cpu", description="Extracting (train)" if verbose else None
    )
    expected_images = torch.stack([sample["image"] for sample in loader.dataset])
    expected_labels = torch.stack([sample["label"] for sample in loader.dataset])
    np.testing.assert_array_equal(X, expected_images.flatten(1).numpy())
    np.testing.assert_array_equal(y, expected_labels.numpy())
    assert ("Extracting (train)" in capsys.readouterr().err) == verbose


def test_extraction_uses_inference_mode_and_restores_grad_state() -> None:
    class _InferenceModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled()
            return x.flatten(1)

    with torch.enable_grad():
        X, y = extract_features(_InferenceModel(), _make_loader(), device="cpu", description=None)
        assert torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()

    assert X.shape == (8, 48)
    assert y.shape == (8,)


@pytest.mark.parametrize("key", ["norm", "global_pool", "head.global_pool"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_dict_outputs_preserve_values_and_singleton_batches(key: str, batch_size: int) -> None:
    class DictModel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
            features = images.flatten(1)
            return {key: features[:, None, :] if key == "head.global_pool" else features}

    loader = _make_loader(n=4, batch_size=batch_size, multi_label=True)
    features, labels = extract_features(DictModel(), loader, "cpu", description=None)
    np.testing.assert_array_equal(
        features, torch.stack([sample["image"] for sample in loader.dataset]).flatten(1).numpy()
    )
    np.testing.assert_array_equal(
        labels, torch.stack([sample["label"] for sample in loader.dataset]).numpy()
    )


def test_vector_output_restores_singleton_batch_axis() -> None:
    class VectorModel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.flatten(1).squeeze(0)

    loader = _make_loader(n=3, batch_size=1)
    features, _ = extract_features(VectorModel(), loader, "cpu", description=None)
    np.testing.assert_array_equal(
        features, torch.stack([sample["image"] for sample in loader.dataset]).flatten(1).numpy()
    )


def test_missing_label_key_raises() -> None:
    images = torch.zeros(4, 3, 4, 4)
    dataset = [{"image": images[i]} for i in range(4)]
    loader = DataLoader(dataset, batch_size=4, generator=torch.Generator().manual_seed(0))
    model = _IdentityModel()
    with pytest.raises(KeyError, match="label"):
        extract_features(model, loader, device="cpu", description=None)


def test_unknown_dict_key_raises() -> None:
    class _BadModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"unknown_key": x.flatten(1)}

    loader = _make_loader()
    with pytest.raises(ValueError, match="Unexpected features"):
        extract_features(_BadModel(), loader, device="cpu", description=None)


def test_transforms_are_applied_before_extraction() -> None:
    loader = _make_loader(c=3)
    model = _IdentityModel()

    def transform(images: torch.Tensor) -> torch.Tensor:
        return images * 2.0

    X, _y = extract_features(model, loader, device="cpu", transforms=transform, description=None)
    expected = torch.stack([sample["image"] for sample in loader.dataset]).flatten(1) * 2.0
    np.testing.assert_array_equal(X, expected.numpy())


def test_3d_output_mean_pooled() -> None:
    class _SeqModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.flatten(2).permute(0, 2, 1)  # (B, HW, C)

    loader = _make_loader(n=4, c=2, h=3)
    X, _y = extract_features(_SeqModel(), loader, device="cpu", description=None)
    expected = torch.stack([sample["image"] for sample in loader.dataset]).mean(dim=(2, 3))
    np.testing.assert_allclose(X, expected.numpy(), rtol=1e-6)
