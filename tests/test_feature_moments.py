import pytest
import torch

from geo_eval_harness.models.segmentation_heads import FeatureStandardizer, feature_moments


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_moments_preserve_noncontiguous_inputs_and_gradients(dtype: torch.dtype) -> None:
    features = torch.randint(-8, 9, (2, 3, 5, 7)).to(dtype) + 10_000
    features[:, 0] = 10_000
    features = features.transpose(2, 3).requires_grad_()
    original = features.clone()
    total, sq_total, count = feature_moments(features)
    reference = features.double().transpose(0, 1).reshape(3, -1)
    expected_total = reference.sum(dim=1)
    expected_sq_total = reference.square().sum(dim=1)
    assert total.dtype == sq_total.dtype == torch.float64
    assert count == 70
    torch.testing.assert_close(total, expected_total, rtol=0, atol=0)
    torch.testing.assert_close(sq_total, expected_sq_total, rtol=0, atol=0)
    torch.testing.assert_close(features, original, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(total.sum() + sq_total.sum(), features)[0]
    expected_grad = torch.autograd.grad(expected_total.sum() + expected_sq_total.sum(), features)[0]
    torch.testing.assert_close(actual_grad, expected_grad)

    standardizer = FeatureStandardizer([3]).double()
    standardizer.set_moments([total], [sq_total], [count])
    mean, std = standardizer.stats(0)
    torch.testing.assert_close(mean, reference.mean(dim=1))
    torch.testing.assert_close(std, (reference.var(dim=1, correction=0) + standardizer.eps).sqrt())
