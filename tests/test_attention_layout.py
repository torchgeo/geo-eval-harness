import pytest
import torch
import torch.nn.functional as F

from geo_eval_harness.models._thor_encoder import _Attention
from geo_eval_harness.models.terrafm import _TerraFMAttention


@pytest.mark.parametrize("attention_class", [_Attention, _TerraFMAttention])
def test_attention_matches_independent_heads_and_gradients(
    attention_class: type[_Attention] | type[_TerraFMAttention],
) -> None:
    dim, heads = 12, 3
    model = attention_class(dim, heads).double()
    images = torch.randn(2, 5, dim, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(1, heads, 5, 5, dtype=torch.float64)
    actual = model(images, bias) if isinstance(model, _Attention) else model(images)

    head_dim = dim // heads
    outputs = []
    for head in range(heads):
        q, k, v = [
            F.linear(
                images,
                model.qkv.weight[offset + head * head_dim : offset + (head + 1) * head_dim],
                model.qkv.bias[offset + head * head_dim : offset + (head + 1) * head_dim],
            )
            for offset in (0, dim, 2 * dim)
        ]
        scores = (q @ k.transpose(-1, -2)) / head_dim**0.5
        if isinstance(model, _Attention):
            scores = scores + bias[:, head]
        outputs.append(scores.softmax(-1) @ v)
    expected = model.proj(torch.cat(outputs, dim=-1))
    torch.testing.assert_close(actual, expected)
    inputs = (images, *model.parameters())
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
    expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
    for got, want in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(got, want)
