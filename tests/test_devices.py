# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""CPU-only coverage of Torch device parsing and CUDA capability checks."""

from unittest import mock

import pytest
import torch

from geo_eval_harness.devices import resolve_device


@pytest.mark.parametrize("requested", ["cpu", "cpu:0", "meta", "mps"])
@pytest.mark.parametrize("device_object", [False, True])
def test_non_cuda_devices_never_query_cuda(requested: str, *, device_object: bool) -> None:
    with (
        mock.patch.object(torch.cuda, "is_available") as available,
        mock.patch.object(torch.cuda, "current_device") as current,
        mock.patch.object(torch.cuda, "device_count") as count,
        mock.patch.object(torch.cuda, "_lazy_init") as init,
    ):
        value = torch.device(requested) if device_object else requested
        assert resolve_device(value) == torch.device(requested)
    for query in (available, current, count, init):
        query.assert_not_called()


def test_auto_without_cuda_does_not_query_indices() -> None:
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch.object(torch.cuda, "current_device") as current,
        mock.patch.object(torch.cuda, "device_count") as count,
    ):
        assert resolve_device("auto") == torch.device("cpu")
    current.assert_not_called()
    count.assert_not_called()


@pytest.mark.parametrize(
    ("requested", "expected", "queries_current"),
    [
        ("auto", "cuda:1", True),
        ("cuda", "cuda:1", True),
        (torch.device("cuda"), "cuda:1", True),
        ("cuda:0", "cuda:0", False),
        (torch.device("cuda:0"), "cuda:0", False),
        ("cuda:1", "cuda:1", False),
        (torch.device("cuda:1"), "cuda:1", False),
    ],
)
def test_available_cuda_resolves_current_or_explicit_index(
    requested: str | torch.device, expected: str, *, queries_current: bool
) -> None:
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True) as available,
        mock.patch.object(torch.cuda, "current_device", return_value=1) as current,
        mock.patch.object(torch.cuda, "device_count", return_value=2) as count,
    ):
        assert resolve_device(requested) == torch.device(expected)
    available.assert_called_once_with()
    count.assert_called_once_with()
    assert current.call_count == int(queries_current)


@pytest.mark.parametrize("requested", ["cuda", "cuda:0", "cuda:2", torch.device("cuda:1")])
def test_explicit_cuda_requires_availability(requested: str | torch.device) -> None:
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch.object(torch.cuda, "current_device") as current,
        mock.patch.object(torch.cuda, "device_count") as count,
        pytest.raises(ValueError, match="CUDA is unavailable"),
    ):
        resolve_device(requested)
    current.assert_not_called()
    count.assert_not_called()


@pytest.mark.parametrize("requested", ["cuda:2", torch.device("cuda:2"), "cuda:127"])
def test_explicit_cuda_index_must_be_in_range(requested: str | torch.device) -> None:
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "device_count", return_value=2),
        mock.patch.object(torch.cuda, "current_device") as current,
        pytest.raises(ValueError, match=r"CUDA index .*only 2 CUDA devices"),
    ):
        resolve_device(requested)
    current.assert_not_called()


@pytest.mark.parametrize(("current", "count"), [(-1, 2), (2, 2), (0, 0)])
@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_current_cuda_index_must_be_in_range(requested: str, current: int, count: int) -> None:
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "current_device", return_value=current),
        mock.patch.object(torch.cuda, "device_count", return_value=count),
        pytest.raises(ValueError, match="CUDA index"),
    ):
        resolve_device(requested)


@pytest.mark.parametrize(
    "requested",
    [
        "",
        "invalid",
        "CUDA",
        "auto:0",
        "cpu:-1",
        "cuda:",
        "cuda:-1",
        "cuda:1.5",
        "cuda:abc",
        "cuda:01",
        "cuda:128",
        "cuda:256",
        "cuda:999999999999999999999",
    ],
)
def test_invalid_device_requests_fail_before_cuda_queries(requested: str) -> None:
    with (
        mock.patch.object(torch.cuda, "is_available") as available,
        pytest.raises(ValueError, match="Invalid device request"),
    ):
        resolve_device(requested)
    available.assert_not_called()
