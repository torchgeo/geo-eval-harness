# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""CPU-only coverage of Torch device parsing and CUDA capability checks."""

from unittest import mock

import pytest
import torch

from geo_eval_harness.devices import resolve_device


@pytest.mark.parametrize(
    ("requested", "expected"), [("cpu", "cpu"), (torch.device("mps"), "mps:0"), ("mps:0", "mps:0")]
)
def test_non_cuda_devices_never_query_cuda(requested: str | torch.device, expected: str) -> None:
    with (
        mock.patch.object(torch.backends.mps, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "is_available") as available,
        mock.patch.object(torch.cuda, "current_device") as current,
        mock.patch.object(torch.cuda, "device_count") as count,
        mock.patch.object(torch.cuda, "_lazy_init") as init,
    ):
        assert resolve_device(requested) == torch.device(expected)
    for query in (available, current, count, init):
        query.assert_not_called()


def test_explicit_mps_requires_availability() -> None:
    with (
        mock.patch.object(torch.backends.mps, "is_available", return_value=False),
        pytest.raises(ValueError, match="MPS is unavailable"),
    ):
        resolve_device("mps")


@pytest.mark.parametrize(("mps", "expected"), [(True, "mps:0"), (False, "cpu")])
def test_auto_without_cuda_does_not_query_indices(*, mps: bool, expected: str) -> None:
    with (
        mock.patch.object(torch.backends.mps, "is_available", return_value=mps),
        mock.patch.object(torch.cuda, "is_available", return_value=False),
        mock.patch.object(torch.cuda, "current_device") as current,
        mock.patch.object(torch.cuda, "device_count") as count,
    ):
        assert resolve_device("auto") == torch.device(expected)
    current.assert_not_called()
    count.assert_not_called()


@pytest.mark.parametrize(
    ("requested", "expected", "queries_current"),
    [
        ("auto", "cuda:1", True),
        (torch.device("cuda"), "cuda:1", True),
        ("cuda:0", "cuda:0", False),
        (torch.device("cuda:1"), "cuda:1", False),
    ],
)
def test_available_cuda_resolves_current_or_explicit_index(
    requested: str | torch.device, expected: str, *, queries_current: bool
) -> None:
    with (
        mock.patch.object(torch.backends.mps, "is_available") as mps,
        mock.patch.object(torch.cuda, "is_available", return_value=True) as available,
        mock.patch.object(torch.cuda, "current_device", return_value=1) as current,
        mock.patch.object(torch.cuda, "device_count", return_value=2) as count,
    ):
        assert resolve_device(requested) == torch.device(expected)
    mps.assert_not_called()
    available.assert_called_once_with()
    count.assert_called_once_with()
    assert current.call_count == int(queries_current)


@pytest.mark.parametrize("requested", ["cuda", torch.device("cuda:1")])
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


@pytest.mark.parametrize("requested", ["cuda:2", "cuda:127"])
def test_explicit_cuda_index_must_be_in_range(requested: str) -> None:
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "device_count", return_value=2),
        mock.patch.object(torch.cuda, "current_device") as current,
        pytest.raises(ValueError, match=r"CUDA index .*only 2 CUDA devices"),
    ):
        resolve_device(requested)
    current.assert_not_called()


@pytest.mark.parametrize(
    ("requested", "current", "count"), [("auto", -1, 2), ("cuda", 2, 2), ("auto", 0, 0)]
)
def test_current_cuda_index_must_be_in_range(requested: str, current: int, count: int) -> None:
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "current_device", return_value=current),
        mock.patch.object(torch.cuda, "device_count", return_value=count),
        pytest.raises(ValueError, match="CUDA index"),
    ):
        resolve_device(requested)


@pytest.mark.parametrize("requested", ["invalid", "auto:0", "cuda:128", "cuda:256"])
def test_invalid_device_requests_fail_before_cuda_queries(requested: str) -> None:
    with (
        mock.patch.object(torch.cuda, "is_available") as available,
        pytest.raises(ValueError, match="Invalid device request"),
    ):
        resolve_device(requested)
    available.assert_not_called()
