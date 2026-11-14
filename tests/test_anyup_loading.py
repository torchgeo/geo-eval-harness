import hashlib
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from geo_eval_harness.models import anyup


def test_invalid_anyup_checkpoint_is_rejected_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "anyup_multi_backbone.pth"
    path.write_bytes(b"untrusted checkpoint")
    monkeypatch.setattr(anyup, "_checkpoint_path", lambda: path)

    def load_before_verification(*args: Any, **kwargs: Any) -> None:
        pytest.fail("checkpoint loading started before digest verification")

    monkeypatch.setattr(torch.hub, "load", load_before_verification)
    monkeypatch.setattr(torch, "load", load_before_verification)
    with pytest.raises(ValueError, match="SHA-256 checksum mismatch"):
        anyup.load_anyup()


@pytest.mark.parametrize("cached", [False, True])
def test_anyup_loads_only_verified_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cached: bool
) -> None:
    source = tmp_path / "source.pth"
    expected = nn.Linear(2, 2)
    torch.save(expected.state_dict(), source)
    contents = source.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    path = tmp_path / "checkpoints" / "anyup_multi_backbone.pth"
    if cached:
        path.parent.mkdir()
        path.write_bytes(contents)
    monkeypatch.setattr(anyup, "_checkpoint_path", lambda: path)
    monkeypatch.setattr(anyup, "ANYUP_CHECKPOINT_SHA256", digest)

    def download(url: str, destination: str, hash_prefix: str) -> None:
        assert not cached
        assert url == anyup.ANYUP_CHECKPOINT_URL
        assert hash_prefix == digest
        Path(destination).write_bytes(contents)

    def construct(repo: str, entrypoint: str, **kwargs: Any) -> nn.Module:
        assert repo == f"{anyup.ANYUP_REPO}:{anyup.ANYUP_REF}"
        assert entrypoint == anyup.ANYUP_ENTRYPOINT
        assert kwargs["pretrained"] is False
        return nn.Linear(2, 2)

    load = torch.load

    def load_weights(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["weights_only"] is True
        return load(*args, **kwargs)

    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    monkeypatch.setattr(torch.hub, "load", construct)
    monkeypatch.setattr(torch, "load", load_weights)
    actual = anyup.load_anyup()
    torch.testing.assert_close(actual.state_dict(), expected.state_dict())
    assert not actual.training
    assert all(not parameter.requires_grad for parameter in actual.parameters())
