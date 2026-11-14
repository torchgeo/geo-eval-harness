"""AID split loading and band selection."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from geo_eval_harness.datasets.aid import AID


@pytest.mark.parametrize(
    ("bands", "expected"),
    [
        (None, [11.0, 22.0, 33.0]),
        (("red", "green", "blue"), [11.0, 22.0, 33.0]),
        (("red",), [11.0]),
        (("blue", "red"), [33.0, 11.0]),
        (("blue", "green", "red"), [33.0, 22.0, 11.0]),
    ],
)
def test_split_and_band_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bands: tuple[str, ...] | None,
    expected: list[float],
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "data" / "aid"
    image_dir = root / "AID" / "airport"
    image_dir.mkdir(parents=True)
    for name in ("train", "test"):
        Image.new("RGB", (8, 8), (11, 22, 33)).save(image_dir / f"{name}.png")
    (root / "aid-train.txt").write_text("train.png\n")
    seen: list[torch.Tensor] = []

    def transform(sample: dict) -> dict:
        seen.append(sample["image"].clone())
        sample["image"] = sample["image"] + 1
        return sample

    dataset = AID().get_dataset("train", bands=bands, transform=transform)
    assert len(dataset) == 1
    sample = dataset[0]
    assert seen[0].shape == (len(expected), 8, 8)
    torch.testing.assert_close(seen[0][:, 0, 0], torch.tensor(expected))
    torch.testing.assert_close(sample["image"], seen[0] + 1)
    assert sample["label"].item() == 0


def test_missing_split_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="download aid"):
        AID().get_dataset("train")
