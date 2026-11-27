"""Offline smoke test for the embedding download sheets."""

import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import visualize_embeddings as viz


class FakeBench:
    rgb_bands = ("red", "green", "blue")
    class_names = ("oak", "pine", "beech")

    def get_dataset(self, split: str, bands: tuple[str, ...]) -> "FakeBench":
        return self

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        image = np.random.default_rng(index).random((3, 8, 8), dtype=np.float32)
        return {"image": image, "label": np.array([1, 0, 1])}


def test_gallery_writes_sheets_and_marks_missing_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, output = tmp_path / "downloads", tmp_path / "gallery"
    rng = np.random.default_rng(0)
    for product in ("aef", "tessera"):
        for index in (3, 9, 12, 15):
            if product == "tessera" and index == 9:
                continue
            embedding = rng.normal(size=(5, 4, 4)).astype(np.float16)
            embedding[:, 0, 0] = np.nan
            path = root / product / "treesatai" / "test" / f"{index:06d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, embedding=embedding, label=np.array([1, 0, 1]), year=2020)
    monkeypatch.setattr(viz, "get_bench_dataset_class", lambda name: FakeBench)
    argv = ["viz", "--root", str(root), "--output-dir", str(output), "--datasets", "treesatai"]
    monkeypatch.setattr(sys, "argv", argv)

    viz.main()

    assert [sheet.name for sheet in sorted(output.glob("*.png"))] == [
        "treesatai_000.png",
        "treesatai_001.png",
    ]
    embedding = viz.load(root / "aef" / "treesatai" / "test" / "000009.npz")
    row = viz.sample_row(FakeBench()[9], 9, [embedding, None], ["aef", "tessera"])
    assert row["caption"] == "000009: aef 94%, tessera not extracted"
