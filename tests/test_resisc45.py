"""Channel-selection tests for RESISC45, whose upstream loader has no ``bands`` argument.

The upstream class is mocked, so no dataset download is needed.
"""

from pathlib import Path

import pytest
import torch

from geo_eval_harness.datasets.resisc45 import RESISC45


class _FakeRESISC45:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.transforms = kwargs.get("transforms")

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        # Pixel values identify the source channel after selection.
        image = torch.arange(3, dtype=torch.float32).view(3, 1, 1).expand(3, 8, 8).clone()
        sample = {"image": image, "label": torch.tensor(index)}
        return self.transforms(sample) if self.transforms is not None else sample


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> RESISC45:
    import geo_eval_harness.datasets.resisc45 as mod

    monkeypatch.setattr(mod, "TGRESISC45", _FakeRESISC45)
    return RESISC45()


class TestMetadata:
    def test_declared_shape_matches_bands(self) -> None:
        bench = RESISC45()
        assert bench.num_channels == 3
        assert [b.name for b in bench.bands] == ["red", "green", "blue"]
        assert bench.rgb_indices == [0, 1, 2]

    def test_split_sizes_sum_to_the_published_total(self) -> None:
        # 45 classes x 700 images; the 60/20/20 split is torchgeo's.
        assert sum(RESISC45.split_sizes.values()) == 31500

    def test_statistics_are_uint8_scale(self) -> None:
        """Stats must stay in raw sensor units so ``model_native`` detects uint8."""
        from geo_eval_harness.models._input_units import InputUnit, detect_input_unit

        assert detect_input_unit(RESISC45().bands) is InputUnit.UINT8

    def test_uses_aerial_sensor_tag(self) -> None:
        """RESISC45 is tagged ``aerial`` so sensor-routed models accept it."""
        assert {band.sensor for band in RESISC45.bands} == {"aerial"}

    def test_data_root_is_fixed(self) -> None:
        # Compare Paths, not strings: str() is backslash-separated on Windows.
        assert RESISC45.data_root() == Path("data/resisc45")


class TestBandSelection:
    @pytest.mark.parametrize("bands", [None, ("red", "green", "blue")])
    def test_identity_selection_avoids_copying(
        self, patched: RESISC45, bands: tuple[str, ...] | None
    ) -> None:
        """Selecting all bands should avoid a per-sample channel copy."""
        ds = patched.get_dataset("train", bands=bands)
        assert ds.kwargs["transforms"] is None
        torch.testing.assert_close(ds[0]["image"][:, 0, 0], torch.tensor([0.0, 1.0, 2.0]))

    @pytest.mark.parametrize(
        ("bands", "expected"),
        [(("blue", "red"), [2.0, 0.0]), (("blue", "green", "red"), [2.0, 1.0, 0.0])],
    )
    def test_selection_preserves_requested_order(
        self, patched: RESISC45, bands: tuple[str, ...], expected: list[float]
    ) -> None:
        ds = patched.get_dataset("train", bands=bands)
        image = ds[0]["image"]
        assert image.shape == (len(expected), 8, 8)
        torch.testing.assert_close(image[:, 0, 0], torch.tensor(expected))

    def test_unknown_band_is_rejected(self, patched: RESISC45) -> None:
        with pytest.raises(ValueError, match="unknown band 'nir'"):
            patched.get_dataset("train", bands=("nir",))

    def test_unknown_split_is_rejected(self, patched: RESISC45) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            patched.get_dataset("invalid")

    def test_split_is_forwarded_and_partition_ignored(self, patched: RESISC45) -> None:
        ds = patched.get_dataset("test", partition="0.01x_train", bands=None)
        assert ds.kwargs["split"] == "test"
        assert "partition" not in ds.kwargs


class TestTransformComposition:
    def test_selection_runs_before_the_caller_transform(self, patched: RESISC45) -> None:
        seen: list[tuple[int, ...]] = []

        def _resize(sample: dict) -> dict:
            seen.append(tuple(sample["image"].shape))
            return sample

        ds = patched.get_dataset("train", bands=("red",), transform=_resize)
        ds[0]
        assert seen == [(1, 8, 8)]

    def test_caller_transform_survives_when_selection_is_identity(self, patched: RESISC45) -> None:
        calls: list[int] = []

        def _mark(sample: dict) -> dict:
            calls.append(1)
            return sample

        ds = patched.get_dataset("train", bands=None, transform=_mark)
        assert ds.kwargs["transforms"] is _mark
        ds[0]
        assert calls == [1]
