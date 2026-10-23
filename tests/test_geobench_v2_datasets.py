"""Tests for the high-level get_datasets API for GeoBench V2 datasets."""

from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from geo_eval_harness.datasets import get_bench_dataset_class, get_datasets
from geo_eval_harness.datasets.burn_scars import BurnScars
from geo_eval_harness.datasets.pastis import PASTIS
from tests.support.data import require_dataset_data


class MockV2Dataset:
    """Stand-in for ``geobench_v2.datasets.GeoBench<X>`` upstream classes."""

    def __init__(
        self,
        root: Path,
        split: str,
        transforms: Callable | None = None,
        band_order: dict[str, list[str]] | list[str] | None = None,
        **kwargs: object,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.band_order = band_order

        if isinstance(band_order, dict):
            self.c = sum(len(v) for v in band_order.values())
        elif band_order is not None:
            self.c = len(band_order)
        else:
            self.c = 3
        self.h, self.w = 32, 32

    def __len__(self) -> int:
        return 10

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        channels = torch.arange(1, self.c + 1, dtype=torch.float32) * 1000
        image = channels[:, None, None].expand(self.c, self.h, self.w).clone()
        sample = {"image": image}
        if self.root.name == "burn_scars":
            sample["mask"] = torch.arange(self.h * self.w).reshape(self.h, self.w) % 2
        else:
            sample["label"] = torch.tensor(idx % 2)
        return self.transforms(sample) if self.transforms is not None else sample


@pytest.fixture
def mock_v2_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    upstream = {}
    for dataset, name in {"benv2": "GeoBenchBENV2", "burn_scars": "GeoBenchBurnScars"}.items():
        upstream[dataset] = MagicMock(side_effect=MockV2Dataset)
        monkeypatch.setattr(f"geobench_v2.datasets.{name}", upstream[dataset])
    return upstream


class TestV2Loading:
    @pytest.mark.parametrize(
        ("dataset_name", "download"),
        [
            ("m-eurosat", "geobench_v1 --datasets m-eurosat"),
            ("burn_scars", "geobench_v2 --datasets burn_scars"),
            ("eurosat", "eurosat"),
            ("resisc45", "resisc45"),
        ],
    )
    def test_missing_data_requires_download(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        dataset_name: str,
        download: str,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with (
            patch("huggingface_hub.snapshot_download") as snapshot,
            patch("geobench_v2.datasets.base.download_url") as download_url,
            pytest.raises(FileNotFoundError) as error,
        ):
            get_datasets(dataset_name=dataset_name, num_workers=0)

        assert f"geo-eval-harness download {download}" in str(error.value)
        snapshot.assert_not_called()
        download_url.assert_not_called()

    def test_benv2_classification(self, mock_v2_env: dict[str, MagicMock]) -> None:
        ds, train_dl, _, _ = get_datasets(
            dataset_name="benv2",
            return_val=True,
            batch_size=4,
            num_workers=0,
        )

        assert isinstance(train_dl, DataLoader)
        assert len(ds) == 10
        assert get_bench_dataset_class("benv2").task == "classification"

        batch = next(iter(train_dl))
        assert batch["image"].shape == (4, 3, 32, 32)
        assert batch["label"].shape == (4,)
        assert batch["label"].dtype == torch.long
        torch.testing.assert_close(
            batch["image"][0, :, 0, 0], torch.tensor([1000.0, 2000.0, 3000.0])
        )

    def test_burn_scars_segmentation(self, mock_v2_env: dict[str, MagicMock]) -> None:
        _, train_dl, _ = get_datasets(
            dataset_name="burn_scars",
            batch_size=2,
            return_val=False,
            num_workers=0,
        )

        assert get_bench_dataset_class("burn_scars").task == "segmentation"
        batch = next(iter(train_dl))
        assert batch["image"].shape == (2, 3, 32, 32)
        assert batch["mask"].shape == (2, 32, 32)
        assert batch["mask"].dtype == torch.long

    def test_partition_warning(self, mock_v2_env: dict[str, MagicMock]) -> None:
        with pytest.warns(UserWarning, match="does not support custom partitions"):
            get_datasets(
                dataset_name="benv2",
                partition_name="0.10x_train",
                num_workers=0,
            )

    @pytest.mark.parametrize("dataset_name", ["benv2", "burn_scars"])
    def test_resize_preserves_raw_images_and_categorical_masks(
        self, mock_v2_env: dict[str, MagicMock], dataset_name: str
    ) -> None:
        target = 64
        ds, _, _ = get_datasets(
            dataset_name=dataset_name,
            image_size=target,
            batch_size=4,
            num_workers=0,
        )

        sample = ds[0]
        assert sample["image"].shape == (3, target, target)
        expected = torch.tensor([1000.0, 2000.0, 3000.0])[:, None, None].expand(3, target, target)
        torch.testing.assert_close(sample["image"], expected)
        if dataset_name == "burn_scars":
            source_mask = torch.arange(32 * 32).reshape(32, 32) % 2
            expected_mask = source_mask.repeat_interleave(2, 0).repeat_interleave(2, 1)
            torch.testing.assert_close(sample["mask"], expected_mask)

    def test_bad_dataset_name(self) -> None:
        with pytest.raises(KeyError, match="Unknown dataset 'phantom_dataset'"):
            get_datasets(dataset_name="phantom_dataset")

    @pytest.mark.parametrize(
        ("dataset_name", "expected_bands"),
        [
            ("benv2", {"s2": ["B04", "B03", "B02"]}),
            ("burn_scars", ["B04", "B03", "B02"]),
        ],
    )
    def test_upstream_receives_fixed_root_splits_and_sensor_band_order(
        self,
        mock_v2_env: dict[str, MagicMock],
        dataset_name: str,
        expected_bands: dict[str, list[str]] | list[str],
    ) -> None:
        get_datasets(dataset_name=dataset_name, bands="rgb", batch_size=2, num_workers=0)
        calls = mock_v2_env[dataset_name].call_args_list
        assert [call.kwargs["split"] for call in calls] == ["train", "validation", "test"]
        for call in calls:
            kwargs = call.kwargs
            assert Path(kwargs["root"]) == Path("data/geobenchv2") / dataset_name
            assert kwargs["band_order"] == expected_bands
            assert kwargs["download"] is False
            assert kwargs["data_normalizer"] is torch.nn.Identity
            if dataset_name == "benv2":
                assert kwargs["return_stacked_image"] is True


class MockKuroSiwo:
    """Match upstream Kuro Siwo samples, with each image shaped ``(C, H, W)``.

    SAR keys follow ``time_step``; DEM has its own ``image_dem`` key.
    """

    def __init__(  # noqa: PLR0913 - matches the upstream dataset constructor.
        self,
        root: Path,
        split: str,
        *,
        band_order: dict[str, list[str]] | None = None,
        time_step: tuple[str, ...] = ("pre_1", "pre_2", "post"),
        transforms: Callable | None = None,
        return_stacked_image: bool = False,
        **kwargs: object,
    ) -> None:
        self.root = root
        self.split = split
        self.band_order = band_order or {}
        self.time_step = list(time_step)
        self.transforms = transforms
        self.return_stacked_image = return_stacked_image
        self.h, self.w = 16, 16

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {
            "invalid_data": torch.ones(1, self.h, self.w, dtype=torch.long),
            "mask": torch.zeros(self.h, self.w, dtype=torch.long),
        }
        if "sar" in self.band_order:
            sar_c = len(self.band_order["sar"])
            for ts in self.time_step:
                sample[f"image_{ts}"] = torch.ones(sar_c, self.h, self.w) * float(
                    {"pre_1": 1.0, "pre_2": 2.0, "post": 3.0}[ts]
                )
        if "dem" in self.band_order:
            sample["image_dem"] = torch.full((len(self.band_order["dem"]), self.h, self.w), 99.0)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


class TestKuroSiwoCanonicalization:
    @pytest.fixture
    def mocked_kuro_siwo(self) -> Iterator[MagicMock]:
        with patch(
            "geobench_v2.datasets.GeoBenchKuroSiwo",
            MagicMock(side_effect=MockKuroSiwo),
        ) as mocked:
            yield mocked

    @pytest.mark.parametrize(
        ("bands", "expected_values"),
        [
            (("vv", "vh"), [3.0, 3.0]),
            (("vv",), [3.0]),
            (("dem",), [99.0]),
            (("vv", "dem"), [3.0, 99.0]),
            (None, [3.0, 3.0, 99.0]),
        ],
    )
    def test_image_combines_post_event_sar_and_dem_in_order(
        self,
        mocked_kuro_siwo: MagicMock,
        bands: tuple[str, ...] | None,
        expected_values: list[float],
    ) -> None:
        bench = get_bench_dataset_class("kuro_siwo")()
        ds = bench.get_dataset("train", bands=bands)
        sample = ds[0]
        img = sample["image"]
        torch.testing.assert_close(
            img, torch.tensor(expected_values)[:, None, None].expand(len(expected_values), 16, 16)
        )

        for stale in ("image_pre_1", "image_pre_2", "image_post", "image_dem"):
            assert stale not in sample, f"per-modality key {stale!r} should be folded into 'image'"

        mocked_kuro_siwo.assert_called_once()
        assert mocked_kuro_siwo.call_args.kwargs["time_step"] == ["post"]
        assert mocked_kuro_siwo.call_args.kwargs["return_stacked_image"] is False

    def test_resize_runs_after_canonicalization(self, mocked_kuro_siwo: MagicMock) -> None:
        """Resize needs a single image tensor, not separate modality keys."""
        _, train_dl, _ = get_datasets(
            dataset_name="kuro_siwo",
            bands=("vv", "vh"),
            image_size=32,
            batch_size=2,
            num_workers=0,
        )
        batch = next(iter(train_dl))
        assert batch["image"].shape[-2:] == (32, 32)
        assert batch["mask"].shape[-2:] == (32, 32)
        assert [call.kwargs["split"] for call in mocked_kuro_siwo.call_args_list] == [
            "train",
            "val",
            "test",
        ]


@pytest.mark.slow
class TestKuroSiwoLive:
    """Smoke tests against locally supplied Kuro Siwo data."""

    @pytest.mark.parametrize(
        ("bands", "expected_channels"),
        [
            (("vv", "vh"), 2),
            (None, 3),  # all bands: vv, vh, dem
            (("vv", "dem"), 2),
        ],
    )
    def test_real_sample_is_3d(self, bands: tuple[str, ...] | None, expected_channels: int) -> None:
        require_dataset_data("kuro_siwo")
        bench = get_bench_dataset_class("kuro_siwo")()
        ds = bench.get_dataset("train", bands=bands)
        sample = ds[0]
        img = sample["image"]
        assert img.dim() == 3, f"expected 3-D, got shape {tuple(img.shape)}"
        assert img.shape[0] == expected_channels, (
            f"expected {expected_channels} channels, got {img.shape[0]}"
        )


class MockPASTIS:
    """Stand-in for ``geobench_v2.datasets.GeoBenchPASTIS``.

    Mirrors the real upstream loader: per-sensor ``image_<sensor>`` keys, folded
    into a single stacked ``image`` only when ``return_stacked_image`` is set.
    """

    def __init__(
        self,
        root,
        split,
        *,
        band_order=None,
        transforms=None,
        return_stacked_image=False,
        **kwargs,
    ):
        del kwargs
        self.root = root
        self.split = split
        self.band_order = band_order or {}
        self.transforms = transforms
        self.return_stacked_image = return_stacked_image
        self.h, self.w = 16, 16

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        del idx
        sample: dict[str, torch.Tensor] = {"mask": torch.zeros(self.h, self.w, dtype=torch.long)}
        for sensor, bands in self.band_order.items():
            sample[f"image_{sensor}"] = torch.ones(len(bands), self.h, self.w)
        if self.return_stacked_image:
            sample["image"] = torch.cat(
                [sample.pop(f"image_{sensor}") for sensor in self.band_order], 0
            )
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


class TestPASTISSampleConstruction:
    """PASTIS must build samples through the shared ``_V2Dataset`` path."""

    @pytest.fixture
    def mocked_pastis(self):
        with patch(
            "geobench_v2.datasets.GeoBenchPASTIS",
            MagicMock(side_effect=MockPASTIS),
        ) as mocked:
            yield mocked

    @pytest.mark.parametrize(
        ("bands", "expected_channels"), [(("b04", "b03", "b02"), 3), (None, 16)]
    )
    def test_image_is_stacked_and_3d(self, mocked_pastis, bands, expected_channels):
        bench = get_bench_dataset_class("pastis")()
        sample = bench.get_dataset("train", bands=bands)[0]
        assert "image" in sample
        assert sample["image"].shape == (expected_channels, 16, 16)
        assert not [k for k in sample if k.startswith("image_")]
        assert mocked_pastis.called

    def test_resize_transform_reaches_the_image(self, mocked_pastis):
        """A framework transform must see a canonical ``image``, not ``image_s2``."""
        del mocked_pastis
        _, train_dl, _, _ = get_datasets(
            dataset_name="pastis",
            return_val=True,
            batch_size=2,
            num_workers=0,
            image_size=8,
            bands="rgb",
        )
        batch = next(iter(train_dl))
        assert batch["image"].shape == (2, 3, 8, 8)

    def test_time_steps_requests_a_time_series(self, mocked_pastis):
        PASTIS().get_dataset("train", bands=("b04", "b03", "b02"), time_steps=4)
        kwargs = mocked_pastis.call_args.kwargs
        assert kwargs["num_time_steps"] == 4
        assert kwargs["temporal_output_format"] == "TCHW"

    def test_time_steps_rejected_when_not_multi_temporal(self):
        with pytest.raises(ValueError, match="not multi-temporal"):
            BurnScars().get_dataset("train", time_steps=2)
