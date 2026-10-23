"""Unset-aware YAML serialization and the coordinate command's full output."""

import pytest
import yaml

from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.config.profile import ProfileConfig
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import InputConfig, StrictModel
from geo_eval_harness.coordbench.config import CoordConfig


@pytest.mark.parametrize(
    ("config_class", "values"),
    [
        (
            RunConfig,
            {
                "model": {"name": "rcf"},
                "datasets": ["m-eurosat"],
                "input": {"image_size": None},
                "classification": {"linear": {"refit_train_val": False}},
                "segmentation": {"layers": []},
                "output": {"file": None, "resume": False},
            },
        ),
        (
            ProfileConfig,
            {
                "model": {"name": "rcf"},
                "dataset": "m-eurosat",
                "input": {"image_size": None},
                "count_flops": False,
                "warmup": 0,
            },
        ),
        (
            FlopsConfig,
            {
                "model": {"name": "rcf"},
                "segmentation": {"heads": [], "band_configs": []},
                "output": {"resume": False},
            },
        ),
    ],
)
def test_explicit_values_survive_yaml_round_trip(
    config_class: type[StrictModel], values: dict[str, object]
) -> None:
    config = config_class.model_validate(values)
    output = config.model_dump_yaml()
    assert output == values
    restored = config_class.model_validate(yaml.safe_load(yaml.safe_dump(output)))
    assert restored == config
    assert restored.model_fields_set == config.model_fields_set
    assert restored.model_dump_yaml() == values


def test_nested_sections_share_unset_aware_serialization() -> None:
    assert InputConfig().model_dump_yaml() == {}
    assert InputConfig(image_size=None).model_dump_yaml() == {"image_size": None}


def test_coordinate_yaml_keeps_effective_defaults() -> None:
    config = CoordConfig.model_validate({"model": {"name": "sincos"}})
    output = config.model_dump_yaml()
    assert output == config.model_dump(mode="json")
    assert output["schema_version"] == 1
    assert output["runtime"]["device"] == "cpu"
    assert output["evaluation"]["methods"] == ["knn", "linear"]
    assert CoordConfig.model_validate(yaml.safe_load(yaml.safe_dump(output))) == config
