"""Shared schema-version requirements across all command configurations."""

import pytest
from pydantic import ValidationError

from geo_eval_harness.config.flops import FlopsConfig
from geo_eval_harness.config.profile import ProfileConfig
from geo_eval_harness.config.run import RunConfig
from geo_eval_harness.config.schema import InputConfig, StrictModel
from geo_eval_harness.coordbench.config import CoordConfig


@pytest.mark.parametrize(
    ("config_class", "required"),
    [
        (RunConfig, {"model": {"name": "rcf"}, "datasets": ["m-eurosat"]}),
        (ProfileConfig, {"model": {"name": "rcf"}, "dataset": "m-eurosat"}),
        (FlopsConfig, {"model": {"name": "rcf"}}),
        (CoordConfig, {"model": {"name": "sincos"}}),
    ],
)
class TestSchemaVersions:
    @pytest.mark.parametrize("version", [True, False, 1.0, "1", None, 0, 2])
    def test_invalid_version_is_rejected(
        self, config_class: type[StrictModel], required: dict[str, object], version: object
    ) -> None:
        with pytest.raises(ValidationError, match="schema_version") as error:
            config_class.model_validate({**required, "schema_version": version})
        assert error.value.errors()[0]["loc"] == ("schema_version",)

    @pytest.mark.parametrize("explicit", [False, True])
    def test_integer_version_and_default_are_preserved(
        self, config_class: type[StrictModel], required: dict[str, object], *, explicit: bool
    ) -> None:
        values = {**required, **({"schema_version": 1} if explicit else {})}
        config = config_class.model_validate(values)
        version = config.model_dump()["schema_version"]
        assert type(version) is int
        assert version == 1

    def test_json_schema_declares_integer_one(
        self, config_class: type[StrictModel], required: dict[str, object]
    ) -> None:
        schema = config_class.model_json_schema()
        version = schema["properties"]["schema_version"]
        assert version["default"] == 1
        if "$ref" in version:
            version = schema["$defs"][version["$ref"].rsplit("/", 1)[1]]
        assert version["const"] == 1
        assert version["type"] == "integer"


def test_nested_sections_do_not_gain_a_schema_version() -> None:
    assert "schema_version" not in InputConfig.model_fields
    with pytest.raises(ValidationError, match="schema_version"):
        InputConfig.model_validate({"schema_version": 1})
