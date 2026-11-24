"""Lightweight discovery of packaged Pydantic model presets."""

from importlib.resources import files
from pathlib import Path

CONF_DIR = Path(str(files("geo_eval_harness") / "conf"))


def list_model_configs() -> list[str]:
    """Return portable names accepted by ``--model`` and YAML ``model.name``."""
    model_dir = CONF_DIR / "model"
    return sorted(
        path.relative_to(model_dir).as_posix().removesuffix(".yaml")
        for path in model_dir.rglob("*.yaml")
    )


def model_config_path(name: str) -> Path:
    """Return a catalog-validated preset path, rejecting path traversal."""
    if name not in list_model_configs():
        raise ValueError(f"Unknown model config {name!r}.")
    return CONF_DIR / "model" / f"{name}.yaml"
