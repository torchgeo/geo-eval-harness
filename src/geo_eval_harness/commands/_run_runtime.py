"""Execute validated image settings without adapting the runtime configuration."""

from ..config.run import RunConfig
from ..errors import UnsupportedNormalizationError
from ..main import main
from ._config import exit_config_error


def run(config: RunConfig) -> None:
    """Pass typed settings to the image runner, which validates the execution device."""
    try:
        main(config, strict=True)
    except UnsupportedNormalizationError as error:  # allow-except: declared normalization errors
        exit_config_error(error)
