"""Execute validated image settings without adapting the runtime configuration."""

from ..config.run import RunConfig
from ..errors import DeviceError, MissingDatasetError, UnsupportedNormalizationError
from ..main import main
from ._config import exit_config_error


def run(config: RunConfig) -> None:
    """Run the image benchmark, reporting an unusable device or missing data without a traceback."""
    try:
        main(config, strict=True)
    except UnsupportedNormalizationError as error:  # allow-except: declared normalization errors
        exit_config_error(error)
    except (
        DeviceError,
        MissingDatasetError,
    ) as error:  # allow-except: report unusable devices and datasets that are not downloaded
        raise SystemExit(f"error: {error}") from error
