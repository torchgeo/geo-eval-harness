"""Installation guidance for optional model dependencies."""

from importlib import import_module

TERRATORCH_INSTALL = (
    "install it from source as described at "
    "https://torchgeo.org/geo-eval-harness/user/installation.html#terratorch-models"
)


def require_package(package: str, install: str) -> None:
    """Import an optional dependency, naming how to install it when it is absent.

    Args:
        package: Importable top-level package name.
        install: Installation instruction appended to the error.

    Raises:
        ModuleNotFoundError: If the package is not installed.
    """
    try:
        import_module(package)
    except ModuleNotFoundError as error:  # allow-except: add installation guidance
        if error.name != package:
            raise
        raise ModuleNotFoundError(
            f"This model requires {package}: {install}", name=package
        ) from error
