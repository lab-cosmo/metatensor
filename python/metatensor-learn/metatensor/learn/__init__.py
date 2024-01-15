import importlib.metadata


__version__ = importlib.metadata.version("metatensor-learn")

try:
    import torch  # noqa

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    from . import data, nn  # noqa

    __all__ = ["data", "nn"]

__all__ = []
