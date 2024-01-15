import importlib.metadata


__version__ = importlib.metadata.version("metatensor-learn")

try:
    import torch
    HAS_TORCH = True
except ImportError:

    HAS_TORCH = False


if HAS_TORCH:
    from . import nn
    __all__ = ["nn"]

__all__ = []
