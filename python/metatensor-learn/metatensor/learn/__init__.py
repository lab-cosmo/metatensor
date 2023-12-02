import importlib.metadata


__version__ = importlib.metadata.version("metatensor-learn")

from . import nn

__all__ = ["nn"]
