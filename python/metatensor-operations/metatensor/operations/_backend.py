# This file defines the default set of classes to use in metatensor-operations
#
# The code from metatensor-operations can be used in two different modes: either pure
# Python mode or TorchScript mode. In the second case, we need to use a different
# version of the classes above; to do without having to rewrite everything, we re-import
# metatensor-operations in a special context.
#
# See metatensor-torch/metatensor/torch/operations.py for more information.
#
# Any change to this file MUST be also be made to `metatensor/torch/operations.py`.
from typing import Union

import numpy as np

import metatensor


try:
    import torch

    Array = Union[np.ndarray, torch.Tensor]
except ImportError:
    Array = np.ndarray


Labels = metatensor.Labels
TensorBlock = metatensor.TensorBlock
TensorMap = metatensor.TensorMap


def torch_jit_is_scripting():
    return False


def torch_jit_script(function):
    """
    This function takes the place of ``@torch.jit.script`` when running in pure Python
    mode (i.e. when using classes from metatensor-core).

    When used as a decorator (``@torch_jit_script def foo()``), it does nothing to the
    function.
    """
    return function


def torch_jit_annotate(type, value):
    """
    This function takes the place of ``torch.jit.annotate`` when running in pure Python
    mode (i.e. when using classes from metatensor-core).
    """
    return value


check_isinstance = isinstance
