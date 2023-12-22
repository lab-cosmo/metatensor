# This file defines the default set of classes to use in metatensor-learn
#
# The code from metatensor-learn can be used in two different modes: either pure Python
# mode or TorchScript mode. In the second case, we need to use a different version of
# the classes above; to do so without having to rewrite everything, we re-import
# metatensor-learn in a special context.
#
# See metatensor-torch/metatensor/torch/learn.py for more information.
#
# Any change to this file MUST be also be made to `metatensor/torch/learn.py`.

from metatensor import Labels, LabelsEntry, TensorBlock, TensorMap


def torch_jit_is_scripting():
    return False


check_isinstance = isinstance

__all__ = [
    "Labels",
    "LabelsEntry",
    "TensorBlock",
    "TensorMap",
    "torch_jit_is_scripting",
    "check_isinstance",
]
