import importlib
import sys

import torch

import metatensor.learn
from metatensor.torch import Labels, LabelsEntry, TensorBlock, TensorMap


# ==================================================================================== #
# see operations.py for an explanation of what's going on here.                        #
# ==================================================================================== #


# Step 1: create the `_classes` module as an empty module
spec = importlib.util.spec_from_loader(
    "metatensor.torch.learn._classes",
    loader=None,
)
module = importlib.util.module_from_spec(spec)
# This module only exposes a handful of things, defined here. Any changes here MUST also
# be made to the `metatensor/learn/_classes.py` file, which is used in non
# TorchScript mode.
module.__dict__["Labels"] = Labels
module.__dict__["LabelsEntry"] = LabelsEntry
module.__dict__["TensorBlock"] = TensorBlock
module.__dict__["TensorMap"] = TensorMap
module.__dict__["torch_jit_is_scripting"] = torch.jit.is_scripting


def check_isinstance(obj, ty):
    if isinstance(ty, torch.ScriptClass):
        # This branch is taken when `ty` is a custom class (TensorMap, …). since `ty` is
        # an instance of `torch.ScriptClass` and not a class itself, there is no way to
        # check if obj is an "instance" of this class, so we always return True and hope
        # for the best. Most errors should be caught by the TorchScript compiler anyway.
        return True
    else:
        assert isinstance(ty, type)
        return isinstance(obj, ty)


module.__dict__["check_isinstance"] = check_isinstance

# register the module in sys.modules, so future import find it directly
sys.modules[spec.name] = module


# Step 2: create a module named `metatensor.torch.learn` (like the one that will be
# created by importing the current file), but using code from `metatensor.learn`
spec = importlib.util.spec_from_file_location(
    # create a module with this name
    "metatensor.torch.learn",
    # using the code from there
    metatensor.learn.__file__,
)

module = importlib.util.module_from_spec(spec)
# override `metatensor.torch.learn` (the module associated with the current file)
# with the newly created module
sys.modules[spec.name] = module
spec.loader.exec_module(module)
