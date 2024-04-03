import io

import torch
from packaging import version

import metatensor.torch

from ._data import load_data


def test_remove_gradients():
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type
    tensor = load_data("qm7-power-spectrum.npz")

    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    assert set(tensor.block(0).gradients_list()) == set(["strain", "positions"])

    tensor = metatensor.torch.remove_gradients(tensor, ["positions"])

    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    assert tensor.block(0).gradients_list() == ["strain"]


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.remove_gradients, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
