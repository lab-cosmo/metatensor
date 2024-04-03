import io

import torch
from packaging import version

import metatensor.torch
from metatensor.torch import Labels

from ._data import load_data


def test_drop_blocks():
    # this only runs basic checks functionality checks, and that the code produces
    # output with the right type
    tensor = load_data("qm7-power-spectrum.npz")
    tensor = tensor.keys_to_properties("neighbor_1_type")
    tensor = tensor.keys_to_properties("neighbor_2_type")

    assert tensor.keys == Labels(
        names=["center_type"], values=torch.tensor([[1], [6], [8]])
    )

    keys_to_drop = Labels(names=["center_type"], values=torch.tensor([[1], [8]]))

    tensor = metatensor.torch.drop_blocks(tensor, keys_to_drop)

    # check type
    assert isinstance(tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert tensor._type().name() == "TensorMap"

    # check remaining block
    expected_keys = Labels(names=["center_type"], values=torch.tensor([[6]]))
    assert tensor.keys == expected_keys


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.drop_blocks, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
