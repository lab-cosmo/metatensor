import io

import torch
from packaging import version

import metatensor.torch

from ._data import load_data


def test_dot():
    tensor = load_data("qm7-power-spectrum.npz")
    dot_tensor = metatensor.torch.dot(tensor, metatensor.torch.remove_gradients(tensor))

    # right output type
    assert isinstance(dot_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert dot_tensor._type().name() == "TensorMap"

    # right metadata
    for key in tensor.keys:
        assert dot_tensor.block(key).samples == tensor.block(key).samples
        assert dot_tensor.block(key).properties == tensor.block(key).samples

    # right values
    for key in tensor.keys:
        assert torch.allclose(
            dot_tensor.block(key).values,
            tensor.block(key).values @ tensor.block(key).values.T,
        )


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.dot, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
