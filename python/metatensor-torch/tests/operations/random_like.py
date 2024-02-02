import io

import torch
from packaging import version

import metatensor.torch

from .data import load_data


def check_operation(random_uniform_like):
    tensor = load_data("qm7-power-spectrum.npz")
    random_tensor = random_uniform_like(tensor)

    # right output type
    assert isinstance(random_tensor, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert random_tensor._type().name() == "TensorMap"

    # right metadata
    assert metatensor.torch.equal_metadata(random_tensor, tensor)


def test_operation_as_python():
    check_operation(metatensor.torch.random_uniform_like)


def test_operation_as_torch_script():
    scripted = torch.jit.script(metatensor.torch.random_uniform_like)
    check_operation(scripted)


def test_save():
    scripted = torch.jit.script(metatensor.torch.random_uniform_like)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
