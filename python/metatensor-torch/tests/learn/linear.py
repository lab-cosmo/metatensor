import io

import pytest
import torch

import metatensor.torch
from metatensor.torch import Labels, allclose_raise
from metatensor.torch.learn.nn import Linear

from .utils import random_single_block_no_components_tensor_map


@pytest.fixture
def single_block_tensor():
    return random_single_block_no_components_tensor_map()


@pytest.fixture(scope="module", autouse=True)
def set_random_generator():
    """
    Set the random generator to same seed before each test is run. Otherwise test
    behavior is dependent on the order of the tests in this file and the number of
    parameters of the test.
    """
    torch.random.manual_seed(122578741812)


def test_linear_single_block_tensor(single_block_tensor):
    # testing initialization by non sequence arguments
    tensor_module_init_nonseq = Linear(
        in_keys=single_block_tensor.keys,
        in_features=[2],
        out_features=[2],
        bias=[True],
        out_properties=[single_block_tensor[0].properties],
    )
    # testing initialization by sequence arguments
    tensor_module_init_seq = Linear(
        in_keys=single_block_tensor.keys,
        in_features=2,
        out_features=2,
        bias=True,
        out_properties=single_block_tensor[0].properties,
    )
    for i in range(len(tensor_module_init_seq)):
        assert (
            tensor_module_init_seq[i].in_features
            == tensor_module_init_nonseq[i].in_features
        ), (
            "in_features differ when using sequential and non sequential input for"
            " initialization"
        )
        assert (
            tensor_module_init_seq[i].out_features
            == tensor_module_init_nonseq[i].out_features
        ), (
            "out_features differ when using sequential and non sequential input for"
            " initialization"
        )
        assert (
            tensor_module_init_seq[i].bias.shape
            == tensor_module_init_nonseq[i].bias.shape
        ), (
            "bias differ when using sequential and non sequential input for"
            " initialization"
        )

    tensor_module = tensor_module_init_nonseq

    with torch.no_grad():
        out_tensor = tensor_module(single_block_tensor)

    for i, item in enumerate(single_block_tensor.items()):
        key, block = item
        module = tensor_module[i]
        assert (
            tensor_module.get_module(key) is module
        ), "modules should be initialized in the same order as keys"

        with torch.no_grad():
            ref_values = module(block.values)
        out_block = out_tensor.block(key)
        assert torch.allclose(ref_values, out_block.values)
        assert block.properties == out_block.properties

        for parameter, gradient in block.gradients():
            with torch.no_grad():
                ref_gradient_values = module(gradient.values)
            out_gradient = out_block.gradient(parameter)
            assert torch.allclose(ref_gradient_values, out_gradient.values)
            assert gradient.properties == out_gradient.properties


def test_linear_from_weight(single_block_tensor):
    weights = metatensor.torch.slice(
        single_block_tensor,
        axis="samples",
        labels=Labels(["sample"], torch.tensor([[0], [1]])),
    )
    bias = metatensor.torch.slice(
        single_block_tensor,
        axis="samples",
        labels=Labels(["sample"], torch.tensor([[3]])),
    )
    module = Linear.from_weights(weights, bias)
    module(single_block_tensor)


def test_torchscript_linear(single_block_tensor):
    tensor_module = Linear(
        single_block_tensor.keys,
        in_features=len(single_block_tensor[0].properties),
        out_features=5,
    )
    ref_tensor = tensor_module(single_block_tensor)

    tensor_module_script = torch.jit.script(tensor_module)
    out_tensor = tensor_module_script(single_block_tensor)

    allclose_raise(ref_tensor, out_tensor)

    # tests if member functions work that do not appear in forward
    tensor_module_script.get_module(single_block_tensor.keys[0])

    # test save load
    scripted = torch.jit.script(tensor_module_script)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
