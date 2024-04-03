import numpy as np
import pytest

import metatensor
from metatensor import Labels


torch = pytest.importorskip("torch")

from metatensor.learn.nn import Linear  # noqa: E402

from .utils import random_single_block_no_components_tensor_map  # noqa: E402


@pytest.fixture()
def single_block_tensor_torch():
    """
    random tensor map with no components using torch as array backend
    """
    return random_single_block_no_components_tensor_map(use_torch=True)


@pytest.fixture(scope="module", autouse=True)
def set_random_generator():
    """Set the random generator to same seed before each test is run.
    Otherwise test behaviour is dependend on the order of the tests
    in this file and the number of parameters of the test.
    """
    torch.random.manual_seed(122578741812)


def test_linear_single_block_tensor(single_block_tensor_torch):
    # testing initialization by non sequence arguments
    tensor_module_init_nonseq = Linear(
        in_keys=single_block_tensor_torch.keys,
        in_features=[2],
        out_features=[2],
        bias=[True],
        out_properties=[single_block_tensor_torch[0].properties],
    )
    # testing initialization by sequence arguments
    tensor_module_init_seq = Linear(
        in_keys=single_block_tensor_torch.keys,
        in_features=2,
        out_features=2,
        bias=True,
        out_properties=single_block_tensor_torch[0].properties,
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
        out_tensor = tensor_module(single_block_tensor_torch)

    for i, item in enumerate(single_block_tensor_torch.items()):
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


def test_linear_from_weight(single_block_tensor_torch):
    weights = metatensor.slice(
        single_block_tensor_torch,
        axis="samples",
        labels=Labels(["sample", "system"], np.array([[0, 0], [1, 1]])),
    )
    bias = metatensor.slice(
        single_block_tensor_torch,
        axis="samples",
        labels=Labels(["sample", "system"], np.array([[3, 3]])),
    )
    module = Linear.from_weights(weights, bias)
    module(single_block_tensor_torch)
