import io

import pytest
import torch
from torch.nn import Module, Sigmoid

from metatensor.torch import Labels, allclose_raise
from metatensor.torch.learn.nn import ModuleMap

from .utils import random_single_block_no_components_tensor_map


@pytest.fixture
def single_block_tensor():
    return random_single_block_no_components_tensor_map()


class MockModule(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self._linear = torch.nn.Linear(in_features, out_features)
        self._activation = Sigmoid()
        self._last_layer = torch.nn.Linear(out_features, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._last_layer(self._activation(self._linear(input)))


@pytest.fixture(scope="module", autouse=True)
def set_random_generator():
    """Set the random generator to same seed before each test is run.
    Otherwise test behaviour is dependent on the order of the tests
    in this file and the number of parameters of the test.
    """
    torch.random.manual_seed(122578741812)


@pytest.mark.parametrize(
    "out_properties", [None, [Labels(["a", "b"], torch.tensor([[1, 1]]))]]
)
def test_module_map_single_block_tensor(single_block_tensor, out_properties):
    modules = []
    for key in single_block_tensor.keys:
        modules.append(
            MockModule(
                in_features=len(single_block_tensor.block(key).properties),
                out_features=5,
            )
        )

    tensor_module = ModuleMap(
        single_block_tensor.keys, modules, out_properties=out_properties
    )
    with torch.no_grad():
        out_tensor = tensor_module(single_block_tensor)

    for i, item in enumerate(single_block_tensor.items()):
        key, block = item
        module = modules[i]
        assert (
            tensor_module.get_module(key) is module
        ), "modules should be initialized in the same order as keys"

        with torch.no_grad():
            ref_values = module(block.values)
        out_block = out_tensor.block(key)
        assert torch.allclose(ref_values, out_block.values)
        if out_properties is None:
            assert out_block.properties == Labels.range("_", len(out_block.properties))
        else:
            assert out_block.properties == out_properties[0]

        for parameter, gradient in block.gradients():
            with torch.no_grad():
                ref_gradient_values = module(gradient.values)
            assert torch.allclose(
                ref_gradient_values, out_block.gradient(parameter).values
            )
            if out_properties is None:
                assert out_block.gradient(parameter).properties == Labels.range(
                    "_", len(out_block.gradient(parameter).properties)
                )
            else:
                assert out_block.gradient(parameter).properties == out_properties[0]


def test_torchscript_module_map(single_block_tensor):
    modules = []
    for key in single_block_tensor.keys:
        modules.append(
            MockModule(
                in_features=len(single_block_tensor.block(key).properties),
                out_features=5,
            )
        )
    tensor_module = ModuleMap(single_block_tensor.keys, modules)
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
