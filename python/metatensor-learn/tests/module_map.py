import numpy as np
import pytest

from metatensor import Labels


torch = pytest.importorskip("torch")
from torch.nn import Module, Sigmoid  # noqa: E402

from metatensor.learn.nn import ModuleMap  # noqa: E402

from .utils import random_single_block_no_components_tensor_map  # noqa: E402


@pytest.fixture
def single_block_tensor_torch():
    """
    random tensor map with no components using torch as array backend
    """
    return random_single_block_no_components_tensor_map(use_torch=True)


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
    Otherwise test behaviour is dependend on the order of the tests
    in this file and the number of parameters of the test.
    """
    torch.random.manual_seed(122578741812)


@pytest.mark.parametrize(
    "out_properties", [None, [Labels(["a", "b"], np.array([[1, 1]]))]]
)
def test_module_map_single_block_tensor(single_block_tensor_torch, out_properties):
    modules = []
    for key in single_block_tensor_torch.keys:
        modules.append(
            MockModule(
                in_features=len(single_block_tensor_torch.block(key).properties),
                out_features=5,
            )
        )

    tensor_module = ModuleMap(
        single_block_tensor_torch.keys, modules, out_properties=out_properties
    )
    with torch.no_grad():
        out_tensor = tensor_module(single_block_tensor_torch)

    for i, item in enumerate(single_block_tensor_torch.items()):
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
