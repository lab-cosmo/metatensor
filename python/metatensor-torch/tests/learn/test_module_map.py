import pytest
import torch
from torch.nn import Module, Sigmoid

from metatensor.torch import allclose_raise
from metatensor.torch.learn.nn import Linear, ModuleMap

from .utils import TORCH_KWARGS, random_single_block_no_components_tensor_map


class MockModule(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self._linear = torch.nn.Linear(in_features, out_features)
        self._activation = Sigmoid()
        self._last_layer = torch.nn.Linear(out_features, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._last_layer(self._activation(self._linear(input)))


class TestModuleMap:
    @pytest.fixture(autouse=True)
    def set_random_generator(self):
        """Set the random generator to same seed before each test is run.
        Otherwise test behaviour is dependend on the order of the tests
        in this file and the number of parameters of the test.
        """
        torch.random.manual_seed(122578741812)
        torch.set_default_device(TORCH_KWARGS["device"])
        torch.set_default_dtype(TORCH_KWARGS["dtype"])

    @pytest.mark.parametrize(
        "tensor",
        [
            random_single_block_no_components_tensor_map(True, True),
        ],
    )
    def test_module_tensor(self, tensor):
        modules = []
        for key in tensor.keys:
            modules.append(
                MockModule(
                    in_features=len(tensor.block(key).properties), out_features=5
                )
            )

        tensor_module = ModuleMap(tensor.keys, modules)
        with torch.no_grad():
            out_tensor = tensor_module(tensor)

        for i, item in enumerate(tensor.items()):
            key, block = item
            module = modules[i]
            assert (
                tensor_module.get_module(key) is module
            ), "modules should be initialized in the same order as keys"

            with torch.no_grad():
                ref_values = module(block.values)
            out_block = out_tensor.block(key)
            assert torch.allclose(ref_values, out_block.values)

            for parameter, gradient in block.gradients():
                with torch.no_grad():
                    ref_gradient_values = module(gradient.values)
                assert torch.allclose(
                    ref_gradient_values, out_block.gradient(parameter).values
                )

    @pytest.mark.parametrize(
        "tensor",
        [
            random_single_block_no_components_tensor_map(True, True),
        ],
    )
    def test_linear_module_init(self, tensor):
        tensor_module = Linear(tensor, tensor)
        with torch.no_grad():
            out_tensor = tensor_module(tensor)

        for i, item in enumerate(tensor.items()):
            key, block = item
            module = tensor_module.modules[i]
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

    @pytest.mark.parametrize(
        "tensor",
        [
            random_single_block_no_components_tensor_map(True, True),
        ],
    )
    def test_linear_module_from_module(self, tensor):
        tensor_module = Linear.from_module(
            tensor.keys, in_features=len(tensor[0].properties), out_features=5
        )
        with torch.no_grad():
            out_tensor = tensor_module(tensor)

        for i, item in enumerate(tensor.items()):
            key, block = item
            module = tensor_module.modules[i]
            assert (
                tensor_module.get_module(key) is module
            ), "modules should be initialized in the same order as keys"

            with torch.no_grad():
                ref_values = module(block.values)
            out_block = out_tensor.block(key)
            assert torch.allclose(ref_values, out_block.values)

            for parameter, gradient in block.gradients():
                with torch.no_grad():
                    ref_gradient_values = module(gradient.values)
                assert torch.allclose(
                    ref_gradient_values, out_block.gradient(parameter).values
                )

    @pytest.mark.parametrize(
        "tensor",
        [
            random_single_block_no_components_tensor_map(True, True),
        ],
    )
    def test_torchscript_module_tensor(self, tensor):
        modules = []
        for key in tensor.keys:
            modules.append(
                MockModule(
                    in_features=len(tensor.block(key).properties), out_features=5
                )
            )
        tensor_module = ModuleMap(tensor.keys, modules)
        ref_tensor = tensor_module(tensor)

        tensor_module_script = torch.jit.script(tensor_module)
        out_tensor = tensor_module_script(tensor)

        allclose_raise(ref_tensor, out_tensor)

        # tests if member functions work that do not appear in forward
        tensor_module_script.get_module(tensor.keys[0])

    @pytest.mark.parametrize(
        "tensor",
        [
            random_single_block_no_components_tensor_map(True, True),
        ],
    )
    def test_torchscript_linear_module(self, tensor):
        tensor_module = Linear.from_module(
            tensor.keys, in_features=len(tensor[0].properties), out_features=5
        )
        ref_tensor = tensor_module(tensor)

        tensor_module_script = torch.jit.script(tensor_module)
        out_tensor = tensor_module_script(tensor)

        allclose_raise(ref_tensor, out_tensor)

        # tests if member functions work that do not appear in forward
        tensor_module_script.get_module(tensor.keys[0])
