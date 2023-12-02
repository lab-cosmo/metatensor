import pytest

from .utils import random_single_block_no_components_tensor_map


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from torch.nn import Module, ModuleDict, Sigmoid

    from metatensor.learn.nn import Linear, ModuleMap


if HAS_TORCH:

    class MockModule(Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self._linear = torch.nn.Linear(in_features, out_features)
            self._activation = Sigmoid()
            self._last_layer = torch.nn.Linear(out_features, 1)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self._last_layer(self._activation(self._linear(input)))


@pytest.mark.skipif(not (HAS_TORCH), reason="requires torch to be run")
class TestModuleMap:
    @pytest.fixture(autouse=True)
    def set_random_generator(self):
        """Set the random generator to same seed before each test is run.
        Otherwise test behaviour is dependend on the order of the tests
        in this file and the number of parameters of the test.
        """
        torch.random.manual_seed(122578741812)

    @pytest.mark.parametrize(
        "tensor",
        [
            random_single_block_no_components_tensor_map(HAS_TORCH, False),
        ],
    )
    def test_module_tensor(self, tensor):
        module_map = ModuleDict()
        for key in tensor.keys:
            module_map[ModuleMap.module_key(key)] = MockModule(
                in_features=len(tensor.block(key).properties), out_features=5
            )
        tensor_module = ModuleMap(module_map)
        with torch.no_grad():
            out_tensor = tensor_module(tensor)

        for key, block in tensor.items():
            module = module_map[Linear.module_key(key)]
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
            random_single_block_no_components_tensor_map(HAS_TORCH, False),
        ],
    )
    def test_linear_module_init(self, tensor):
        tensor_module = Linear(tensor, tensor)
        with torch.no_grad():
            out_tensor = tensor_module(tensor)

        for key, block in tensor.items():
            module = tensor_module.module_map[Linear.module_key(key)]
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
            random_single_block_no_components_tensor_map(HAS_TORCH, False),
        ],
    )
    def test_linear_module_from_module(self, tensor):
        tensor_module = Linear.from_module(
            tensor.keys, in_features=len(tensor[0].properties), out_features=5
        )
        with torch.no_grad():
            out_tensor = tensor_module(tensor)

        for key, block in tensor.items():
            module = tensor_module.module_map[Linear.module_key(key)]
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
