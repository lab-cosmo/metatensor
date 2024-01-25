import pytest

from .utils import TORCH_KWARGS, random_single_block_no_components_tensor_map


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from torch.nn import Module, Sigmoid

    from metatensor.learn.nn import ModuleMap


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
        torch.set_default_device(TORCH_KWARGS["device"])
        torch.set_default_dtype(TORCH_KWARGS["dtype"])

    @pytest.mark.parametrize(
        "tensor",
        [
            random_single_block_no_components_tensor_map(HAS_TORCH, False),
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

