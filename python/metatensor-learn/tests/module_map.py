import pytest

from .utils import TORCH_KWARGS, random_single_block_no_components_tensor_map


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from torch.nn import Module, Sigmoid

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

    @pytest.mark.parametrize(
        "tensor",
        [
            random_single_block_no_components_tensor_map(HAS_TORCH, False),
        ],
    )
    def test_linear_module_init(self, tensor):
        # testing initialization by non sequence arguments
        tensor_module_init_nonseq = Linear(
            in_keys=tensor.keys,
            in_features=[2],
            out_features=[2],
            bias=[False],
            out_properties=[tensor[0].properties],
        )
        # testing initialization by sequence arguments
        tensor_module_init_seq = Linear(
            in_keys=tensor.keys,
            in_features=2,
            out_features=2,
            bias=False,
            out_properties=tensor[0].properties,
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
                tensor_module_init_seq[i].bias == tensor_module_init_nonseq[i].bias
            ), (
                "bias differ when using sequential and non sequential input for"
                " initialization"
            )

        tensor_module = tensor_module_init_nonseq

        with torch.no_grad():
            out_tensor = tensor_module(tensor)

        for i, item in enumerate(tensor.items()):
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
