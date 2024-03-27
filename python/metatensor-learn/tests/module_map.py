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


try:
    if torch.cuda.is_available():
        HAS_CUDA = True
    else:
        HAS_CUDA = False
except ImportError:
    HAS_CUDA = False


class MockModule(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self._linear = torch.nn.Linear(in_features, out_features)
        self._activation = Sigmoid()
        self._last_layer = torch.nn.Linear(out_features, 1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._last_layer(self._activation(self._linear(input)))


@pytest.fixture(scope="function", autouse=True)
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
    out_tensor = tensor_module(single_block_tensor_torch)

    for i, item in enumerate(single_block_tensor_torch.items()):
        key, block = item
        module = modules[i]
        assert (
            tensor_module.get_module(key) is module
        ), "modules should be initialized in the same order as keys"

        ref_values = module(block.values)
        out_block = out_tensor.block(key)
        assert torch.allclose(ref_values, out_block.values)
        if out_properties is None:
            assert out_block.properties == Labels.range("_", len(out_block.properties))
        else:
            assert out_block.properties == out_properties[0]

        for parameter, gradient in block.gradients():
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


@pytest.mark.parametrize(
    "out_properties", [None, [Labels(["a", "b"], np.array([[1, 1]]))]]
)
@pytest.mark.filterwarnings(
    "ignore:.*If you are using Pytorch and need the labels to also be on meta.*"
)
def test_module_map_meta(single_block_tensor_torch, out_properties):  # noqa F811
    """
    Checks the `to` function of module map by moving the module to cuda and checking
    that the output tensor is on cuda device.
    """
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

    assert tensor_module._in_keys.device == "cpu"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device == "cpu"

    tensor_module.to("meta")

    # at this point, the parameters should have been moved,
    # but the input keys and output properties should still be on cpu
    for parameter in tensor_module.parameters():
        assert parameter.device.type == "meta"

    assert tensor_module._in_keys.device == "cpu"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device == "cpu"

    single_block_tensor_torch = single_block_tensor_torch.to(device="meta")
    out_tensor = tensor_module(single_block_tensor_torch)
    assert out_tensor.device.type == "meta"

    # the input keys and output properties are still on cpu, because
    # metatensor.Labels are never moved to a different device
    assert tensor_module._in_keys.device == "cpu"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device == "cpu"


@pytest.mark.parametrize(
    "out_properties", [None, [Labels(["a", "b"], np.array([[1, 1]]))]]
)
@pytest.mark.skipif(not HAS_CUDA, reason="requires cuda")
@pytest.mark.filterwarnings(
    "ignore:.*If you are using Pytorch and need the labels to also be on cuda.*"
)
def test_module_map_cuda(single_block_tensor_torch, out_properties):  # noqa F811
    """
    Checks the `to` function of module map by moving the module to cuda and checking
    that the output tensor is on cuda device.
    """
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

    assert tensor_module._in_keys.device == "cpu"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device == "cpu"

    tensor_module.to("cuda")

    # at this point, the parameters should have been moved,
    # but the input keys and output properties should still be on cpu
    for parameter in tensor_module.parameters():
        assert parameter.device.type == "cuda"

    assert tensor_module._in_keys.device == "cpu"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device == "cpu"

    single_block_tensor_torch = single_block_tensor_torch.to(device="cuda")
    out_tensor = tensor_module(single_block_tensor_torch)
    assert out_tensor.device.type == "cuda"

    # the input keys and output properties are still on cpu, because
    # metatensor.Labels are never moved to a different device
    assert tensor_module._in_keys.device == "cpu"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device == "cpu"


def test_module_map_dtype(single_block_tensor_torch):
    modules = []
    for key in single_block_tensor_torch.keys:
        modules.append(
            MockModule(
                in_features=len(single_block_tensor_torch.block(key).properties),
                out_features=5,
            )
        )

    tensor_module = ModuleMap(single_block_tensor_torch.keys, modules)
    tensor_module(single_block_tensor_torch)

    tensor_module = tensor_module.to(torch.float64)
    single_block_tensor_torch = single_block_tensor_torch.to(dtype=torch.float64)
    tensor_module(single_block_tensor_torch)
