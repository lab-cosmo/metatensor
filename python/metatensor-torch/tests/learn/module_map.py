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


@pytest.mark.parametrize(
    "out_properties", [None, [Labels(["a", "b"], torch.tensor([[1, 1]]))]]
)
@pytest.mark.skipif(not HAS_CUDA, reason="requires cuda")
def test_module_map_cuda(single_block_tensor, out_properties):  # noqa F811
    """
    We set the correct default device for initialization and check if the module this
    works once the default device has been changed. This catches cases where the default
    device or a hard coded device is used for tensor created within the forward
    function.
    """
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

    assert tensor_module._in_keys.device.type == "cpu"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device.type == "cpu"

    tensor_module.to("cuda")

    # at this point, the parameters should have been moved,
    # but the input keys and output properties should still be on cpu
    for parameter in tensor_module.parameters():
        assert parameter.device.type == "cuda"

    assert tensor_module._in_keys.device.type == "cpu"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device.type == "cpu"

    single_block_tensor = single_block_tensor.to(device="cuda")
    out_tensor = tensor_module(single_block_tensor)
    assert out_tensor.device.type == "cuda"

    # the input keys and output properties should be on cuda
    assert tensor_module._in_keys.device.type == "cuda"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device.type == "cuda"


def test_module_map_dtype(single_block_tensor):
    modules = []
    for key in single_block_tensor.keys:
        modules.append(
            MockModule(
                in_features=len(single_block_tensor.block(key).properties),
                out_features=5,
            )
        )

    tensor_module = ModuleMap(single_block_tensor.keys, modules)
    tensor_module(single_block_tensor)

    tensor_module = tensor_module.to(torch.float64)
    single_block_tensor = single_block_tensor.to(torch.float64)
    tensor_module(single_block_tensor)


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


@pytest.mark.parametrize(
    "out_properties", [None, [Labels(["a", "b"], torch.tensor([[1, 1]]))]]
)
@pytest.mark.skipif(not HAS_CUDA, reason="requires cuda")
def test_torchscript_module_map_cuda(out_properties, single_block_tensor):
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
    tensor_module = torch.jit.script(tensor_module)

    assert tensor_module._in_keys.device.type == "cpu"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device.type == "cpu"

    tensor_module.to("cuda")

    # at this point, the parameters should have been moved,
    # but the input keys and output properties should still be on cpu
    for parameter in tensor_module.parameters():
        assert parameter.device.type == "cuda"

    assert tensor_module._in_keys.device.type == "cpu"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device.type == "cpu"

    single_block_tensor = single_block_tensor.to(device="cuda")
    out_tensor = tensor_module(single_block_tensor)
    assert out_tensor.device.type == "cuda"

    # the input keys and output properties should be on cuda
    assert tensor_module._in_keys.device.type == "cuda"
    if out_properties is not None:
        for label in tensor_module._out_properties:
            assert label.device.type == "cuda"
