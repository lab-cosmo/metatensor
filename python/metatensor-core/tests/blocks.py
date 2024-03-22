import copy
import warnings

import numpy as np
import pytest
from numpy.testing import assert_equal


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from metatensor import DeviceWarning, Labels, MetatensorError, TensorBlock


@pytest.fixture
def block():
    return TensorBlock(
        values=np.full((3, 2), -1.0),
        samples=Labels(["s"], np.array([[0], [2], [4]])),
        components=[],
        properties=Labels(["p"], np.array([[5], [3]])),
    )


@pytest.fixture
def block_components():
    return TensorBlock(
        values=np.full((3, 3, 2, 2), -1.0),
        samples=Labels(["s"], np.array([[0], [2], [4]])),
        components=[
            Labels(["c_1"], np.array([[-1], [0], [1]])),
            Labels(["c_2"], np.array([[-4], [1]])),
        ],
        properties=Labels(["p"], np.array([[5], [3]])),
    )


def test_constructor_errors():
    values = np.full((3, 3, 2, 2), -1.0)
    samples = Labels(["s"], np.array([[0], [2], [4]]))
    components = [
        Labels(["c_1"], np.array([[-1], [0], [1]])),
        Labels(["c_2"], np.array([[-4], [1]])),
    ]
    properties = Labels(["p"], np.array([[5], [3]]))

    # this works
    _ = TensorBlock(values, samples, components, properties)

    message = "`samples` must be metatensor Labels, not <class 'str'>"
    with pytest.raises(TypeError, match=message):
        TensorBlock(values, "samples", components, properties)

    message = "`components` elements must be metatensor Labels, not <class 'str'>"
    with pytest.raises(TypeError, match=message):
        TensorBlock(values, samples, ["c"], properties)

    message = "`properties` must be metatensor Labels, not <class 'str'>"
    with pytest.raises(TypeError, match=message):
        TensorBlock(values, samples, components, "properties")


def test_gradient_errors(block):
    # missing "sample" column
    gradient = TensorBlock(
        values=np.zeros((0, 2)),
        samples=Labels([], np.empty((0, 2))),
        components=[],
        properties=block.properties,
    )

    message = (
        "invalid parameter: gradients samples must have at least "
        "one dimension, named 'sample', we got none"
    )
    with pytest.raises(MetatensorError, match=message):
        block.add_gradient("g", gradient)

    # negative values for "sample" column
    gradient = TensorBlock(
        values=np.zeros((1, 2)),
        samples=Labels(["sample"], np.array([[-3]])),
        components=[],
        properties=block.properties,
    )

    message = (
        "invalid parameter: invalid value for the 'sample' in gradient samples: "
        "we got -3, but the values contain 3 samples"
    )
    with pytest.raises(MetatensorError, match=message):
        block.add_gradient("g", gradient)

    # values too large for "sample" column
    gradient = TensorBlock(
        values=np.zeros((1, 2)),
        samples=Labels(["sample"], np.array([[42]])),
        components=[],
        properties=block.properties,
    )

    message = (
        "invalid parameter: invalid value for the 'sample' in gradient samples: "
        "we got 42, but the values contain 3 samples"
    )
    with pytest.raises(MetatensorError, match=message):
        block.add_gradient("g", gradient)


def test_repr(block):
    expected = """TensorBlock
    samples (3): ['s']
    components (): []
    properties (2): ['p']
    gradients: None"""
    assert block.__repr__() == expected


def test_repr_zero_samples():
    block = TensorBlock(
        values=np.zeros((0, 2)),
        samples=Labels([], np.empty((0, 2))),
        components=[],
        properties=Labels(["p"], np.array([[5], [3]])),
    )
    expected = """TensorBlock
    samples (0): []
    components (): []
    properties (2): ['p']
    gradients: None"""
    assert block.__repr__() == expected


def test_repr_zero_samples_gradient(block):
    block.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.zeros((0, 2)),
            samples=Labels(["sample"], np.empty((0, 2))),
            components=block.components,
            properties=block.properties,
        ),
    )

    expected_block = """TensorBlock
    samples (3): ['s']
    components (): []
    properties (2): ['p']
    gradients: ['g']"""

    assert block.__repr__() == expected_block

    expected_grad = """Gradient TensorBlock ('g')
    samples (0): ['sample']
    components (): []
    properties (2): ['p']
    gradients: None"""

    gradient = block.gradient("g")
    assert gradient.__repr__() == expected_grad


def test_block_no_components(block):
    assert_equal(block.values, np.full((3, 2), -1.0))

    assert block.samples.names == ["s"]
    assert len(block.samples) == 3
    assert tuple(block.samples[0]) == (0,)
    assert tuple(block.samples[1]) == (2,)
    assert tuple(block.samples[2]) == (4,)

    assert len(block.components) == 0

    assert block.properties.names == ["p"]
    assert len(block.properties) == 2
    assert tuple(block.properties[0]) == (5,)
    assert tuple(block.properties[1]) == (3,)


def test_block_with_components(block_components):
    expected = """TensorBlock
    samples (3): ['s']
    components (3, 2): ['c_1', 'c_2']
    properties (2): ['p']
    gradients: None"""
    assert block_components.__repr__() == expected

    assert_equal(block_components.values, np.full((3, 3, 2, 2), -1.0))

    assert block_components.samples.names == ["s"]
    assert len(block_components.samples) == 3
    assert tuple(block_components.samples[0]) == (0,)
    assert tuple(block_components.samples[1]) == (2,)
    assert tuple(block_components.samples[2]) == (4,)

    assert len(block_components.components) == 2
    component_1 = block_components.components[0]
    assert component_1.names == ["c_1"]
    assert len(component_1) == 3
    assert tuple(component_1[0]) == (-1,)
    assert tuple(component_1[1]) == (0,)
    assert tuple(component_1[2]) == (1,)

    component_2 = block_components.components[1]
    assert component_2.names == ["c_2"]
    assert len(component_2) == 2
    assert tuple(component_2[0]) == (-4,)
    assert tuple(component_2[1]) == (1,)

    assert block_components.properties.names, ["p"]
    assert len(block_components.properties) == 2
    assert tuple(block_components.properties[0]) == (5,)
    assert tuple(block_components.properties[1]) == (3,)


def test_gradients(block_components):
    block_components.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((2, 3, 2, 2), 11.0),
            samples=Labels(["sample", "g"], np.array([[0, -2], [2, 3]])),
            components=block_components.components,
            properties=block_components.properties,
        ),
    )

    expected = """TensorBlock
    samples (3): ['s']
    components (3, 2): ['c_1', 'c_2']
    properties (2): ['p']
    gradients: ['g']"""
    assert block_components.__repr__() == expected

    assert block_components.has_gradient("g")
    assert not block_components.has_gradient("something_else")
    assert not block_components.has_gradient("something else")

    assert block_components.gradients_list() == ["g"]

    gradient = block_components.gradient("g")

    expected_grad = """Gradient TensorBlock ('g')
    samples (2): ['sample', 'g']
    components (3, 2): ['c_1', 'c_2']
    properties (2): ['p']
    gradients: None"""
    assert gradient.__repr__() == expected_grad

    assert gradient.samples.names == ["sample", "g"]
    assert len(gradient.samples) == 2
    assert tuple(gradient.samples[0]) == (0, -2)
    assert tuple(gradient.samples[1]) == (2, 3)

    assert_equal(gradient.values, np.full((2, 3, 2, 2), 11.0))


def test_copy():
    block = TensorBlock(
        values=np.full((3, 3, 2), 2.0),
        samples=Labels(["s"], np.array([[0], [2], [4]])),
        components=[
            Labels(["c_1"], np.array([[-1], [0], [1]])),
        ],
        properties=Labels(["p"], np.array([[5], [3]])),
    )

    # using TensorBlock.copy
    clone = block.copy()
    block_values_id = id(block.values)

    del block

    assert id(clone.values) != block_values_id

    assert_equal(clone.values, np.full((3, 3, 2), 2.0))
    assert clone.samples.names == ["s"]
    assert len(clone.samples) == 3
    assert tuple(clone.samples[0]) == (0,)
    assert tuple(clone.samples[1]) == (2,)
    assert tuple(clone.samples[2]) == (4,)

    # using copy.deepcopy
    other_clone = clone.copy()
    block_values_id = id(clone.values)

    del clone

    assert id(other_clone.values) != block_values_id
    assert_equal(other_clone.values, np.full((3, 3, 2), 2.0))


def test_shallow_copy_error(block):
    msg = "shallow copies of TensorBlock are not possible, use a deepcopy instead"
    with pytest.raises(ValueError, match=msg):
        copy.copy(block)


def test_nested_gradients():
    """
    Test that nested gradients are correctly returned when accessed via their
    relative syntax
    """
    grad_grad = TensorBlock(
        values=np.array([[2.0]]),
        samples=Labels.range("sample", 1),
        components=[],
        properties=Labels.single(),
    )

    grad = TensorBlock(
        values=np.array([[1.0]]),
        samples=Labels.range("sample", 1),
        components=[],
        properties=Labels.single(),
    )
    grad.add_gradient("gradient_of_gradient", grad_grad)

    block = TensorBlock(
        values=np.array([[0.0]]),
        samples=Labels.single(),
        components=[],
        properties=Labels.single(),
    )
    block.add_gradient("gradient", grad)

    assert np.all(block.gradient("gradient").values == np.array([[1.0]]))

    grad_grad_values = (
        block.gradient("gradient").gradient("gradient_of_gradient").values
    )
    assert np.all(grad_grad_values == np.array([[2.0]]))


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch to be run")
def test_different_device():
    message = (
        "Values and labels for this block are on different devices: "
        "labels are always on CPU, and values are on device 'meta'."
    )
    with pytest.warns(DeviceWarning, match=message):
        block = TensorBlock(
            values=torch.tensor([[3.0, 4.0]], device="meta"),
            samples=Labels.range("s", 1),
            components=[],
            properties=Labels.range("p", 2),
        )

    gradient = TensorBlock(
        values=torch.tensor([[3.0, 4.0]], device="cpu"),
        samples=Labels.range("sample", 1),
        components=[],
        properties=Labels.range("p", 2),
    )

    message = "values and the new gradient must be on the same device, got meta and cpu"
    with pytest.raises(ValueError, match=message):
        block.add_gradient("g", gradient)


def test_different_dtype():
    block = TensorBlock(
        values=np.array([[3.0, 4.0]], dtype=np.float64),
        samples=Labels.range("s", 1),
        components=[],
        properties=Labels.range("p", 2),
    )

    message = (
        "values and the new gradient must have the same dtype, got float64 and float32"
    )
    with pytest.raises(ValueError, match=message):
        block.add_gradient(
            "g",
            TensorBlock(
                values=np.array([[3.0, 4.0]], dtype=np.float32),
                samples=Labels.range("sample", 1),
                components=[],
                properties=Labels.range("p", 2),
            ),
        )


def test_to():
    block = TensorBlock(
        values=np.array([[3.0, 4.0]]),
        samples=Labels.range("s", 1),
        components=[],
        properties=Labels.range("p", 2),
    )

    block.add_gradient(
        "g",
        TensorBlock(
            values=np.array([[3.0, 4.0]]),
            samples=Labels.range("sample", 1),
            components=[],
            properties=Labels.range("p", 2),
        ),
    )

    assert block.device == "cpu"
    assert block.dtype == np.float64
    assert block.gradient("g").dtype == np.float64

    converted = block.to(dtype=np.float32)
    assert converted.dtype == np.float32
    assert converted.gradient("g").dtype == np.float32

    # check that the code handles both positional and keyword arguments
    device = "cpu"
    moved = block.to(device, dtype=np.float32)
    moved = block.to(np.float32, device)
    moved = block.to(np.float32, device=device)
    moved = block.to(device, np.float32)

    message = "can not give a device twice in `TensorBlock.to`"
    with pytest.raises(ValueError, match=message):
        moved = block.to("cpu", device="cpu")

    message = "can not give a dtype twice in `TensorBlock.to`"
    with pytest.raises(ValueError, match=message):
        moved = block.to(np.float32, dtype=np.float32)

    # string positional arguments are assumed to be devices
    message = "can not move numpy array to non-cpu device: test"
    with pytest.raises(ValueError, match=message):
        moved = block.to("test")

    message = "unexpected type in `TensorBlock.to`: <class 'numpy.ndarray'>"
    with pytest.raises(TypeError, match=message):
        moved = block.to(np.array([0]))

    if HAS_TORCH:
        block = converted.to(arrays="torch")
        assert isinstance(block.values, torch.Tensor)
        assert isinstance(block.gradient("g").values, torch.Tensor)

        devices = ["meta", torch.device("meta")]
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):
            devices.append("mps")
            devices.append(torch.device("mps"))

        if torch.cuda.is_available():
            devices.append("cuda")
            devices.append("cuda:0")
            devices.append(torch.device("cuda"))

        for device in devices:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                moved = block.to(device=device)

            assert moved.device.type == torch.device(device).type
            assert moved.gradient("g").device.type == torch.device(device).type


def test_to_torch_multiple_args():
    block = TensorBlock(
        values=np.array([[3.0, 4.0]]),
        samples=Labels.range("s", 1),
        components=[],
        properties=Labels.range("p", 2),
    )

    block.add_gradient(
        "g",
        TensorBlock(
            values=np.array([[3.0, 4.0]]),
            samples=Labels.range("sample", 1),
            components=[],
            properties=Labels.range("p", 2),
        ),
    )

    assert block.device == "cpu"
    assert block.dtype == np.float64
    assert block.gradient("g").dtype == np.float64

    if HAS_TORCH:
        block = block.to(arrays="torch", dtype=torch.float32)
        assert isinstance(block.values, torch.Tensor)
        assert isinstance(block.gradient("g").values, torch.Tensor)
        assert block.values.dtype == torch.float32
        assert block.gradient("g").values.dtype == torch.float32
