import copy
import sys
import warnings

import numpy as np
import pytest
from numpy.testing import assert_equal

from metatensor import DeviceWarning, Labels, MetatensorError, TensorBlock, TensorMap

from . import utils


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def tensor():
    return utils.tensor()


@pytest.fixture
def large_tensor():
    return utils.large_tensor()


def test_constructor_errors():
    block = TensorBlock(
        values=np.full((3, 1, 1), 1.0),
        samples=Labels(["s"], np.array([[0], [2], [4]])),
        components=[Labels(["c"], np.array([[0]]))],
        properties=Labels(["p"], np.array([[0]])),
    )
    keys = Labels.single()

    # this works
    _ = TensorMap(keys, [block])

    message = "`keys` must be metatensor Labels, not <class 'str'>"
    with pytest.raises(TypeError, match=message):
        TensorMap("keys", [block])

    message = "`blocks` elements must be metatensor TensorBlock, not <class 'str'>"
    with pytest.raises(TypeError, match=message):
        TensorMap(keys, ["block"])


def test_copy():
    # Do not use a fixture here because we want exactly on reference in the copy test.
    tensor = utils.tensor()
    # Using TensorMap.copy
    clone = tensor.copy()
    block_1_values_id = id(tensor.block(0).values)

    # We should have exactly 2 references to the object: one in this function,
    # and one passed to `sys.getrefcount`
    assert sys.getrefcount(tensor) == 2

    del tensor

    assert id(clone.block(0).values) != block_1_values_id
    assert_equal(clone.block(0).values, np.full((3, 1, 1), 1.0))

    # Using copy.deepcopy
    other_clone = copy.deepcopy(clone)
    block_1_values_id = id(clone.block(0).values)

    del clone

    assert id(other_clone.block(0).values) != block_1_values_id
    assert_equal(other_clone.block(0).values, np.full((3, 1, 1), 1.0))


def test_shallow_copy_error(tensor):
    msg = "shallow copies of TensorMap are not possible, use a deepcopy instead"
    with pytest.raises(ValueError, match=msg):
        copy.copy(tensor)


def test_keys(tensor):
    assert tensor.keys.names == ["key_1", "key_2"]
    assert len(tensor.keys) == 4
    assert len(tensor) == 4
    assert tuple(tensor.keys[0]) == (0, 0)
    assert tuple(tensor.keys[1]) == (1, 0)
    assert tuple(tensor.keys[2]) == (2, 2)
    assert tuple(tensor.keys[3]) == (2, 3)


def test_print(tensor):
    """
    Test routine for the print function of the TensorBlock.
    It compare the results with those in a file.
    """
    repr = tensor.__repr__()
    expected = """TensorMap with 4 blocks
keys: key_1  key_2
        0      0
        1      0
        2      2
        2      3"""
    assert expected == repr


def test_print_large(large_tensor):
    _print = large_tensor.__repr__()
    expected = """TensorMap with 12 blocks
keys: key_1  key_2
        0      0
        1      0
          ...
        2      5
        3      5"""
    assert expected == _print


def test_labels_names(tensor):
    assert tensor.sample_names == ["s"]
    assert tensor.component_names == ["c"]
    assert tensor.property_names == ["p"]


def test_block(tensor):
    # block by index
    block = tensor.block(2)
    assert_equal(block.values, np.full((4, 3, 1), 3.0))

    # block by index with __getitem__
    block = tensor[2]
    assert_equal(block.values, np.full((4, 3, 1), 3.0))

    # block by kwargs
    block = tensor.block(key_1=1, key_2=0)
    assert_equal(block.values, np.full((3, 1, 3), 2.0))

    # block by Label entry
    block = tensor.block(tensor.keys[0])
    assert_equal(block.values, np.full((3, 1, 1), 1.0))

    # block by Label entry with __getitem__
    block = tensor[tensor.keys[0]]
    assert_equal(block.values, np.full((3, 1, 1), 1.0))

    # 0 blocks matching criteria
    msg = "couldn't find any block matching \\(key_1=3\\)"
    with pytest.raises(ValueError, match=msg):
        tensor.block(key_1=3)

    # more than one block matching criteria
    msg = (
        "more than one block matched \\(key_2=0\\), use `TensorMap.blocks` "
        "to get all of them"
    )
    with pytest.raises(ValueError, match=msg):
        tensor.block(key_2=0)


def test_blocks(tensor):
    # block by index
    blocks = tensor.blocks(2)
    assert len(blocks) == 1
    assert_equal(blocks[0].values, np.full((4, 3, 1), 3.0))

    # block by kwargs
    blocks = tensor.blocks(key_1=1, key_2=0)
    assert len(blocks) == 1
    assert_equal(blocks[0].values, np.full((3, 1, 3), 2.0))

    # more than one block
    blocks = tensor.blocks(key_2=0)
    assert len(blocks) == 2

    assert_equal(blocks[0].values, np.full((3, 1, 1), 1.0))
    assert_equal(blocks[1].values, np.full((3, 1, 3), 2.0))


def test_iter(tensor):
    expected = [
        ((0, 0), np.full((3, 1, 1), 1.0)),
        ((1, 0), np.full((3, 1, 3), 2.0)),
        ((2, 2), np.full((4, 3, 1), 3.0)),
        ((2, 3), np.full((4, 3, 1), 4.0)),
    ]

    for i, (key, block) in enumerate(tensor.items()):
        expected_key, expected_values = expected[i]

        assert tuple(key) == expected_key
        assert_equal(block.values, expected_values)

    for i, block in enumerate(tensor):
        _, expected_values = expected[i]

        assert_equal(block.values, expected_values)


def test_keys_to_properties(tensor):
    tensor = tensor.keys_to_properties("key_1")

    assert tensor.keys.names == ["key_2"]
    assert tuple(tensor.keys[0]) == (0,)
    assert tuple(tensor.keys[1]) == (2,)
    assert tuple(tensor.keys[2]) == (3,)

    # The new first block contains the old first two blocks merged
    block = tensor.block(0)
    assert tuple(block.samples[0]) == (0,)
    assert tuple(block.samples[1]) == (1,)
    assert tuple(block.samples[2]) == (2,)
    assert tuple(block.samples[3]) == (3,)
    assert tuple(block.samples[4]) == (4,)

    assert len(block.components), 1
    assert tuple(block.components[0][0]), (0,)

    assert block.properties.names == ["key_1", "p"]
    assert tuple(block.properties[0]) == (0, 0)
    assert tuple(block.properties[1]) == (1, 3)
    assert tuple(block.properties[2]) == (1, 4)
    assert tuple(block.properties[3]) == (1, 5)

    expected = np.array(
        [
            [[1.0, 2.0, 2.0, 2.0]],
            [[0.0, 2.0, 2.0, 2.0]],
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 2.0, 2.0, 2.0]],
            [[1.0, 0.0, 0.0, 0.0]],
        ]
    )
    assert_equal(block.values, expected)

    gradient = block.gradient("g")
    assert tuple(gradient.samples[0]) == (0, -2)
    assert tuple(gradient.samples[1]) == (0, 3)
    assert tuple(gradient.samples[2]) == (3, -2)
    assert tuple(gradient.samples[3]) == (4, 3)

    expected = np.array(
        [
            [[11.0, 12.0, 12.0, 12.0]],
            [[0.0, 12.0, 12.0, 12.0]],
            [[0.0, 12.0, 12.0, 12.0]],
            [[11.0, 0.0, 0.0, 0.0]],
        ]
    )
    assert_equal(gradient.values, expected)

    # The new second block contains the old third block
    block = tensor.block(1)
    assert block.properties.names == ["key_1", "p"]
    assert tuple(block.properties[0]) == (2, 0)

    assert_equal(block.values, np.full((4, 3, 1), 3.0))

    # The new third block contains the old fourth block
    block = tensor.block(2)
    assert block.properties.names == ["key_1", "p"]
    assert tuple(block.properties[0]) == (2, 0)

    assert_equal(block.values, np.full((4, 3, 1), 4.0))


def test_keys_to_samples(tensor):
    tensor = tensor.keys_to_samples("key_2", sort_samples=True)

    assert tensor.keys.names == ["key_1"]
    assert tuple(tensor.keys[0]) == (0,)
    assert tuple(tensor.keys[1]) == (1,)
    assert tuple(tensor.keys[2]) == (2,)

    # The first two blocks are not modified
    block = tensor.block(0)
    assert block.samples.names, ("s", "key_2")
    assert tuple(block.samples[0]) == (0, 0)
    assert tuple(block.samples[1]) == (2, 0)
    assert tuple(block.samples[2]) == (4, 0)

    assert_equal(block.values, np.full((3, 1, 1), 1.0))

    block = tensor.block(1)
    assert block.samples.names == ["s", "key_2"]
    assert tuple(block.samples[0]) == (0, 0)
    assert tuple(block.samples[1]) == (1, 0)
    assert tuple(block.samples[2]) == (3, 0)

    assert_equal(block.values, np.full((3, 1, 3), 2.0))

    # The new third block contains the old third and fourth blocks merged
    block = tensor.block(2)

    assert block.samples.names == ["s", "key_2"]
    assert tuple(block.samples[0]) == (0, 2)
    assert tuple(block.samples[1]) == (0, 3)
    assert tuple(block.samples[2]) == (1, 3)
    assert tuple(block.samples[3]) == (2, 3)
    assert tuple(block.samples[4]) == (3, 2)
    assert tuple(block.samples[5]) == (5, 3)
    assert tuple(block.samples[6]) == (6, 2)
    assert tuple(block.samples[7]) == (8, 2)

    expected = np.array(
        [
            [[3.0], [3.0], [3.0]],
            [[4.0], [4.0], [4.0]],
            [[4.0], [4.0], [4.0]],
            [[4.0], [4.0], [4.0]],
            [[3.0], [3.0], [3.0]],
            [[4.0], [4.0], [4.0]],
            [[3.0], [3.0], [3.0]],
            [[3.0], [3.0], [3.0]],
        ]
    )
    assert_equal(block.values, expected)

    gradient = block.gradient("g")
    assert gradient.samples.names == ["sample", "g"]
    assert tuple(gradient.samples[0]) == (1, 1)
    assert tuple(gradient.samples[1]) == (4, -2)
    assert tuple(gradient.samples[2]) == (5, 3)

    expected = np.array(
        [
            [[14.0], [14.0], [14.0]],
            [[13.0], [13.0], [13.0]],
            [[14.0], [14.0], [14.0]],
        ]
    )
    assert_equal(gradient.values, expected)


def test_keys_to_samples_unsorted(tensor):
    tensor = tensor.keys_to_samples("key_2", sort_samples=False)

    block = tensor.block(2)
    assert block.samples.names, ("s", "key_2")
    assert tuple(block.samples[0]) == (0, 2)
    assert tuple(block.samples[1]) == (3, 2)
    assert tuple(block.samples[2]) == (6, 2)
    assert tuple(block.samples[3]) == (8, 2)
    assert tuple(block.samples[4]) == (0, 3)
    assert tuple(block.samples[5]) == (1, 3)
    assert tuple(block.samples[6]) == (2, 3)
    assert tuple(block.samples[7]) == (5, 3)


def test_components_to_properties(tensor):
    tensor = tensor.components_to_properties("c")

    block = tensor.block(0)
    assert block.samples.names == ["s"]
    assert tuple(block.samples[0]) == (0,)
    assert tuple(block.samples[1]) == (2,)
    assert tuple(block.samples[2]) == (4,)

    assert block.components == []

    assert block.properties.names == ["c", "p"]
    assert tuple(block.properties[0]) == (0, 0)

    block = tensor.block(3)
    assert block.samples.names, ("s",)
    assert tuple(block.samples[0]) == (0,)
    assert tuple(block.samples[1]) == (1,)
    assert tuple(block.samples[2]) == (2,)
    assert tuple(block.samples[3]) == (5,)

    assert block.components == []

    assert block.properties.names == ["c", "p"]
    assert tuple(block.properties[0]) == (0, 0)
    assert tuple(block.properties[1]) == (1, 0)
    assert tuple(block.properties[2]) == (2, 0)


def test_empty_tensor():
    empty_tensor = TensorMap(keys=Labels.empty(["key"]), blocks=[])

    assert empty_tensor.keys.names == ["key"]

    assert empty_tensor.sample_names == []
    assert empty_tensor.component_names == []
    assert empty_tensor.property_names == []

    # check the `blocks` function
    assert empty_tensor.blocks() == []

    assert empty_tensor.blocks(key=3) == []
    message = "invalid parameter: 'not_a_key' is not part of the keys for this tensor"
    with pytest.raises(MetatensorError, match=message):
        empty_tensor.blocks(not_a_key=3)

    # check the `block` function
    message = "there are no blocks in this TensorMap"
    with pytest.raises(ValueError, match=message):
        assert empty_tensor.block()

    with pytest.raises(ValueError, match=message):
        empty_tensor.block(key=3)

    message = "invalid parameter: 'not_a_key' is not part of the keys for this tensor"
    with pytest.raises(MetatensorError, match=message):
        empty_tensor.block(not_a_key=3)

    # check the `blocks_matching` function
    assert empty_tensor.blocks_matching(Labels("key", np.array([[3]]))) == []

    message = "invalid parameter: 'not_a_key' is not part of the keys for this tensor"
    with pytest.raises(MetatensorError, match=message):
        empty_tensor.blocks_matching(Labels("not_a_key", np.array([[3]])))

    message = "block index out of bounds: we have 0 blocks but the index is 3"
    with pytest.raises(IndexError, match=message):
        empty_tensor.block(3)

    # check the `keys_to_xxx` function
    message = "invalid parameter: there are no keys to move in an empty TensorMap"
    with pytest.raises(MetatensorError, match=message):
        empty_tensor.keys_to_samples("key")

    with pytest.raises(MetatensorError, match=message):
        empty_tensor.keys_to_properties("key")


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch to be run")
def test_different_device():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        block_1 = TensorBlock(
            values=torch.tensor([[3.0, 4.0]], device="meta"),
            samples=Labels.range("s", 1),
            components=[],
            properties=Labels.range("p", 2),
        )

    message = (
        "Blocks values and keys for this TensorMap are on different devices: "
        "keys are always on CPU, and blocks values are on device 'meta'."
    )
    with pytest.warns(DeviceWarning, match=message):
        _ = TensorMap(Labels.range("k", 1), [block_1.copy()])

    block_2 = TensorBlock(
        values=torch.tensor([[3.0, 4.0]]),
        samples=Labels.range("s", 1),
        components=[],
        properties=Labels.range("p", 2),
    )

    message = (
        "all blocks in a TensorMap must have the same device, got 'meta' and 'cpu'"
    )
    with pytest.raises(ValueError, match=message):
        _ = TensorMap(Labels.range("k", 2), [block_1, block_2])


def test_different_dtype():
    block_1 = TensorBlock(
        values=np.array([[3.0, 4.0]], dtype=np.float64),
        samples=Labels.range("s", 1),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_2 = TensorBlock(
        values=np.array([[3.0, 4.0]], dtype=np.float32),
        samples=Labels.range("s", 1),
        components=[],
        properties=Labels.range("p", 2),
    )

    message = (
        "all blocks in a TensorMap must have the same dtype, got float64 and float32"
    )
    with pytest.raises(ValueError, match=message):
        _ = TensorMap(Labels.range("k", 2), [block_1, block_2])


def test_to():
    block = TensorBlock(
        values=np.array([[3.0, 4.0]]),
        samples=Labels.range("s", 1),
        components=[],
        properties=Labels.range("p", 2),
    )
    tensor = TensorMap(Labels.range("k", 1), [block])

    assert tensor.device == "cpu"
    assert tensor.dtype == np.float64

    converted = tensor.to(dtype=np.float32)
    assert converted.dtype == np.float32

    # check that the code handles both positional and keyword arguments
    device = "cpu"
    moved = tensor.to(device, dtype=np.float32)
    moved = tensor.to(np.float32, device)
    moved = tensor.to(np.float32, device=device)
    moved = tensor.to(device, np.float32)

    message = "can not give a device twice in `TensorMap.to`"
    with pytest.raises(ValueError, match=message):
        moved = tensor.to("cpu", device="cpu")

    message = "can not give a dtype twice in `TensorMap.to`"
    with pytest.raises(ValueError, match=message):
        moved = tensor.to(np.float32, dtype=np.float32)

    # string positional arguments are assumed to be devices
    message = "can not move numpy array to non-cpu device: test"
    with pytest.raises(ValueError, match=message):
        moved = tensor.to("test")

    message = "unexpected type in `TensorMap.to`: <class 'numpy.ndarray'>"
    with pytest.raises(TypeError, match=message):
        moved = tensor.to(np.array([0]))

    if HAS_TORCH:
        tensor = converted.to(arrays="torch")
        assert isinstance(tensor.block(0).values, torch.Tensor)

        devices = ["meta", torch.device("meta")]
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
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

                moved = tensor.to(device=device)

            assert moved.device.type == torch.device(device).type
