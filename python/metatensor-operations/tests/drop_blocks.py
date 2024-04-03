import os

import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def test_tensor_map() -> TensorMap:
    return metatensor.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )


def test_drop_all(test_tensor_map):
    # test the behavior when all the keys are dropped
    # expected behavior: empty TensorMap
    keys = test_tensor_map.keys
    tensor = metatensor.drop_blocks(test_tensor_map, keys)
    empty_tensor = TensorMap(keys=Labels.empty(keys.names), blocks=[])
    assert metatensor.equal(tensor, empty_tensor)


def test_drop_two_random_keys(test_tensor_map):
    # test the behavior when two random keys are dropped
    n_blocks = len(test_tensor_map.keys)
    np.random.seed(0xFEA123)
    indices_to_drop = np.random.randint(0, n_blocks, 2)

    values_to_drop = [test_tensor_map.keys[i].values for i in indices_to_drop]
    keys_to_drop = Labels(
        test_tensor_map.keys.names,
        np.vstack(values_to_drop),
    )

    values_to_keep = [
        test_tensor_map.keys[i].values
        for i in range(n_blocks)
        if i not in indices_to_drop
    ]
    keys_to_keep = Labels(
        test_tensor_map.keys.names,
        np.vstack(values_to_keep),
    )

    ref_blocks = [test_tensor_map[key].copy() for key in keys_to_keep]
    ref_tensor = TensorMap(keys_to_keep, ref_blocks)

    new_tensor = metatensor.drop_blocks(test_tensor_map, keys_to_drop)
    assert metatensor.equal(new_tensor, ref_tensor)


def test_drop_nothing(test_tensor_map):
    # test when an empty set of keys are dropped
    empty_key = Labels.empty(test_tensor_map.keys.names)
    new_tensor = metatensor.drop_blocks(test_tensor_map, empty_key)
    assert new_tensor == test_tensor_map


def test_not_existent(test_tensor_map):
    # test when keys that don't appear in the TensorMap are dropped
    non_existent_key = Labels(
        test_tensor_map.keys.names, np.array([[-1, -1, -1], [0, 0, 0]])
    )
    message = (
        "\\(center_type=-1, neighbor_1_type=-1, neighbor_2_type=-1\\) "
        "is not present in this tensor"
    )
    with pytest.raises(ValueError, match=message):
        metatensor.drop_blocks(test_tensor_map, non_existent_key)


def test_copy_flag(test_tensor_map):
    """
    Checks that settings the copy flag to true or false gives the same result.
    Also checks that inplace modifications of the underlying data doesn't affect
    the copied returned tensor, but it does for the un-copied version.
    """
    # Define some keys to drop
    values_to_drop = [test_tensor_map.keys[i].values for i in range(5)]
    keys_to_drop = Labels(
        test_tensor_map.keys.names,
        np.vstack(values_to_drop),
    )

    # Drop blocks with copy=True flag
    new_tensor_copied = metatensor.drop_blocks(test_tensor_map, keys_to_drop, copy=True)

    # Drop blocks with copy=False flag
    new_tensor_not_copied = metatensor.drop_blocks(
        test_tensor_map, keys_to_drop, copy=False
    )

    # Check that the resulting tensor are equal whether or not they are copied
    assert metatensor.equal(new_tensor_copied, new_tensor_not_copied)

    # Now modify the original tensor's block values in place
    for block in test_tensor_map.blocks():
        block.values[:] += 3.14

    # The copied tensor's values should not have been edited
    for key in new_tensor_copied.keys:
        assert np.all(new_tensor_copied[key].values != test_tensor_map[key].values)

    # But the un-copied tensor's values should have been
    for key in new_tensor_not_copied.keys:
        assert np.all(new_tensor_not_copied[key].values == test_tensor_map[key].values)
        assert np.all(
            new_tensor_not_copied[key].values == new_tensor_copied[key].values[:] + 3.14
        )
