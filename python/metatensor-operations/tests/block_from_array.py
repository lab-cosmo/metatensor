import numpy as np
import pytest

import metatensor


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.parametrize("n_axes", [0, 1])
def test_too_few_axes(n_axes):
    """Test block_from_array when too few axes are provided."""
    with pytest.raises(ValueError, match="at least"):
        metatensor.block_from_array(np.zeros((4,) * n_axes))


def test_without_components():
    """Test block_from_array for a 2D array."""
    array = np.zeros((6, 7))
    block = metatensor.block_from_array(array)
    assert block.values is array

    assert block.samples.names == ["sample"]
    np.testing.assert_equal(
        block.samples.values, np.arange(array.shape[0]).reshape((-1, 1))
    )

    assert block.properties.names == ["property"]
    np.testing.assert_equal(
        block.properties.values, np.arange(array.shape[1]).reshape((-1, 1))
    )


def test_with_components():
    """Test block_from_array with components."""
    array = array = np.zeros((6, 5, 7))
    block = metatensor.block_from_array(array)
    assert block.values is array

    assert block.samples.names == ["sample"]
    np.testing.assert_equal(
        block.samples.values, np.arange(array.shape[0]).reshape((-1, 1))
    )

    assert len(block.components) == 1
    component = block.components[0]
    assert component.names == ["component_1"]
    np.testing.assert_equal(
        component.values, np.arange(array.shape[1]).reshape((-1, 1))
    )

    assert block.properties.names == ["property"]
    np.testing.assert_equal(
        block.properties.values, np.arange(array.shape[2]).reshape((-1, 1))
    )


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_with_components():
    """Test block_from_array with components and torch arrays"""
    array = array = torch.zeros((6, 5, 7))
    block = metatensor.block_from_array(array)
    assert block.values is array

    assert block.samples.names == ["sample"]
    np.testing.assert_equal(
        block.samples.values, np.arange(array.shape[0]).reshape((-1, 1))
    )

    assert len(block.components) == 1
    component = block.components[0]
    assert component.names == ["component_1"]
    np.testing.assert_equal(
        component.values, np.arange(array.shape[1]).reshape((-1, 1))
    )

    assert block.properties.names == ["property"]
    np.testing.assert_equal(
        block.properties.values, np.arange(array.shape[2]).reshape((-1, 1))
    )
