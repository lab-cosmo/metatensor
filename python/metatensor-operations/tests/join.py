from os import path

import numpy as np
import pytest
from numpy.testing import assert_equal

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = path.join(path.dirname(__file__), "data")


@pytest.fixture
def tensor():
    tensor = metatensor.load(
        path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        use_numpy=True,
    )

    msg = (
        "Tensor must have at least one gradient. When no gradients are present certain "
        "tests will pass without testing anything."
    )
    assert len(tensor.block(0).gradients_list()) > 0, msg

    return tensor


@pytest.fixture
def components_tensor():
    components_tensor = metatensor.load(
        path.join(DATA_ROOT, "qm7-spherical-expansion.npz"), use_numpy=True
    )

    # Test if Tensormaps have at least one gradient. This avoids dropping gradient
    # tests silently by removing gradients from the reference data
    assert "positions" in components_tensor.block(0).gradients_list()

    return components_tensor


def test_wrong_axis(tensor):
    """Test error with unknown `axis` keyword."""
    with pytest.raises(ValueError, match="values for the `axis` parameter"):
        metatensor.join([tensor, tensor, tensor], axis="foo")


def test_wrong_type(tensor):
    """Test if a wrong type (e.g., TensorMap) is provided."""
    with pytest.raises(TypeError, match="list or a tuple"):
        metatensor.join(tensor, axis="properties")


def test_wrong_different_keys(tensor):
    """Test if a wrong type (e.g., TensorMap) is provided."""
    match = "'foo' is not a valid option for `different_keys`"
    with pytest.raises(ValueError, match=match):
        metatensor.join([tensor, tensor], axis="properties", different_keys="foo")


@pytest.mark.parametrize("tensor", ([], ()))
def test_no_tensormaps(tensor):
    """Test if an empty list or tuple is provided."""
    with pytest.raises(ValueError, match="provide at least one"):
        metatensor.join(tensor, axis="properties")


def test_single_tensormap(tensor):
    """Test if only one TensorMap is provided."""
    joined_tensor = metatensor.join([tensor], axis="properties")
    assert joined_tensor is tensor


@pytest.mark.parametrize("axis", ["samples", "properties"])
def test_join_components(components_tensor, axis):
    """Test join for tensors with components."""
    metatensor.join([components_tensor, components_tensor], axis=axis)


def test_join_properties_metadata(tensor):
    """Test join function with three tensors along `properties`.

    We check for the values below"""

    joined_tensor = metatensor.join([tensor, tensor, tensor], axis="properties")

    # test property names
    names = tensor.block(0).properties.names
    assert joined_tensor.block(0).properties.names == ["tensor"] + names

    # test property values
    tensor_prop = np.unique(joined_tensor.block(0).properties["tensor"])
    assert set(tensor_prop) == set((0, 1, 2))

    # test if gradients exist
    assert sorted(joined_tensor[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


def test_join_properties_values(tensor):
    """Test values for joining along `properties`"""
    first_property = Labels(
        tensor[0].properties.names,
        tensor[0].properties.values[:1],
    )
    slice_1 = metatensor.slice(tensor, axis="properties", labels=first_property)

    other_properties = Labels(
        tensor[0].properties.names,
        tensor[0].properties.values[1:],
    )
    slice_2 = metatensor.slice(tensor, axis="properties", labels=other_properties)

    joined_tensor = metatensor.join([slice_1, slice_2], axis="properties")

    # We can not use `metatensor.equal_raise` for the checks because the meta data
    # differs by the tensor entry.
    for key, block in tensor.items():
        joined_block = joined_tensor[key]

        assert_equal(joined_block.values, block.values)

        for parameter, gradient in block.gradients():
            joined_gradient = joined_block.gradient(parameter)
            assert_equal(joined_gradient.values, gradient.values)


def test_join_properties_with_same_property_names(tensor):
    """Test join function with three tensor along properties"""

    joined_tensor = metatensor.join([tensor, tensor, tensor], axis="properties")

    # test property names
    names = tensor.block(0).properties.names
    assert joined_tensor.block(0).properties.names == ["tensor"] + names

    # test property values
    tensor_prop = np.unique(joined_tensor.block(0).properties["tensor"])
    assert set(tensor_prop) == set((0, 1, 2))

    # test if gradients exist
    assert sorted(joined_tensor[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


def test_join_properties_with_different_property_names():
    """Test join function with tensors of different property names"""

    keys = Labels.range("frame_a", 1)
    values = np.zeros([1, 1])
    samples = Labels.range("idx", 1)

    tensor_map_a = TensorMap(
        keys=keys,
        blocks=[
            TensorBlock(
                values=values,
                samples=samples,
                components=[],
                properties=Labels.range("p_1", 1),
            )
        ],
    )

    tensor_map_b = TensorMap(
        keys=keys,
        blocks=[
            TensorBlock(
                values=np.zeros([1, 1]),
                samples=samples,
                components=[],
                properties=Labels.range("p_2", 1),
            )
        ],
    )

    joined_tensor = metatensor.join([tensor_map_a, tensor_map_b], axis="properties")
    assert joined_tensor.property_names == ["tensor", "property"]
    assert len(joined_tensor[0].properties) == 2


def test_join_samples_metadata(tensor):
    """Test join function with three tensors along `samples`"""
    joined_tensor = metatensor.join([tensor, tensor, tensor], axis="samples")

    # test sample values
    assert len(joined_tensor.block(0).samples) == 3 * len(tensor.block(0).samples)

    # test if gradients exist
    assert sorted(joined_tensor[0].gradients_list()) == sorted(
        tensor[0].gradients_list()
    )


def test_join_samples_values(tensor):
    """Test values for joining along `samples`"""
    keys = Labels(
        names=tensor.keys.names,
        values=tensor.keys.values[:1],
    )

    tensor = TensorMap(keys, [tensor[0].copy()])
    first_samples = Labels(
        tensor[0].samples.names,
        tensor[0].samples.values[:1],
    )
    slice_1 = metatensor.slice(tensor, axis="samples", labels=first_samples)

    other_samples = Labels(
        tensor[0].samples.names,
        tensor[0].samples.values[1:],
    )
    slice_2 = metatensor.slice(tensor, axis="samples", labels=other_samples)

    joined_tensor = metatensor.join([slice_1, slice_2], axis="samples")

    # We can not use #metatensor.equal_raise for the checks because the meta data
    # differs by the tensor entry.
    for key, block in tensor.items():
        joined_block = joined_tensor[key]

        assert_equal(joined_block.values, block.values)

        for parameter, gradient in block.gradients():
            joined_gradient = joined_block.gradient(parameter)
            assert_equal(joined_gradient.values, gradient.values)


def test_join_samples_with_different_sample_names():
    """Test join function raises an error with different sample names"""
    keys = Labels.range("frame_a", 1)
    values = np.zeros([1, 1])
    properties = Labels.range("idx", 1)

    tensor_map_a = TensorMap(
        keys=keys,
        blocks=[
            TensorBlock(
                values=values,
                samples=Labels.range("samp1", 1),
                components=[],
                properties=properties,
            )
        ],
    )

    tensor_map_b = TensorMap(
        keys=keys,
        blocks=[
            TensorBlock(
                values=np.zeros([1, 1]),
                samples=Labels.range("samp2", 1),
                components=[],
                properties=properties,
            )
        ],
    )

    with pytest.raises(ValueError, match="Sample names are not the same!"):
        metatensor.join([tensor_map_a, tensor_map_b], axis="samples")


def test_split_join_samples(tensor):
    """Test if split and joining along `samples` results in the same TensorMap."""

    labels_1 = Labels(names=["system"], values=np.arange(4).reshape(-1, 1))
    labels_2 = Labels(names=["system"], values=np.arange(4, 10).reshape(-1, 1))

    split_tensors = metatensor.split(
        tensor=tensor, axis="samples", grouped_labels=[labels_1, labels_2]
    )
    joined_tensor = metatensor.join(
        split_tensors, axis="samples", remove_tensor_name=True
    )

    assert joined_tensor == tensor


def test_split_join_properties(tensor):
    """Test if split and joining along `properties` results in the same TensorMap."""
    properties = tensor[0].properties

    labels_1 = Labels(names=properties.names, values=properties.values[:5])
    labels_2 = Labels(names=properties.names, values=properties.values[5:])

    split_tensors = metatensor.split(
        tensor=tensor, axis="properties", grouped_labels=[labels_1, labels_2]
    )
    joined_tensor = metatensor.join(
        split_tensors, axis="properties", remove_tensor_name=True
    )

    assert joined_tensor == tensor


@pytest.mark.parametrize("axis", ["samples", "properties"])
def test_intersection_join(axis, components_tensor):
    tensor_1 = components_tensor

    labels_remove = Labels(names=tensor_1.keys.names, values=tensor_1.keys.values[:1])
    labels_present = Labels(names=tensor_1.keys.names, values=tensor_1.keys.values[1:])

    tensor_2 = metatensor.drop_blocks(components_tensor, labels_remove, copy=True)

    joined_tensor = metatensor.join(
        [tensor_1, tensor_2], axis=axis, different_keys="intersection"
    )

    assert joined_tensor.keys == labels_present


@pytest.mark.parametrize("axis", ["samples", "properties"])
def test_union_join(axis, components_tensor):
    tensor_1 = components_tensor

    labels_remove = Labels(names=tensor_1.keys.names, values=tensor_1.keys.values[:1])

    tensor_2 = metatensor.drop_blocks(components_tensor, labels_remove, copy=True)

    joined_tensor = metatensor.join(
        [tensor_1, tensor_2], axis=axis, different_keys="union"
    )

    assert joined_tensor.keys == tensor_1.keys

    # First block should be same as in tensor_1
    assert joined_tensor[0].values.shape == tensor_1[0].values.shape

    # All other blocks should have doubled the number of samples/features
    for i_block in range(1, len(tensor_1.keys)):
        ref_block = tensor_1[i_block]
        joined_block = joined_tensor[i_block]

        # Check values
        if axis == "samples":
            ref_shape = (2 * ref_block.values.shape[0],) + ref_block.values.shape[1:]
        else:
            ref_shape = ref_block.values.shape[:-1] + (2 * ref_block.values.shape[-1],)

        assert joined_block.values.shape == ref_shape

        # Check values
        for parameter, joined_gradient in joined_block.gradients():
            ref_gradient = ref_block.gradient(parameter)

            if axis == "samples":
                ref_gradient_shape = (
                    2 * ref_gradient.values.shape[0],
                ) + ref_gradient.values.shape[1:]
            else:
                ref_gradient_shape = ref_gradient.values.shape[:-1] + (
                    2 * ref_gradient.values.shape[-1],
                )

            assert joined_gradient.values.shape == ref_gradient_shape


@pytest.mark.parametrize("axis", ["samples", "properties"])
@pytest.mark.parametrize("sort_samples", [True, False])
def test_sort_samples(axis, sort_samples):
    """Test that samples are sorted as requested."""
    values_0 = np.array([[0]])
    values_1 = np.array([[1]])

    keys = Labels(names="_", values=values_0)

    properties = Labels(names="p", values=values_0)
    samples_0 = Labels(names="s", values=values_0)
    samples_1 = Labels(names="s", values=values_1)

    block_0 = TensorBlock(
        values=values_0, samples=samples_0, components=[], properties=properties
    )
    block_1 = TensorBlock(
        values=values_0, samples=samples_1, components=[], properties=properties
    )

    tensor_0 = TensorMap(keys=keys, blocks=[block_0])
    tensor_1 = TensorMap(keys=keys, blocks=[block_1])

    # reverse order of tensors
    joined_tensor = metatensor.join(
        [tensor_1, tensor_0],
        axis=axis,
        sort_samples=sort_samples,
        remove_tensor_name=True,
    )

    if sort_samples:
        ref_samples = Labels(names="s", values=np.array([0, 1]).reshape(-1, 1))
    else:
        ref_samples = Labels(names="s", values=np.array([1, 0]).reshape(-1, 1))

    assert joined_tensor.block(0).samples == ref_samples
