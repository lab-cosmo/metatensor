import os

import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


try:
    import torch  # noqa

    HAS_TORCH = True
    if torch.cuda.is_available():
        HAS_CUDA = True
    else:
        HAS_CUDA = False
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
TEST_FILE = "qm7-spherical-expansion.npz"

# ===== Fixtures and helper functions =====


@pytest.fixture
def tensor() -> TensorMap:
    return metatensor.load(
        os.path.join(DATA_ROOT, TEST_FILE),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )


def _construct_empty_slice_block(block, axis, labels) -> TensorBlock:
    if axis == "samples":
        reference_block = TensorBlock(
            values=block.values[:0, :],
            samples=labels,
            components=block.components,
            properties=block.properties,
        )
        for parameter, gradient in block.gradients():
            reference_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values[:0, ...],
                    samples=Labels.empty(gradient.samples.names),
                    components=gradient.components,
                    properties=block.properties,
                ),
            )
        return reference_block
    elif axis == "properties":
        reference_block = TensorBlock(
            values=block.values[..., :0],
            samples=block.samples,
            components=block.components,
            properties=labels,
        )
    for parameter, gradient in block.gradients():
        reference_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient.values[..., :0],
                samples=gradient.samples,
                components=gradient.components,
                properties=labels,
            ),
        )
    return reference_block


def _check_sliced_block_samples(block, sliced_block, structures_to_keep):
    # no slicing of properties has occurred
    assert np.all(block.properties == sliced_block.properties)

    # samples have been sliced to the correct dimension
    assert len(sliced_block.samples) == len(
        [s for s in block.samples["system"] if s in structures_to_keep]
    )

    # samples in sliced block only feature desired structure indices
    assert np.all([s in structures_to_keep for s in sliced_block.samples["system"]])

    # no components have been sliced
    assert len(sliced_block.components) == len(block.components)
    for sliced_c, c in zip(sliced_block.components, block.components):
        assert np.all(sliced_c == c)

    # we have the right values
    samples_filter = np.array(
        [sample["system"] in structures_to_keep for sample in block.samples]
    )
    assert np.all(sliced_block.values == block.values[samples_filter, ...])

    for parameter, gradient in block.gradients():
        sliced_gradient = sliced_block.gradient(parameter)
        # no slicing of properties has occurred
        assert np.all(sliced_gradient.properties == gradient.properties)

        # samples have been updated to refer to the new samples
        assert np.max(sliced_gradient.samples["sample"]) < sliced_block.values.shape[0]

        # other columns in the gradient samples have been sliced correctly
        gradient_sample_filter = samples_filter[gradient.samples["sample"]]
        if len(gradient.samples.names) > 1:
            expected = gradient.samples.values[gradient_sample_filter, 1:]
            sliced_gradient_samples = sliced_gradient.samples.values[:, 1:]
            assert np.all(sliced_gradient_samples == expected)

        # same components as the original
        assert len(gradient.components) == len(sliced_gradient.components)
        for sliced_c, c in zip(sliced_gradient.components, gradient.components):
            assert np.all(sliced_c == c)

        expected = gradient.values[gradient_sample_filter]
        assert np.all(sliced_gradient.values == expected)


def _check_sliced_block_properties(block, sliced_block, radial_to_keep):
    # no slicing of samples has occurred
    assert np.all(block.samples == sliced_block.samples)

    # properties have been sliced to the correct dimension
    assert len(sliced_block.properties) == len(
        [n for n in block.properties["n"] if n in radial_to_keep]
    )

    # properties in sliced block only feature desired radial indices
    assert np.all([n in radial_to_keep for n in sliced_block.properties["n"]])

    # no components have been sliced
    assert len(sliced_block.components) == len(block.components)
    for sliced_c, c in zip(sliced_block.components, block.components):
        assert np.all(sliced_c == c)

    # we have the right values
    property_filter = [property["n"] in radial_to_keep for property in block.properties]
    assert np.all(sliced_block.values == block.values[..., property_filter])

    for parameter, gradient in block.gradients():
        sliced_gradient = sliced_block.gradient(parameter)
        # no slicing of samples has occurred
        assert np.all(sliced_gradient.samples == gradient.samples)

        # properties have been sliced to the correct dimension
        assert len(sliced_gradient.properties) == len(
            [n for n in gradient.properties["n"] if n in radial_to_keep]
        )

        # properties in sliced block only feature desired radial indices
        assert np.all([n in radial_to_keep for n in sliced_gradient.properties["n"]])

        # same components as the original
        assert len(gradient.components) == len(sliced_gradient.components)
        for sliced_c, c in zip(sliced_gradient.components, gradient.components):
            assert np.all(sliced_c == c)

        # we have the right values
        assert np.all(sliced_gradient.values == gradient.values[..., property_filter])


def _check_empty_block(block, sliced_block, axis):
    if axis == "s":
        sliced_axis, unsliced_axis = 0, -1
    else:
        sliced_axis, unsliced_axis = -1, 0

    # sliced block has no values
    assert len(sliced_block.values.flatten()) == 0
    # sliced block has dimension zero for properties
    assert sliced_block.values.shape[sliced_axis] == 0
    # sliced block has original dimension for samples
    assert sliced_block.values.shape[unsliced_axis] == block.values.shape[unsliced_axis]

    for parameter, gradient in block.gradients():
        sliced_gradient = sliced_block.gradient(parameter)
        # no slicing of samples has occurred
        if axis == "s":
            assert np.all(sliced_gradient.properties == gradient.properties)
        else:
            assert np.all(sliced_gradient.samples == gradient.samples)

        # sliced block contains zero properties
        assert sliced_gradient.values.shape[sliced_axis] == 0


# ===== Tests for slicing along samples =====


def test_slice_block_samples(tensor):
    # Slice only 'systems' 2, 4, 6, 8
    systems_to_keep = np.arange(2, 10, 2).reshape(-1, 1)
    samples = Labels(
        names=["system"],
        values=systems_to_keep,
    )
    block = tensor.block(0)
    sliced_block = metatensor.slice_block(
        block,
        axis="samples",
        labels=samples,
    )
    _check_sliced_block_samples(block, sliced_block, systems_to_keep)

    # Slice to an empty block
    # Slice only 'systems' -1 (i.e. a sample that doesn't exist in the data)
    samples = Labels(
        names=["system"],
        values=np.array([-1]).reshape(-1, 1),
    )

    sliced_block = metatensor.slice_block(
        block,
        axis="samples",
        labels=samples,
    )

    _check_empty_block(block, sliced_block, "s")


def test_slice_samples(tensor):
    # Slice only 'systems' 2, 4, 6, 8
    systems_to_keep = np.arange(2, 10, 2).reshape(-1, 1)
    samples = Labels(
        names=["system"],
        values=systems_to_keep,
    )
    sliced_tensor = metatensor.slice(
        tensor,
        axis="samples",
        labels=samples,
    )

    for key, block in tensor.items():
        sliced_block = sliced_tensor.block(key)
        _check_sliced_block_samples(block, sliced_block, systems_to_keep)

    # all the keys in the sliced tensor are in the original
    assert np.all(tensor.keys == sliced_tensor.keys)

    # ===== Slice to all empty blocks =====
    # Slice only 'systems' -1 (i.e. a sample that doesn't exist in the data)
    samples = Labels(
        names=["system"],
        values=np.array([-1]).reshape(-1, 1),
    )

    sliced_tensor = metatensor.slice(
        tensor,
        axis="samples",
        labels=samples,
    )

    for sliced_block in sliced_tensor:
        _check_empty_block(tensor.block(key), sliced_block, "s")


# ===== Tests for slicing along properties =====


def test_slice_block_properties(tensor):
    # Slice only 'n' (i.e. radial channels) 1, 3
    radial_to_keep = np.arange(1, 5, 2).reshape(-1, 1)
    properties = Labels(
        names=["n"],
        values=radial_to_keep,
    )

    block = tensor.block(0)
    sliced_block = metatensor.slice_block(
        block,
        axis="properties",
        labels=properties,
    )
    _check_sliced_block_properties(block, sliced_block, radial_to_keep)

    # Slice to an empty block
    # Slice only 'n' (i.e. radial channels) -1 (i.e. non-existent channel)
    properties = Labels(
        names=["n"],
        values=np.array([-1]).reshape(-1, 1),
    )

    sliced_block = metatensor.slice_block(
        block,
        axis="properties",
        labels=properties,
    )

    _check_empty_block(block, sliced_block, "p")


def test_slice_properties(tensor):
    # Slice only 'n' (i.e. radial channels) 1, 3
    radial_to_keep = np.arange(1, 5, 2).reshape(-1, 1)
    properties = Labels(
        names=["n"],
        values=radial_to_keep,
    )

    sliced_tensor = metatensor.slice(
        tensor,
        axis="properties",
        labels=properties,
    )

    for key, block in tensor.items():
        sliced_block = sliced_tensor.block(key)
        _check_sliced_block_properties(block, sliced_block, radial_to_keep)

    # Check 5: all the keys in the sliced tensor are in the original
    assert np.all(tensor.keys == sliced_tensor.keys)

    # ===== Slice to all empty blocks =====
    # Slice only 'n' (i.e. radial channels) -1 (i.e. non-existent channel)
    properties = Labels(
        names=["n"],
        values=np.array([-1]).reshape(-1, 1),
    )

    sliced_tensor = metatensor.slice(
        tensor,
        axis="properties",
        labels=properties,
    )

    for key, block in tensor.items():
        sliced_block = sliced_tensor.block(key)
        _check_empty_block(block, sliced_block, "p")


# ===== Tests slicing both samples and properties =====


def test_slice_block_samples_and_properties(tensor):
    block = tensor.block(5)
    # Slice 'center' 1, 3, 5
    centers_to_keep = np.arange(1, 7, 2).reshape(-1, 1)
    samples = Labels(
        names=["atom"],
        values=centers_to_keep,
    )
    # Slice 'n' (i.e. radial channel) 0, 1, 2
    channels_to_keep = np.arange(0, 3).reshape(-1, 1)
    properties = Labels(
        names=["n"],
        values=channels_to_keep,
    )

    # First, slice on samples and then on properties
    sliced_block = metatensor.slice_block(
        block,
        axis="samples",
        labels=samples,
    )
    sliced_block = metatensor.slice_block(
        sliced_block,
        axis="properties",
        labels=properties,
    )

    # only desired samples are in the output.
    assert np.all([c in centers_to_keep for c in sliced_block.samples["atom"]])

    # only desired properties are in the output
    assert np.all([n in channels_to_keep for n in sliced_block.properties["n"]])

    # There are the correct number of samples
    assert sliced_block.values.shape[0] == len(
        [s for s in block.samples if s["atom"] in centers_to_keep]
    )

    # There are the correct number of properties
    assert sliced_block.values.shape[-1] == len(
        [p for p in block.properties if p["n"] in channels_to_keep]
    )

    # we have the right values
    samples_filter = [sample["atom"] in centers_to_keep for sample in block.samples]
    properties_filter = [
        property["n"] in channels_to_keep for property in block.properties
    ]
    expected = block.values[samples_filter][..., properties_filter]
    assert np.all(sliced_block.values == expected)

    # Second, slice on properties and then on samples
    sliced_block = metatensor.slice_block(
        block,
        axis="properties",
        labels=properties,
    )
    sliced_block = metatensor.slice_block(
        sliced_block,
        axis="samples",
        labels=samples,
    )

    # only desired samples are in the output.
    assert np.all([c in centers_to_keep for c in sliced_block.samples["atom"]])

    # only desired properties are in the output
    assert np.all([n in channels_to_keep for n in sliced_block.properties["n"]])

    # There are the correct number of samples
    assert sliced_block.values.shape[0] == len(
        [s for s in block.samples if s["atom"] in centers_to_keep]
    )

    # There are the correct number of properties
    assert sliced_block.values.shape[-1] == len(
        [p for p in block.properties if p["n"] in channels_to_keep]
    )

    # we have the right values
    samples_filter = [sample["atom"] in centers_to_keep for sample in block.samples]
    properties_filter = [
        property["n"] in channels_to_keep for property in block.properties
    ]
    expected = block.values[samples_filter][..., properties_filter]
    assert np.all(sliced_block.values == expected)


def test_slicing_by_empty(tensor):
    empty_labels_samples = Labels.empty(tensor.sample_names)

    # Empty block returned if no samples to slice by are passed
    reference_block = _construct_empty_slice_block(
        tensor.block(0), "samples", empty_labels_samples
    )
    assert metatensor.equal_block(
        metatensor.slice_block(
            tensor.block(0), axis="samples", labels=empty_labels_samples
        ),
        reference_block,
    )

    # Empty tensor returned if no samples to slice by are passed
    block_list = [
        _construct_empty_slice_block(block, "samples", empty_labels_samples)
        for block in tensor
    ]
    reference_tensor = TensorMap(tensor.keys, block_list)
    assert metatensor.equal(
        metatensor.slice(tensor, axis="samples", labels=empty_labels_samples),
        reference_tensor,
    )

    empty_labels_properties = Labels.empty(tensor.property_names)
    # Empty block returned if no properties to slice by are passed
    reference_block = _construct_empty_slice_block(
        tensor.block(0), "properties", empty_labels_properties
    )
    assert metatensor.equal_block(
        metatensor.slice_block(
            tensor.block(0), axis="properties", labels=empty_labels_properties
        ),
        reference_block,
    )

    # Empty tensor returned if no properties to slice by are passed
    block_list = [
        _construct_empty_slice_block(block, "properties", empty_labels_properties)
        for block in tensor
    ]
    reference_tensor = TensorMap(tensor.keys, block_list)
    assert metatensor.equal(
        metatensor.slice(tensor, axis="properties", labels=empty_labels_properties),
        reference_tensor,
    )


def test_slicing_all(tensor):
    # Original block returned if sliced on all samples
    assert metatensor.equal_block(
        metatensor.slice_block(
            tensor.block(0),
            axis="samples",
            labels=metatensor.unique_metadata(
                tensor, axis="samples", names=tensor.sample_names
            ),
        ),
        tensor.block(0),
    )

    # Original tensor returned if sliced on all samples
    assert metatensor.equal(
        metatensor.slice(
            tensor,
            axis="samples",
            labels=metatensor.unique_metadata(
                tensor, axis="samples", names=tensor.sample_names
            ),
        ),
        tensor,
    )

    # Original block returned if sliced on all properties
    assert metatensor.equal_block(
        metatensor.slice_block(
            tensor.block(0),
            axis="properties",
            labels=metatensor.unique_metadata(
                tensor,
                axis="properties",
                names=tensor.property_names,
            ),
        ),
        tensor.block(0),
    )

    # Original tensor returned if sliced on all properties
    assert metatensor.equal(
        metatensor.slice(
            tensor,
            axis="properties",
            labels=metatensor.unique_metadata(
                tensor, axis="properties", names=tensor.property_names
            ),
        ),
        tensor,
    )


@pytest.mark.skipif(not HAS_CUDA, reason="requires cuda")
@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_slice_different_device():
    array = torch.empty((6, 5, 7), device="cuda")  # doesn't work with device=meta
    block = metatensor.block_from_array(array)
    tensor = TensorMap(Labels.range("_", 1), [block])

    # Slice only samples 2, 4
    structures_to_keep = np.arange(2, 10, 2).reshape(-1, 1)
    samples = Labels(
        names=["sample"],
        values=structures_to_keep,
    )
    metatensor.slice(
        tensor,
        axis="samples",
        labels=samples,
    )


# ===== Tests Errors =====


def test_slice_errors(tensor):
    centers_to_keep = np.arange(1, 7, 2).reshape(-1, 1)
    samples = Labels(
        names=["atom"],
        values=centers_to_keep,
    )

    message = (
        "`tensor` must be a metatensor TensorMap, "
        "not <class 'metatensor.block.TensorBlock'>"
    )
    with pytest.raises(TypeError, match=message):
        metatensor.slice(tensor.block(0), axis="samples", labels=samples)

    message = "`labels` must be metatensor Labels, not <class 'numpy.ndarray'>"
    with pytest.raises(TypeError, match=message):
        metatensor.slice(
            tensor,
            axis="samples",
            labels=np.array([[5], [6]]),
        )


def test_slice_block_errors(tensor):
    centers_to_keep = np.arange(1, 7, 2).reshape(-1, 1)
    samples = Labels(
        names=["atom"],
        values=centers_to_keep,
    )

    message = (
        "`block` must be a metatensor TensorBlock, "
        "not <class 'metatensor.tensor.TensorMap'>"
    )
    with pytest.raises(TypeError, match=message):
        metatensor.slice_block(tensor, axis="samples", labels=samples)

    block = tensor.block(0)
    message = "`labels` must be metatensor Labels, not <class 'numpy.ndarray'>"
    with pytest.raises(TypeError, match=message):
        metatensor.slice_block(
            block,
            axis="samples",
            labels=np.array([[5], [6]]),
        )
