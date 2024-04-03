import os

import numpy as np
import pytest

import metatensor
from metatensor import Labels


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_split_block_samples():
    # use a TensorMap with multiple components
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        use_numpy=True,
    )
    block = tensor.block(o3_lambda=2, center_type=6, neighbor_type=6)

    grouped_labels = [
        Labels(names=["system"], values=np.array([[0], [6], [7]])),
        Labels(names=["system"], values=np.array([[2], [3], [4]])),
        Labels(names=["system"], values=np.array([[1], [5], [8], [9]])),
    ]

    splitted = metatensor.split_block(
        block, axis="samples", grouped_labels=grouped_labels
    )

    assert len(splitted) == 3
    assert sum(len(b.samples) for b in splitted) == len(block.samples)

    for split_block, expected_samples in zip(splitted, grouped_labels):
        structure_values = np.unique(split_block.samples["system"]).reshape(-1, 1)
        assert np.all(structure_values == expected_samples.values)

        # no changes to components & properties
        assert split_block.components == block.components
        assert split_block.properties == block.properties

        # check that the values where split in the right way
        mask = np.zeros(len(block.samples), dtype=bool)
        for structure in expected_samples.values:
            mask = np.logical_or(block.samples["system"] == structure, mask)

        assert np.all(split_block.values == block.values[mask])

        assert len(block.gradients_list()) != 0
        for parameter, gradient in block.gradients():
            split_gradient = split_block.gradient(parameter)

            # only check the non-"sample" dimension, since the "sample" dimension should
            # have been updated.
            gradient_mask = mask[gradient.samples["sample"]]
            assert np.all(
                gradient.samples.values[gradient_mask, 1:]
                == split_gradient.samples.values[:, 1:]
            )

            # no changes to components & properties
            assert split_gradient.components == gradient.components
            assert split_gradient.properties == gradient.properties


def test_split_block_samples_not_everything():
    # use a TensorMap with multiple components
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        use_numpy=True,
    )
    block = tensor.block(o3_lambda=2, center_type=6, neighbor_type=6)

    # using `grouped_labels` with some samples not present in the initial block,
    # and not including all samples
    grouped_labels = [
        # 1 is in the samples, 12 is not
        Labels(names=["system"], values=np.array([[1], [12]])),
        # 18 is not in the samples
        Labels(names=["system"], values=np.array([[18]])),
    ]
    splitted = metatensor.split_block(
        block, axis="samples", grouped_labels=grouped_labels
    )

    assert len(splitted) == 2
    partial = splitted[0]
    assert np.unique(partial.samples["system"]) == 1
    assert partial.components == block.components
    assert partial.properties == block.properties

    empty = splitted[1]
    assert len(empty.samples) == 0
    assert empty.components == block.components
    assert empty.properties == block.properties

    # an empty list of grouped_labels gets you an empty list of blocks
    assert metatensor.split_block(block, axis="samples", grouped_labels=[]) == []


def test_split_samples():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        use_numpy=True,
    )

    grouped_labels = [
        Labels(names=["system"], values=np.array([[0], [6], [7]])),
        Labels(names=["system"], values=np.array([[2], [3], [4]])),
        Labels(names=["system"], values=np.array([[1], [5], [8], [9]])),
    ]

    splitted = metatensor.split(tensor, axis="samples", grouped_labels=grouped_labels)

    for split_tensor in splitted:
        assert split_tensor.keys == tensor.keys

    for split_tensor, expected_samples in zip(splitted, grouped_labels):
        for split_block, block in zip(split_tensor, tensor):
            split_structures = Labels(
                ["system"],
                np.unique(split_block.samples["system"]).reshape(-1, 1),
            )

            block_structures = Labels(
                ["system"], np.unique(block.samples["system"]).reshape(-1, 1)
            )
            assert split_structures == block_structures.intersection(expected_samples)

            # no changes to components & properties
            assert split_block.components == block.components
            assert split_block.properties == block.properties

    # an empty list of grouped_labels gets you an empty list of tensors
    assert metatensor.split(tensor, axis="samples", grouped_labels=[]) == []


def test_split_block_properties():
    # TensorMap with multiple properties
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        use_numpy=True,
    )
    block = tensor.block(center_type=8, neighbor_1_type=6, neighbor_2_type=8)

    grouped_labels = [
        Labels(names=["l", "n_2"], values=np.array([[0, 0], [1, 3], [3, 1]])),
        Labels(names=["l", "n_2"], values=np.array([[4, 2], [4, 3], [4, 1]])),
        Labels(names=["l", "n_2"], values=np.array([[3, 2], [1, 1]])),
    ]
    splitted = metatensor.split_block(
        block, axis="properties", grouped_labels=grouped_labels
    )

    assert len(splitted) == 3
    for split_block, expected_properties in zip(splitted, grouped_labels):
        # no changes to samples & components
        assert split_block.samples == block.samples
        assert split_block.components == block.components

        for p in split_block.properties.view(expected_properties.names):
            assert p in expected_properties

        # check that the values where split in the right way
        mask = np.zeros(len(block.properties), dtype=bool)
        for i, p in enumerate(block.properties.view(expected_properties.names)):
            mask[i] = p in expected_properties

        assert np.all(split_block.values == block.values[..., mask])

        assert len(block.gradients_list()) != 0
        for parameter, gradient in block.gradients():
            split_gradient = split_block.gradient(parameter)

            # no changes to samples & components
            assert split_gradient.samples == gradient.samples
            assert split_gradient.components == gradient.components

            assert np.all(split_gradient.values == gradient.values[..., mask])

    # an empty list of grouped_labels gets you an empty list of blocks
    assert metatensor.split_block(block, axis="properties", grouped_labels=[]) == []


def test_split_properties():
    # TensorMap with multiple properties
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        use_numpy=True,
    )

    grouped_labels = [
        Labels(names=["l", "n_2"], values=np.array([[0, 0], [1, 3], [3, 1]])),
        Labels(names=["l", "n_2"], values=np.array([[4, 2], [4, 3], [4, 1]])),
        Labels(names=["l", "n_2"], values=np.array([[3, 2], [1, 1]])),
    ]
    splitted = metatensor.split(
        tensor, axis="properties", grouped_labels=grouped_labels
    )

    for split_tensor in splitted:
        assert split_tensor.keys == tensor.keys

    for split_tensor, expected_properties in zip(splitted, grouped_labels):
        for split_block, block in zip(split_tensor, tensor):
            # no changes to samples & components
            assert split_block.samples == block.samples
            assert split_block.components == block.components

            for p in split_block.properties.view(expected_properties.names):
                assert p in expected_properties

    # an empty list of grouped_labels gets you an empty list of blocks
    assert metatensor.split(tensor, axis="properties", grouped_labels=[]) == []


def test_split_errors():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        use_numpy=True,
    )
    block = tensor.block(4)
    grouped_labels = [
        Labels(names=["system"], values=np.array([[0], [6], [7]])),
        Labels(names=["system"], values=np.array([[2], [3], [4]])),
        Labels(names=["system"], values=np.array([[1], [5], [8], [9]])),
    ]

    message = (
        "`tensor` must be a metatensor TensorMap, "
        "not <class 'metatensor.block.TensorBlock'>"
    )
    with pytest.raises(TypeError, match=message):
        metatensor.split(block, axis="samples", grouped_labels=grouped_labels)

    message = "axis must be a string, not <class 'float'>"
    with pytest.raises(TypeError, match=message):
        metatensor.split(tensor, axis=3.14, grouped_labels=grouped_labels)

    message = "axis must be either 'samples' or 'properties'"
    with pytest.raises(ValueError, match=message):
        metatensor.split(tensor, axis="buongiorno!", grouped_labels=grouped_labels)

    message = (
        "`grouped_labels` must be a list, " "not <class 'metatensor.labels.Labels'>"
    )
    with pytest.raises(TypeError, match=message):
        metatensor.split(tensor, axis="samples", grouped_labels=grouped_labels[0])

    message = "`grouped_labels` elements must be metatensor Labels, not <class 'str'>"
    with pytest.raises(TypeError, match=message):
        metatensor.split(tensor, axis="samples", grouped_labels=["a", "b", "c"])

    grouped_labels = [
        Labels(names=["red"], values=np.array([[0], [6], [7]])),
        Labels(names=["red"], values=np.array([[2], [3], [4]])),
        Labels(names=["wine"], values=np.array([[1], [5], [8], [9]])),
    ]

    message = (
        "the dimensions names of all Labels in `grouped_labels` must be the same, "
        "got \\['red'\\] and \\['wine'\\]"
    )
    with pytest.raises(ValueError, match=message):
        metatensor.split(tensor, axis="samples", grouped_labels=grouped_labels)

    grouped_labels = [
        Labels(names=["missing", "atom"], values=np.array([[0, 1], [6, 7], [7, 4]])),
        Labels(names=["missing", "atom"], values=np.array([[2, 4], [3, 3], [4, 7]])),
        Labels(
            names=["missing", "atom"],
            values=np.array([[1, 5], [5, 3], [8, 10]]),
        ),
    ]

    message = (
        "the 'missing' dimension name in `grouped_labels` is not part of "
        "the samples names of the input tensor"
    )
    with pytest.raises(ValueError, match=message):
        metatensor.split(tensor, axis="samples", grouped_labels=grouped_labels)

    message = (
        "`block` must be a metatensor TensorBlock, "
        "not <class 'metatensor.tensor.TensorMap'>"
    )
    with pytest.raises(TypeError, match=message):
        metatensor.split_block(tensor, axis="samples", grouped_labels=[])
