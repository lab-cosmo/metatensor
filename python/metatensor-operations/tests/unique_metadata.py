import os

import numpy as np
import pytest

import metatensor
from metatensor import Labels

from . import utils


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
TEST_FILE = "qm7-spherical-expansion.npz"


@pytest.fixture
def tensor():
    return utils.tensor()


@pytest.fixture
def large_tensor():
    return utils.large_tensor()


@pytest.fixture
def real_tensor():
    return metatensor.load(os.path.join(DATA_ROOT, TEST_FILE), use_numpy=True)


def test_unique_metadata_block(large_tensor):
    # unique metadata along sample axis
    target_samples = Labels(names=["s"], values=np.array([0, 1, 3]).reshape(-1, 1))
    actual_samples = metatensor.unique_metadata_block(
        large_tensor.block(1),
        axis="samples",
        names="s",
    )
    assert target_samples == actual_samples

    # unique metadata of gradient along sample axis
    names = ["sample", "g"]
    target_samples = Labels(names=names, values=np.array([[0, -2], [0, 3], [2, -2]]))
    actual_samples = metatensor.unique_metadata_block(
        large_tensor.block(1),
        axis="samples",
        names=names,
        gradient="g",
    )
    assert target_samples == actual_samples

    # unique metadata along properties axis
    properties = [3, 4, 5]
    target_properties = Labels(names=["p"], values=np.array([[p] for p in properties]))
    actual_properties = metatensor.unique_metadata_block(
        large_tensor.block(1), axis="properties", names="p"
    )
    assert target_properties == actual_properties

    # unique metadata of gradient along properties axis
    names = ["p"]
    target_properties = Labels(names=names, values=np.array([[3], [4], [5]]))
    actual_properties = metatensor.unique_metadata_block(
        large_tensor.block(1),
        axis="properties",
        names=names,
        gradient="g",
    )
    assert target_properties == actual_properties


def test_empty_block(real_tensor):
    target_samples = Labels(names=["system"], values=np.empty((0, 1)))
    # slice block to be empty
    sliced_block = metatensor.slice_block(
        real_tensor.block(0),
        axis="samples",
        labels=Labels(names=["system"], values=np.array([[-1]])),
    )
    actual_samples = metatensor.unique_metadata_block(
        sliced_block,
        axis="samples",
        names="system",
    )
    assert target_samples == actual_samples
    assert len(actual_samples) == 0

    target_properties = Labels(names=["n"], values=np.empty((0, 1)))
    sliced_block = metatensor.slice_block(
        real_tensor.block(0),
        axis="properties",
        labels=Labels(names=["n"], values=np.array([[-1]])),
    )
    actual_properties = metatensor.unique_metadata_block(
        sliced_block,
        axis="properties",
        names="n",
    )
    assert target_properties == actual_properties
    assert len(actual_properties) == 0


def test_unique_metadata(tensor, large_tensor):
    # unique metadata along samples
    target_samples = Labels(
        names=["s"],
        values=np.array([0, 1, 2, 3, 4, 5, 6, 8]).reshape(-1, 1),
    )
    actual_samples = metatensor.unique_metadata(tensor, "samples", "s")
    assert target_samples == actual_samples

    actual_samples = metatensor.unique_metadata(large_tensor, "samples", "s")
    assert actual_samples == target_samples

    # unique metadata along samples for gradients
    names = ["sample", "g"]
    target_samples = Labels(
        names=names,
        values=np.array([[0, -2], [0, 1], [0, 3], [1, -2], [2, -2], [2, 3], [3, 3]]),
    )
    actual_samples = metatensor.unique_metadata(
        tensor,
        axis="samples",
        names=names,
        gradient="g",
    )
    assert actual_samples == target_samples

    # unique metadata along properties
    target_properties = Labels(
        names=["p"], values=np.array([0, 3, 4, 5]).reshape(-1, 1)
    )
    actual_properties = metatensor.unique_metadata(
        tensor,
        axis="properties",
        names=["p"],  # names passed as list
    )
    assert target_properties == actual_properties

    target_properties = Labels(
        names=["p"], values=np.array([0, 1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
    )
    actual_properties = metatensor.unique_metadata(
        large_tensor,
        axis="properties",
        names=("p",),  # names passed as tuple
    )
    assert target_properties == actual_properties

    names = ["p"]
    target_properties = Labels(
        names=names,
        values=np.array([0, 3, 4, 5]).reshape(-1, 1),
    )
    actual_properties = metatensor.unique_metadata(
        tensor, axis="properties", names=names, gradient="g"
    )
    assert target_properties == actual_properties


def test_unique_metadata_block_errors(real_tensor):
    message = (
        "`block` must be a metatensor TensorBlock, "
        "not <class 'metatensor.tensor.TensorMap'>"
    )
    with pytest.raises(TypeError, match=message):
        metatensor.unique_metadata_block(real_tensor, "samples", ["system"])

    message = (
        "`tensor` must be a metatensor TensorMap, "
        "not <class 'metatensor.block.TensorBlock'>"
    )
    with pytest.raises(TypeError, match=message):
        metatensor.unique_metadata(real_tensor.block(0), "samples", ["system"])

    message = "`axis` must be a string, not <class 'float'>"
    with pytest.raises(TypeError, match=message):
        metatensor.unique_metadata_block(
            real_tensor.block(0),
            axis=3.14,
            names=["system"],
        )

    message = "`names` must be a list of strings, not <class 'float'>"
    with pytest.raises(TypeError, match=message):
        metatensor.unique_metadata_block(
            real_tensor.block(0),
            axis="properties",
            names=3.14,
        )

    message = "`names` elements must be a strings, not <class 'float'>"
    with pytest.raises(TypeError, match=message):
        metatensor.unique_metadata_block(
            real_tensor.block(0),
            axis="properties",
            names=["system", 3.14],
        )

    message = "`gradient` must be a string, not <class 'float'>"
    with pytest.raises(TypeError, match=message):
        metatensor.unique_metadata_block(
            real_tensor.block(0),
            axis="properties",
            names=["system"],
            gradient=3.14,
        )

    message = "`axis` must be either 'samples' or 'properties', not 'ciao'"
    with pytest.raises(ValueError, match=message):
        metatensor.unique_metadata_block(
            real_tensor.block(0),
            axis="ciao",
            names=["system"],
        )

    message = "'ciao' not found in the dimensions of these Labels"
    with pytest.raises(ValueError, match=message):
        metatensor.unique_metadata(real_tensor, axis="samples", names=["ciao"])
