import os

import numpy as np
import pytest

import metatensor


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_zeros_like():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )
    zeros_tensor = metatensor.zeros_like(tensor)
    zeros_tensor_positions = metatensor.zeros_like(tensor, gradients="positions")

    assert metatensor.equal_metadata(zeros_tensor, tensor)

    tensor_no_strain = metatensor.remove_gradients(tensor, "strain")
    assert metatensor.equal_metadata(zeros_tensor_positions, tensor_no_strain)

    # check the values
    for key, block in tensor.items():
        zeros_block = zeros_tensor[key]

        assert np.all(zeros_block.values == np.zeros_like(block.values))

        for parameter, gradient in block.gradients():
            zeros_gradient = zeros_block.gradient(parameter)
            assert np.all(zeros_gradient.values == np.zeros_like(gradient.values))


def test_zeros_like_error():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )

    message = "requested gradient 'err' in 'zeros_like' is not defined in this tensor"
    with pytest.raises(ValueError, match=message):
        tensor = metatensor.zeros_like(tensor, gradients=["positions", "err"])
