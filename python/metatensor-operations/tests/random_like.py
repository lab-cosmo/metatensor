import os

import numpy as np
import pytest

import metatensor


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_random_uniform_like():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )
    random_tensor = metatensor.random_uniform_like(tensor)
    random_tensor_positions = metatensor.random_uniform_like(
        tensor, gradients="positions"
    )

    assert metatensor.equal_metadata(random_tensor, tensor)

    tensor_no_strain = metatensor.remove_gradients(tensor, "strain")
    assert metatensor.equal_metadata(random_tensor_positions, tensor_no_strain)

    # check the values
    for random_block in random_tensor:
        assert np.all(random_block.values >= 0)
        assert np.all(random_block.values < 1)

        for _, random_gradient in random_block.gradients():
            assert np.all(random_gradient.values >= 0)
            assert np.all(random_gradient.values < 1)


def test_random_uniform_like_error():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )

    message = (
        "requested gradient 'err' in 'random_uniform_like' is not defined "
        "in this tensor"
    )
    with pytest.raises(ValueError, match=message):
        tensor = metatensor.random_uniform_like(tensor, gradients=["positions", "err"])
