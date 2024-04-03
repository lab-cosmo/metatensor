import os

import pytest

import metatensor


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_empty_like():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )
    empty_tensor = metatensor.empty_like(tensor)
    empty_tensor_positions = metatensor.empty_like(tensor, gradients="positions")

    assert metatensor.equal_metadata(empty_tensor, tensor)

    tensor_no_strain = metatensor.remove_gradients(tensor, "strain")
    assert metatensor.equal_metadata(empty_tensor_positions, tensor_no_strain)


def test_empty_like_error():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )

    message = "requested gradient 'err' in 'empty_like' is not defined in this tensor"
    with pytest.raises(ValueError, match=message):
        tensor = metatensor.empty_like(tensor, gradients=["positions", "err"])
