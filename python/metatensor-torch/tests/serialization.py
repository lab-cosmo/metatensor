import os

import numpy as np
import torch

import metatensor.torch

from . import utils


def check_tensor(tensor):
    assert tensor.keys.names == [
        "spherical_harmonics_l",
        "center_species",
        "neighbor_species",
    ]
    assert len(tensor.keys) == 27

    block = tensor.block(
        dict(spherical_harmonics_l=2, center_species=6, neighbor_species=1)
    )
    assert block.samples.names == ["structure", "center"]
    assert block.values.shape == (9, 5, 3)

    gradient = block.gradient("positions")
    assert gradient.samples.names == ["sample", "structure", "atom"]
    assert gradient.values.shape == (59, 3, 5, 3)


def test_load():
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor",
        "tests",
        "data.npz",
    )

    loaded = metatensor.torch.load(path)
    check_tensor(loaded)

    loaded = metatensor.torch.TensorMap.load(path)
    check_tensor(loaded)


def test_load_buffer():
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor",
        "tests",
        "data.npz",
    )

    buffer = torch.tensor(np.fromfile(path, dtype="uint8"))

    loaded = metatensor.torch.load_buffer(buffer)
    check_tensor(loaded)

    loaded = metatensor.torch.TensorMap.load_buffer(buffer)
    check_tensor(loaded)


def test_save(tmpdir):
    """Check that we can save and load a tensor to a file"""
    tmpfile = "serialize-test.npz"

    tensor = utils.tensor(dtype=torch.float64)

    with tmpdir.as_cwd():
        metatensor.torch.save(tmpfile, tensor)
        data = metatensor.torch.load(tmpfile)

        assert len(data.keys) == 4

        tensor.save(tmpfile)
        data = metatensor.torch.load(tmpfile)

        assert len(data.keys) == 4


def test_save_buffer():
    """Check that we can save and load a tensor to an in-memory buffer"""
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor",
        "tests",
        "data.npz",
    )

    buffer = torch.tensor(np.fromfile(path, dtype="uint8"))
    tensor = metatensor.torch.load_buffer(buffer)

    saved = metatensor.torch.save_buffer(tensor)
    assert torch.all(buffer == saved)

    saved = tensor.save_buffer()
    assert torch.all(buffer == saved)


def test_pickle(tmpdir):
    tensor = metatensor.torch.load(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "metatensor",
            "tests",
            "data.npz",
        ),
    )
    tmpfile = "serialize-test.npz"

    with tmpdir.as_cwd():
        torch.save(tensor, tmpfile)
        loaded = torch.load(tmpfile)

    check_tensor(loaded)


def test_save_load_zero_length_block(tmpdir):
    """
    Tests that attempting to save and load a TensorMap with a zero-length axis block
    does not raise an error.
    """
    tensor_zero_len_block = utils.tensor_zero_len_block()

    with tmpdir.as_cwd():
        file = "serialize-test-zero-len-block.npz"
        metatensor.torch.save(file, tensor_zero_len_block)
        metatensor.torch.load(file)
