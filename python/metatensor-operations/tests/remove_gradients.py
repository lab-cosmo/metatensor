import os

import metatensor


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_remove_everything():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )

    assert set(tensor.block(0).gradients_list()) == set(["strain", "positions"])

    tensor = metatensor.remove_gradients(tensor)
    assert tensor.block(0).gradients_list() == []


def test_remove_subset():
    tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )

    assert set(tensor.block(0).gradients_list()) == set(["strain", "positions"])

    tensor = metatensor.remove_gradients(tensor, ["positions"])
    assert tensor.block(0).gradients_list() == ["strain"]
