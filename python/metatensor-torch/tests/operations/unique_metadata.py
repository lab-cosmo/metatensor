import io

import torch
from packaging import version

import metatensor.torch

from ._data import load_data


def test_unique_metadata():
    tensor = load_data("qm7-power-spectrum.npz")

    unique_labels = metatensor.torch.unique_metadata(
        tensor,
        axis="samples",
        names=["system"],
    )

    # check type
    assert isinstance(unique_labels, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert unique_labels._type().name() == "Labels"

    # check label names
    assert unique_labels.names == ["system"]

    # repeat with gradients
    unique_labels = metatensor.torch.unique_metadata(
        tensor,
        axis="samples",
        names=["atom"],
        gradient="positions",
    )

    assert isinstance(unique_labels, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert unique_labels._type().name() == "Labels"

    assert unique_labels.names == ["atom"]


def test_unique_metadata_block():
    tensor = load_data("qm7-power-spectrum.npz")
    block = tensor.block(0)

    unique_labels = metatensor.torch.unique_metadata_block(
        block,
        axis="samples",
        names=["system"],
    )

    # check type
    assert isinstance(unique_labels, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert unique_labels._type().name() == "Labels"

    # check label names
    assert unique_labels.names == ["system"]

    # repeat with gradients
    unique_labels = metatensor.torch.unique_metadata_block(
        block,
        axis="samples",
        names=["atom"],
        gradient="positions",
    )

    assert isinstance(unique_labels, torch.ScriptObject)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert unique_labels._type().name() == "Labels"

    assert unique_labels.names == ["atom"]


def test_save_load():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.unique_metadata, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
