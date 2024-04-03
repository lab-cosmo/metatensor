import io

import torch

import metatensor.torch


def test_one_hot():
    original_labels = metatensor.torch.Labels(
        names=["atom", "type"],
        values=torch.tensor([[0, 6], [1, 1], [4, 6]]),
    )
    possible_labels = metatensor.torch.Labels("type", torch.tensor([[1], [6]]))
    correct_encoding = torch.tensor([[0, 1], [1, 0], [0, 1]])
    one_hot_encoding = metatensor.torch.one_hot(original_labels, possible_labels)

    assert isinstance(one_hot_encoding, torch.Tensor)
    assert torch.all(one_hot_encoding == correct_encoding)


def test_save():
    with io.BytesIO() as buffer:
        torch.jit.save(metatensor.torch.one_hot, buffer)
        buffer.seek(0)
        torch.jit.load(buffer)
