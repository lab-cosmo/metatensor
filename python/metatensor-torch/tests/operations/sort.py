import io

import pytest
import torch

import metatensor.torch
from metatensor.torch import Labels, TensorBlock, TensorMap


@pytest.fixture
def tensor():
    # samples are descending, components and properties are ascending
    block_1 = TensorBlock(
        values=torch.tensor([[3, 5], [1, 2]]),
        samples=Labels(["s"], torch.tensor([[2], [0]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[0], [1]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.tensor([[[8, 3], [9, 4]], [[6, 1], [7, 2]]]),
            samples=Labels(["sample", "g"], torch.tensor([[1, 1], [0, 1]])),
            components=[
                Labels(["c"], torch.tensor([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )

    # samples are disordered, components are ascending, properties are descending
    block_2 = TensorBlock(
        values=torch.tensor([[3, 4], [5, 6], [1, 2]]),
        samples=Labels(["s"], torch.tensor([[7], [0], [2]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[1], [0]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.tensor(
                [[[15, 14], [11, 10]], [[13, 12], [15, 14]], [[11, 10], [13, 12]]]
            ),
            samples=Labels(
                ["sample", "g"],
                torch.tensor([[1, 1], [2, 1], [0, 1]]),
            ),
            components=[
                Labels(["c"], torch.tensor([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )
    keys = Labels(names=["key_1", "key_2"], values=torch.tensor([[1, 0], [0, 0]]))
    # block order is descending
    return TensorMap(keys, [block_2, block_1])


@pytest.fixture
def tensor_sorted_ascending():
    """
    This is the `tensor` fixture sorted in ascending order how it should be returned
    when applying metatensor.operations.sort with `descending=False` option.
    """
    block_1 = TensorBlock(
        values=torch.tensor([[1, 2], [3, 5]]),
        samples=Labels(["s"], torch.tensor([[0], [2]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[0], [1]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.tensor([[[8, 3], [9, 4]], [[6, 1], [7, 2]]]),
            samples=Labels(["sample", "g"], torch.tensor([[0, 1], [1, 1]])),
            components=[
                Labels(["c"], torch.tensor([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )
    block_2 = TensorBlock(
        values=torch.tensor([[6, 5], [2, 1], [4, 3]]),
        samples=Labels(["s"], torch.tensor([[0], [2], [7]])),
        components=[],
        properties=Labels(["p"], torch.tensor([[0], [1]])),
    )

    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.tensor(
                [[[14, 15], [10, 11]], [[12, 13], [14, 15]], [[10, 11], [12, 13]]]
            ),
            samples=Labels(
                ["sample", "g"],
                torch.tensor([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels(["c"], torch.tensor([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )

    keys = Labels(names=["key_1", "key_2"], values=torch.tensor([[0, 0], [1, 0]]))
    return TensorMap(keys, [block_1, block_2])


def check_operation(operation, tensor, tensor_ref):
    metatensor.torch.allclose_raise(operation(tensor), tensor_ref)


def test_operation_as_torch_script(tensor, tensor_sorted_ascending):
    check_operation(
        torch.jit.script(metatensor.torch.sort), tensor, tensor_sorted_ascending
    )


def test_operation_as_python(tensor, tensor_sorted_ascending):
    check_operation(metatensor.torch.sort, tensor, tensor_sorted_ascending)


def test_save():
    scripted = torch.jit.script(metatensor.torch.sort)
    buffer = io.BytesIO()
    torch.jit.save(scripted, buffer)
    buffer.seek(0)
    torch.jit.load(buffer)
    buffer.close()
