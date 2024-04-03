import torch

from metatensor.torch import Labels, TensorBlock, TensorMap


def tensor(dtype=torch.float32, device="cpu"):
    """A dummy tensor map to be used in tests"""
    block_1 = TensorBlock(
        values=torch.full((3, 1, 1), 1.0, dtype=dtype, device=device),
        samples=Labels(["s"], torch.tensor([[0], [2], [4]])),
        components=[Labels(["c"], torch.tensor([[0]]))],
        properties=Labels(["p"], torch.tensor([[0]])),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            samples=Labels(["sample", "g"], torch.tensor([[0, -2], [2, 3]])),
            values=torch.full((2, 1, 1), 11.0, dtype=dtype, device=device),
            components=block_1.components,
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=torch.full((3, 1, 3), 2.0, dtype=dtype, device=device),
        samples=Labels(["s"], torch.tensor([[0], [1], [3]])),
        components=[Labels(["c"], torch.tensor([[0]]))],
        properties=Labels(["p"], torch.tensor([[3], [4], [5]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.full((3, 1, 3), 12.0, dtype=dtype, device=device),
            samples=Labels(["sample", "g"], torch.tensor([[0, -2], [0, 3], [2, -2]])),
            components=block_2.components,
            properties=block_2.properties,
        ),
    )

    block_3 = TensorBlock(
        values=torch.full((4, 3, 1), 3.0, dtype=dtype, device=device),
        samples=Labels(["s"], torch.tensor([[0], [3], [6], [8]])),
        components=[Labels(["c"], torch.tensor([[0], [1], [2]]))],
        properties=Labels(["p"], torch.tensor([[0]])),
    )
    block_3.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.full((1, 3, 1), 13.0, dtype=dtype, device=device),
            samples=Labels(["sample", "g"], torch.tensor([[1, -2]])),
            components=block_3.components,
            properties=block_3.properties,
        ),
    )

    block_4 = TensorBlock(
        values=torch.full((4, 3, 1), 4.0, dtype=dtype, device=device),
        samples=Labels(["s"], torch.tensor([[0], [1], [2], [5]])),
        components=[Labels(["c"], torch.tensor([[0], [1], [2]]))],
        properties=Labels(["p"], torch.tensor([[0]])),
    )
    block_4.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.full((2, 3, 1), 14.0, dtype=dtype, device=device),
            samples=Labels(["sample", "g"], torch.tensor([[0, 1], [3, 3]])),
            components=block_4.components,
            properties=block_4.properties,
        ),
    )

    keys = Labels(
        names=["key_1", "key_2"],
        values=torch.tensor([[0, 0], [1, 0], [2, 2], [2, 3]]),
    )

    return TensorMap(keys, [block_1, block_2, block_3, block_4])


def large_tensor(dtype=torch.float32, device="cpu"):
    t = tensor(dtype=dtype, device=device)
    blocks = [block.copy() for _, block in t.items()]

    for i in range(8):
        block = TensorBlock(
            values=torch.full((4, 3, 1), 4.0, dtype=dtype, device=device),
            samples=Labels(["s"], torch.tensor([[0], [1], [4], [5]])),
            components=[Labels(["c"], torch.tensor([[0], [1], [2]]))],
            properties=Labels(["p"], torch.tensor([[i]])),
        )
        block.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                values=torch.full((2, 3, 1), 14.0, dtype=dtype, device=device),
                samples=Labels(["sample", "g"], torch.tensor([[0, 1], [3, 3]])),
                components=block.components,
                properties=block.properties,
            ),
        )
        blocks.append(block)

    keys = Labels(
        names=["key_1", "key_2"],
        values=torch.tensor(
            [
                [0, 0],
                [1, 0],
                [2, 2],
                [2, 3],
                [0, 4],
                [1, 4],
                [2, 4],
                [3, 4],
                [0, 5],
                [1, 5],
                [2, 5],
                [3, 5],
            ],
        ),
    )
    return TensorMap(keys, blocks)


def tensor_zero_len_block():
    """
    A dummy TensorMap with a single block whose samples axis length is zero.
    """
    return TensorMap(
        keys=Labels.single(),
        blocks=[
            TensorBlock(
                values=torch.zeros((0, 1), dtype=torch.float64),
                samples=Labels(
                    names=["_"], values=torch.zeros((0, 1), dtype=torch.int32)
                ),
                components=[],
                properties=Labels.single(),
            )
        ],
    )
