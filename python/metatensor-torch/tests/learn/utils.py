import torch

from metatensor.torch import Labels, TensorBlock, TensorMap


TORCH_KWARGS = {"device": "cpu", "dtype": torch.float32}


def random_single_block_no_components_tensor_map():
    block = TensorBlock(
        values=torch.rand(4, 2, **TORCH_KWARGS),
        samples=Labels(
            ["sample", "system"],
            torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]]),
        ),
        components=[],
        properties=Labels(["properties"], torch.tensor([[0], [1]])),
    )
    positions_gradient = TensorBlock(
        values=torch.rand(7, 3, 2, **TORCH_KWARGS),
        samples=Labels(
            ["sample", "system", "atom"],
            torch.tensor(
                [
                    [0, 0, 1],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 2],
                    [2, 2, 0],
                    [3, 3, 0],
                ],
            ),
        ),
        components=[Labels(["direction"], torch.tensor([[0], [1], [2]]))],
        properties=block.properties,
    )
    block.add_gradient("positions", positions_gradient)

    cell_gradient = TensorBlock(
        values=torch.rand(4, 6, 2, **TORCH_KWARGS),
        samples=Labels(
            ["sample"],
            torch.tensor([[0], [1], [2], [3]]),
        ),
        components=[
            Labels(
                ["voigt_index"],
                torch.tensor([[0], [1], [2], [3], [4], [5]]),
            )
        ],
        properties=block.properties,
    )
    block.add_gradient("cell", cell_gradient)

    return TensorMap(Labels.single(), [block])
