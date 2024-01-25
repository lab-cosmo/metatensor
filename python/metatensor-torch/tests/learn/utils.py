from functools import partial

import pytest


torch = pytest.importorskip("torch")

TORCH_KWARGS = {"device": "cpu", "dtype": torch.float32}


def random_single_block_no_components_tensor_map(use_torch, use_metatensor_torch):
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    if not use_torch and use_metatensor_torch:
        raise ValueError(
            "torch.TensorMap cannot be created without torch.Tensor block values."
        )
    if use_metatensor_torch:
        import torch

        from metatensor.torch import Labels, TensorBlock, TensorMap

        create_int32_array = partial(torch.tensor, dtype=torch.int32)
    else:
        import numpy as np

        from metatensor import Labels, TensorBlock, TensorMap

        create_int32_array = partial(np.array, dtype=np.int32)

    if use_torch:
        import torch

        create_random_array = partial(torch.rand, **TORCH_KWARGS)
    else:
        import numpy as np

        create_random_array = np.random.rand

    block_1 = TensorBlock(
        values=create_random_array(4, 2),
        samples=Labels(
            ["sample", "structure"],
            create_int32_array([[0, 0], [1, 1], [2, 2], [3, 3]]),
        ),
        components=[],
        properties=Labels(["properties"], create_int32_array([[0], [1]])),
    )
    positions_gradient = TensorBlock(
        values=create_random_array(7, 3, 2),
        samples=Labels(
            ["sample", "structure", "center"],
            create_int32_array(
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
        components=[Labels(["direction"], create_int32_array([[0], [1], [2]]))],
        properties=block_1.properties,
    )
    block_1.add_gradient("positions", positions_gradient)

    cell_gradient = TensorBlock(
        values=create_random_array(4, 6, 2),
        samples=Labels(
            ["sample", "structure"],
            create_int32_array([[0, 0], [1, 1], [2, 2], [3, 3]]),
        ),
        components=[
            Labels(
                ["voigt_index"],
                create_int32_array([[0], [1], [2], [3], [4], [5]]),
            )
        ],
        properties=block_1.properties,
    )
    block_1.add_gradient("cell", cell_gradient)

    return TensorMap(Labels.single(), [block_1])


@pytest.fixture(scope="class")
def single_block_tensor():
    return random_single_block_no_components_tensor_map(True, True)
