"""Utilities for testing metatensor-learn."""

import os
from functools import partial

import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


torch = pytest.importorskip("torch")

from metatensor.learn.data import Dataset, IndexedDataset  # noqa E402


TORCH_KWARGS = {"device": "cpu", "dtype": torch.float32}


def random_single_block_no_components_tensor_map(use_torch):
    """Create a dummy tensor map to be used in tests."""

    if use_torch:
        create_random_array = partial(torch.rand, **TORCH_KWARGS)
    else:
        create_random_array = np.random.rand

    block = TensorBlock(
        values=create_random_array(4, 2),
        samples=Labels(
            ["sample", "system"],
            np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32),
        ),
        components=[],
        properties=Labels(["properties"], np.array([[0], [1]], dtype=np.int32)),
    )
    positions_gradient = TensorBlock(
        values=create_random_array(7, 3, 2),
        samples=Labels(
            ["sample", "system", "atom"],
            np.array(
                [
                    [0, 0, 1],
                    [0, 0, 2],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 1, 2],
                    [2, 2, 0],
                    [3, 3, 0],
                ],
                dtype=np.int32,
            ),
        ),
        components=[Labels(["direction"], np.array([[0], [1], [2]], dtype=np.int32))],
        properties=block.properties,
    )
    block.add_gradient("positions", positions_gradient)

    cell_gradient = TensorBlock(
        values=create_random_array(4, 6, 2),
        samples=Labels(
            ["sample", "system"],
            np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32),
        ),
        components=[
            Labels(
                ["voigt_index"],
                np.array([[0], [1], [2], [3], [4], [5]], dtype=np.int32),
            )
        ],
        properties=block.properties,
    )
    block.add_gradient("cell", cell_gradient)

    return TensorMap(Labels.single(), [block])


def tensor(sample_indices):
    """A dummy tensor map to be used in tests"""
    n_blocks = 20
    blocks = []
    for _ in range(n_blocks):
        block = TensorBlock(
            values=np.full((len(sample_indices), 1, 1), 1.0),
            samples=Labels(
                ["sample_index", "foo"], np.array([[s, 0] for s in sample_indices])
            ),
            components=[Labels(["c"], np.array([[0]]))],
            properties=Labels(["p"], np.array([[0]])),
        )
        block.add_gradient(
            parameter="g",
            gradient=TensorBlock(
                samples=Labels(["sample", "g"], np.array([[0, -2], [2, 3]])),
                values=np.full((2, 1, 1), 11.0),
                components=block.components,
                properties=block.properties,
            ),
        )
        blocks.append(block)

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array([[0, 2 * b] for b in range(n_blocks)]),
    )

    return TensorMap(keys, blocks)


def generate_data(sample_indices):
    """
    Generates data sliced according to sample index "sample". Saves them to disk
    and returns them in memory along with the sample indices.

    The data items are: input (rascaline SphericalExpansion), output (ones like
    input), and auxiliary (zeros like input). Each is sliced to a structure
    index A = {0, ..., 99}
    """
    input = tensor(sample_indices)
    output = metatensor.ones_like(input)
    auxiliary = metatensor.zeros_like(input)

    # Slice to per-structure TensorMaps
    inputs, outputs, auxiliaries = [], [], []
    for A in sample_indices:
        input_A = metatensor.slice(
            input,
            "samples",
            labels=Labels(names=["sample_index"], values=np.array([A]).reshape(-1, 1)),
        )
        output_A = metatensor.slice(
            output,
            "samples",
            labels=Labels(names=["sample_index"], values=np.array([A]).reshape(-1, 1)),
        )
        auxiliary_A = metatensor.slice(
            auxiliary,
            "samples",
            labels=Labels(names=["sample_index"], values=np.array([A]).reshape(-1, 1)),
        )
        # Store in memory
        inputs.append(input_A)
        outputs.append(output_A)
        auxiliaries.append(auxiliary_A)

        # Save to disk
        if not os.path.exists(f"{A}"):
            os.mkdir(f"{A}")
        metatensor.save(f"{A}/input.npz", input_A)
        metatensor.save(f"{A}/output.npz", output_A)
        metatensor.save(f"{A}/auxiliary.npz", auxiliary_A)

    return inputs, outputs, auxiliaries


def transform(sample_index: int, filename: str):
    """
    Loads a TensorMap for a given sample indexed by `sample_index` from disk and
    converts it to a torch tensor.
    """
    path = os.path.join(f"{sample_index}/{filename}.npz")

    tensor = metatensor.io.load_custom_array(
        path, create_array=metatensor.io.create_torch_array
    )
    return tensor.to(**TORCH_KWARGS)


def dataset_in_mem(sample_indices):
    """Create a dataset with everything in memory"""
    inputs, outputs, auxiliaries = generate_data(range(len(sample_indices)))
    return Dataset(
        size=len(sample_indices),
        input=inputs,
        output=outputs,
        auxiliary=auxiliaries,
    )


def dataset_on_disk(sample_indices):
    """Create a dataset with everything on disk"""
    _ = generate_data(range(len(sample_indices)))
    return Dataset(
        size=len(sample_indices),
        input=partial(transform, filename="input"),
        output=partial(transform, filename="output"),
        auxiliary=partial(transform, filename="auxiliary"),
    )


def dataset_mixed_mem_disk(sample_indices):
    """Create a dataset with the inputs and outputs in memory, but auxiliaries
    on disk"""
    inputs, outputs, _ = generate_data(range(len(sample_indices)))
    return Dataset(
        size=len(sample_indices),
        input=inputs,
        output=outputs,
        auxiliary=partial(transform, filename="auxiliary"),
    )


def indexed_dataset_in_mem(sample_indices):
    """Create an indexed dataset with everything in memory"""
    inputs, outputs, auxiliaries = generate_data(sample_indices)
    return IndexedDataset(
        sample_id=sample_indices,
        input=inputs,
        output=outputs,
        auxiliary=auxiliaries,
    )


def indexed_dataset_on_disk(sample_indices):
    """Create an indexed dataset with everything on disk"""
    _ = generate_data(sample_indices)
    return IndexedDataset(
        sample_id=sample_indices,
        input=partial(transform, filename="input"),
        output=partial(transform, filename="output"),
        auxiliary=partial(transform, filename="auxiliary"),
    )


def indexed_dataset_mixed_mem_disk(sample_indices):
    """
    Create an indexed dataset with the inputs and outputs in memory, but auxiliaries
    on disk
    """
    inputs, outputs, _ = generate_data(sample_indices)
    return IndexedDataset(
        sample_id=sample_indices,
        input=inputs,
        output=outputs,
        auxiliary=partial(transform, filename="auxiliary"),
    )
