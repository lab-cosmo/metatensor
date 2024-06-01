"""
Module containing the :py:class:`AbsoluteError` and :py:class:`SquaredError` classes.
"""

import torch

from .._backend import TensorBlock, TensorMap, equal_metadata, equal_metadata_block


def absolute_error(A: TensorMap, B: TensorMap) -> TensorMap:
    if not equal_metadata(A, B):
        raise ValueError(
            "The two maps must have the same metadata in `absolute_error`."
        )

    keys = []
    blocks = []
    for key, block_A in A.items():
        block_B = B.block(key)
        keys.append(key)
        blocks.append(absolute_error_block(block_A, block_B))

    return TensorMap(keys, blocks)


def absolute_error_block(A: TensorBlock, B: TensorBlock) -> TensorBlock:
    if not equal_metadata_block(A, B):
        raise ValueError(
            "The two blocks must have the same metadata in `absolute_error_block`."
        )

    values = torch.abs(A.values - B.values)
    block = TensorBlock(
        values=values,
        samples=A.samples,
        components=A.components,
        properties=A.properties,
    )
    for gradient_name, gradient_A in A.gradients():
        gradient_B = B.gradient(gradient_name)
        block.add_gradient(
            gradient_name,
            absolute_error_block(gradient_A, gradient_B),
        )

    return block


def squared_error(A: TensorMap, B: TensorMap) -> TensorMap:
    if not equal_metadata(A, B):
        raise ValueError("The two maps must have the same metadata in `squared_error`.")

    keys = []
    blocks = []
    for key, block_A in A.items():
        block_B = B.block(key)
        keys.append(key)
        blocks.append(squared_error_block(block_A, block_B))

    return TensorMap(keys, blocks)


def squared_error_block(A: TensorBlock, B: TensorBlock) -> TensorBlock:
    if not equal_metadata_block(A, B):
        raise ValueError(
            "The two blocks must have the same metadata in `squared_error_block`."
        )

    values = (A.values - B.values) ** 2
    block = TensorBlock(
        values=values,
        samples=A.samples,
        components=A.components,
        properties=A.properties,
    )
    for gradient_name, gradient_A in A.gradients():
        gradient_B = B.gradient(gradient_name)
        block.add_gradient(
            gradient_name,
            squared_error_block(gradient_A, gradient_B),
        )

    return block
