"""
Module to transpose a :py:class:`TensorMap` or :py:class:`TensorBlock`.
"""
import numpy as np

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap
from . import _dispatch


def transpose(tensor: TensorMap) -> TensorMap:
    """
    Transposes a :py:class:`TensorMap`.
    """
    # Check input
    if not isinstance(tensor, TensorMap):
        raise TypeError(f"Expected a TensorMap, got {type(tensor)}.")
    # Transpose TensorMap
    keys = tensor.keys
    return TensorMap(keys, [transpose_block(tensor[key]) for key in keys])


def transpose_block(block: TensorBlock) -> TensorBlock:
    """
    Transposes a :py:class:`TensorBlock`.
    """
    # Check input
    if not isinstance(block, TensorBlock):
        raise TypeError(f"Expected a TensorBlock, got {type(block)}.")
    # Define the new samples and components to use for the output block. The new
    # samples are just the original blocks's properties and the new components
    # are just the originals, reversed.
    new_samples = block.properties
    new_components = block.components[::-1]

    # Define the shape that is common to this block and all of its
    # gradients. This is the size of all components, and the properties,
    # i.e. (c1, c2, ..., cn, p), but not the samples
    common_shape = block.values.shape[1:]

    # Transpose the values and store in an array
    transposed_block_values = _dispatch.transpose(block.values)

    # Define number of properties in the transposed block. This will be
    # increased by the length of the new properties of each gradient and used as
    # a numeric index for the properties in the returned block.
    q_max = transposed_block_values.shape[-1]

    # Iterate over the gradients. Reshape, transpose and store the gradient data
    arrays_to_join = [transposed_block_values]
    for _, gradient in block.gradients():
        # Reshape the gradient data such that the gradient specific samples are
        # moved to the samples dimension, but the components common with the
        # associated values block are kept in the components dimension. For
        # instance, if the gradient has shape (100, 3, 3, 5, 6, 400), where (3,
        # 3) are the gradient-specific components, the reshaped gradient should
        # have shape (900, 5, 6, 400).
        n_grad_samples = np.prod(gradient.data.shape[: -len(common_shape)])
        reshaped_grad_data = gradient.data.reshape(n_grad_samples, *common_shape)

        # Transpose the gradient data and store. For instance, (900, 5, 6, 400)
        # becomes (400, 6, 5, 900).
        tranposed_grad_data = _dispatch.transpose(reshaped_grad_data)
        arrays_to_join.append(tranposed_grad_data)

        # Update q_max for the next gradient
        q_max += tranposed_grad_data.shape[-1]

    # Concatenate block values and gradients data tensors along the properties
    # (final) axis.
    joined_values = _dispatch.concatenate(
        arrays_to_join, axis=len(arrays_to_join[0].shape) - 1
    )

    return TensorBlock(
        values=joined_values,
        samples=new_samples,
        components=new_components,
        properties=Labels(
            names=("q",),
            values=np.arange(q_max).reshape(-1, 1),
        ),
    )
