"""
Module for finding unique metadata for TensorMaps and TensorBlocks
"""
from typing import List, Optional, Tuple, Union

import numpy as np

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap


def unique_metadata(
    tensor: TensorMap,
    axis: str,
    names: Union[List[str], Tuple[str], str],
    gradient_param: Optional[str] = None,
) -> Labels:
    """
    For a given ``axis`` (either "samples" or "properties"), and for the given
    samples/proeprties ``names``, returns a :py:class:`Labels` object of the
    unique metadata in the input :py:class:`TensorMap` ``tensor``.

    If there are no indices in ``tensor`` corresponding to the specified
    ``axis`` and ``names``, an empty Labels object with the correct names as in
    the passed ``names`` is returned

    Passing ``gradient_param`` as a str corresponding to a gradient parameter
    (for instance "cell" or "positions") returns the unique indices only for the
    gradient blocks associated with each block in the input ``tensor``,
    according to the specified ``axis`` and ``names``. Note that gradient blocks
    by definition have the same properties metadata as their parent
    :py:class:`TensorBlock`.

    To find the unique "structure" indices along the "samples" axis present in a
    given TensorMap:

    .. code-block:: python

        unique_samples = unique_metadata(
            tensor, axis="samples", names=["structure"],
        )

    To find the unique "atom" indices along the "samples" axis present in the
    "positions" gradient blocks of a given TensorMap:

    .. code-block:: python

        unique_grad_samples = unique_metadata(
            tensor, axis="samples", names=["atom"], gradient_param="positions",
        )

    :param block: the :py:class:`TensorMap` to find unique indices for.
    :param axis: a str, either "samples" or "properties", corresponding to the
        axis along which the named unique indices should be found.
    :param names: a str, list of str, or tuple of str corresponding to the
        name(s) of the indices along the specified ``axis`` for which the unique
        values should be found.
    :param gradient_param: a str corresponding to the gradient parameter name
        for the gradient blocks to find the unique indices for. If none
        (default), the unique indices of the regular TensorBlocks will be
        calculated.

    :return: a sorted :py:class:`Labels` object containing the unique indices
        for the input ``block`` or its gradient for the specified parameter.
        Each element in the returned :py:class:`Labels` object has len(names)
        entries.
    """
    # Parse input args
    if not isinstance(tensor, TensorMap):
        raise TypeError("``tensor`` must be an equistore TensorMap")
    names = (
        [names]
        if isinstance(names, str)
        else (list(names) if isinstance(names, tuple) else names)
    )
    _check_args(tensor, axis, names, gradient_param)
    # Make a list of the blocks to find unique indices for
    if gradient_param is None:
        blocks = tensor.blocks()
    else:
        blocks = [block.gradient(gradient_param) for block in tensor.blocks()]

    return _unique_from_blocks(blocks, axis, names, gradient_param)


def unique_metadata_block(
    block: TensorBlock,
    axis: str,
    names: Union[List[str], Tuple[str], str],
    gradient_param: Optional[str] = None,
) -> Labels:
    """
    For a given ``axis`` (either "samples" or "properties"), and for the given
    samples/proeprties ``names``, returns a :py:class:`Labels` object of the
    unique indices in the input :py:class:`TensorBlock` ``block``.

    If there are no indices in ``block`` corresponding to the specified ``axis``
    and ``names``, an empty Labels object with the correct names as in the
    passed ``names`` is returned.

    Passing ``gradient_param`` as a str corresponding to a gradient parameter
    (for instance "cell" or "positions") returns the unique indices only for the
    gradient blocks associated with the input ``block``, according to the
    specified ``axis`` and ``names``. Note that gradient blocks by definition
    have the same properties metadata as their parent :py:class:`TensorBlock`.

    To find the unique "structure" indices along the "samples" axis present in a
    given TensorBlock:

    .. code-block:: python

        unique_samples = unique_metadata_block(
            block, axis="samples", names=["structure"],
        )

    To find the unique "atom" indices along the "samples" axis present in the
    "positions" gradient block of a given TensorBlock:

    .. code-block:: python

        unique_grad_samples = unique_metadata_block(
            block, axis="samples", names=["atom"], gradient_param="positions",
        )

    :param block: the :py:class:`TensorBlock` to find unique indices for.
    :param axis: a str, either "samples" or "properties", corresponding to the
        axis along which the named unique indices should be found.
    :param names: a str, list of str, or tuple of str corresponding to the
        name(s) of the indices along the specified ``axis`` for which the unique
        values should be found.
    :param gradient_param: a str corresponding to the gradient parameter name
        for the gradient blocks to find the unique indices for. If none
        (default), the unique indices of the regular TensorBlocks will be
        calculated.

    :return: a sorted :py:class:`Labels` object containing the unique indices
        for the input ``block`` or its gradient for the specified parameter.
        Each element in the returned :py:class:`Labels` object has len(names)
        entries.
    """
    # Parse input args
    if not isinstance(block, TensorBlock):
        raise TypeError("``block`` must be an equistore TensorBlock")
    names = (
        [names]
        if isinstance(names, str)
        else (list(names) if isinstance(names, tuple) else names)
    )
    _check_args(block, axis, names, gradient_param)
    # Make a list of the blocks to find unique indices for
    if gradient_param is None:
        blocks = [block]
    else:
        blocks = [block.gradient(gradient_param)]

    return _unique_from_blocks(blocks, axis, names, gradient_param)


def _unique_from_blocks(
    blocks: List[TensorBlock],
    axis: str,
    names: List[str],
    gradient_param: Optional[str],
) -> Labels:
    """
    Finds the unique metadata of a list of blocks along the given ``axis`` and
    for the specified ``names``. If ``gradient_param`` is specified, only finds
    the unique indices for gradient blocks under the specified param name.
    """
    # Extract indices from each block
    all_idxs = []
    for block in blocks:
        idxs = block.samples[names] if axis == "samples" else block.properties[names]
        for idx in idxs:
            all_idxs.append(idx)

    # If no matching indices across all blocks return a empty Labels w/ the
    # correct names
    if len(all_idxs) == 0:
        # Create Labels with single entry
        labels = Labels(names=names, values=np.array([[i for i in range(len(names))]]))
        # rslice to zero length
        return labels[:0]

    # Define the unique and sorted indices
    unique_idxs = np.unique(all_idxs, axis=0)

    # Return as Labels
    return Labels(names=names, values=np.array([[j for j in i] for i in unique_idxs]))


def _check_args(
    tensor: Union[TensorMap, TensorBlock],
    axis: str,
    names: List[str],
    gradient_param: Optional[str] = None,
):
    """Checks input args for :py:func:`unique_metadata` and
    :py:func:`unique_metadata_block`."""
    # Check tensors
    if isinstance(tensor, TensorMap):
        blocks = tensor.blocks()
        # Check gradients
        if gradient_param is not None:
            if not isinstance(gradient_param, str):
                raise TypeError("``gradient_param`` must be a str")
            # Check all blocks have a gradient under the passed param
            if not np.all([block.has_gradient(gradient_param) for block in blocks]):
                raise ValueError(
                    "not all input blocks have a gradient under the"
                    + f" passed ``gradient_param`` {gradient_param}"
                )
            blocks = [
                block.gradient(gradient_param) for block in blocks
            ]  # redefine blocks
    elif isinstance(tensor, TensorBlock):
        blocks = [tensor]
        # Check gradients
        if gradient_param is not None:
            if not isinstance(gradient_param, str):
                raise TypeError("``gradient_param`` must be a str")
            # Check block has a gradient under the passed param
            if not tensor.has_gradient(gradient_param):
                raise ValueError(
                    "input block does not have a gradient under the"
                    + f" passed ``gradient_param`` {gradient_param}"
                )
            blocks = [tensor.gradient(gradient_param)]  # redefine blocks
    # Check axis
    if not isinstance(axis, str):
        raise TypeError("``axis`` must be a str, either 'samples' or 'properties'")
    if axis not in ["samples", "properties"]:
        raise ValueError("``axis`` must be passsed as either 'samples' or 'properties'")
    # Check names
    if not isinstance(names, list):
        raise TypeError("``names`` must be a list of str")
    for block in blocks:
        tmp_names = block.samples.names if axis == "samples" else block.properties.names
        for name in names:
            if name not in tmp_names:
                raise ValueError(
                    "the block(s) passed must have samples/properties"
                    + " names that matches the one passed in ``names``"
                )