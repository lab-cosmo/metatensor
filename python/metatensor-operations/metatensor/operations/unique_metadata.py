"""
Module for finding unique metadata for TensorMaps and TensorBlocks
"""

from typing import List, Optional, Tuple, Union

from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    check_isinstance,
    torch_jit_is_scripting,
    torch_jit_script,
)


def _unique_from_blocks(
    blocks: List[TensorBlock],
    axis: str,
    names: List[str],
) -> Labels:
    """
    Finds the unique metadata of a list of blocks along the given ``axis`` and for the
    specified ``names``.
    """
    all_values = []
    for block in blocks:
        if axis == "samples":
            all_values.append(block.samples.view(names).values)
        else:
            assert axis == "properties"
            all_values.append(block.properties.view(names).values)

    unique_values = _dispatch.unique(_dispatch.concatenate(all_values, axis=0), axis=0)
    return Labels(names=names, values=unique_values)


def _check_args(
    axis: str,
    names: List[str],
    gradient: Optional[str] = None,
):
    """Checks input args for `unique_metadata_block`"""

    if not torch_jit_is_scripting():
        if not isinstance(axis, str):
            raise TypeError(f"`axis` must be a string, not {type(axis)}")

        if not isinstance(names, list):
            raise TypeError(f"`names` must be a list of strings, not {type(names)}")

        for name in names:
            if not isinstance(name, str):
                raise TypeError(f"`names` elements must be a strings, not {type(name)}")

    if gradient is not None:
        if not torch_jit_is_scripting():
            if not isinstance(gradient, str):
                raise TypeError(f"`gradient` must be a string, not {type(gradient)}")

    if axis not in ["samples", "properties"]:
        raise ValueError(
            f"`axis` must be either 'samples' or 'properties', not '{axis}'"
        )


@torch_jit_script
def unique_metadata(
    tensor: TensorMap,
    axis: str,
    names: Union[List[str], Tuple[str], str],
    gradient: Optional[str] = None,
) -> Labels:
    """
    Returns a :py:class:`Labels` object containing the unique metadata across all blocks
    of the input :py:class:`TensorMap`  ``tensor``. Unique Labels are returned for the
    specified ``axis`` (either ``"samples"`` or ``"properties"``) and metadata
    ``names``.

    Passing ``gradient`` as a ``str`` corresponding to a gradient parameter (for
    instance ``"strain"`` or ``"positions"``) returns the unique indices only for the
    gradient blocks. Note that gradient blocks by definition have the same properties
    metadata as their parent :py:class:`TensorBlock`.

    An empty :py:class:`Labels` object is returned if there are no indices in the
    (gradient) blocks of ``tensor`` corresponding to the specified ``axis`` and
    ``names``. This will have length zero but the names will be the same as passed in
    ``names``.

    For example, to find the unique ``"system"`` indices in the ``"samples"`` metadata
    present in a given :py:class:`TensorMap`:

    >>> import numpy as np
    >>> from metatensor import Labels, TensorBlock, TensorMap
    >>> import metatensor
    >>> block = TensorBlock(
    ...     values=np.random.rand(5, 3),
    ...     samples=Labels(
    ...         names=["system", "atom"],
    ...         values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 3]]),
    ...     ),
    ...     components=[],
    ...     properties=Labels.range("properties", 3),
    ... )
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    >>> tensor = TensorMap(keys, [block.copy()])
    >>> unique_systems = metatensor.unique_metadata(
    ...     tensor,
    ...     axis="samples",
    ...     names=["system"],
    ... )
    >>> unique_systems
    Labels(
        system
          0
          1
          2
    )

    Or, to find the unique ``(system, atom)`` pairs of indices in the ``"samples"``
    metadata present in the ``"positions"`` gradient blocks of a given
    :py:class:`TensorMap`:

    >>> gradient = TensorBlock(
    ...     values=np.random.rand(4, 3, 3),
    ...     samples=Labels(
    ...         names=["sample", "system", "atom"],
    ...         values=np.array([[0, 0, 0], [1, 0, 0], [2, 1, 4], [3, 2, 5]]),
    ...     ),
    ...     components=[Labels.range("xyz", 3)],
    ...     properties=Labels.range("properties", 3),
    ... )
    >>> block.add_gradient("positions", gradient)
    >>> tensor = TensorMap(keys, [block])
    >>> metatensor.unique_metadata(
    ...     tensor,
    ...     axis="samples",
    ...     names=["system", "atom"],
    ...     gradient="positions",
    ... )
    Labels(
        system  atom
          0      0
          1      4
          2      5
    )

    :param tensor: the :py:class:`TensorMap` to find unique indices for.
    :param axis: a ``str``, either ``"samples"`` or ``"properties"``, corresponding to
        the ``axis`` along which the named unique indices should be found.
    :param names: a ``str``, ``list`` of ``str``, or ``tuple`` of ``str`` corresponding
        to the name(s) of the indices along the specified ``axis`` for which the unique
        values should be found.
    :param gradient: a ``str`` corresponding to the gradient parameter name for the
        gradient blocks to find the unique indices for. If :py:obj:`None` (default), the
        unique indices of the :py:class:`TensorBlock` containing the values will be
        calculated.

    :return: a sorted :py:class:`Labels` object containing the unique metadata for the
        blocks of the input ``tensor`` or its gradient blocks for the specified
        parameter. Each element in the returned :py:class:`Labels` object has
        len(``names``) entries.
    """
    # Parse input args
    if not torch_jit_is_scripting():
        if not check_isinstance(tensor, TensorMap):
            raise TypeError(
                f"`tensor` must be a metatensor TensorMap, not {type(tensor)}"
            )

    names = (
        [names]
        if isinstance(names, str)
        else (list(names) if isinstance(names, tuple) else names)
    )
    _check_args(axis, names, gradient)

    # Make a list of the blocks to find unique indices for
    if gradient is None:
        blocks = tensor.blocks()
    else:
        blocks = [block.gradient(gradient) for block in tensor.blocks()]

    return _unique_from_blocks(blocks, axis, names)


@torch_jit_script
def unique_metadata_block(
    block: TensorBlock,
    axis: str,
    names: Union[List[str], Tuple[str], str],
    gradient: Optional[str] = None,
) -> Labels:
    """
    Returns a :py:class:`Labels` object containing the unique metadata in the input
    :py:class:`TensorBlock`  ``block``, for the specified ``axis`` (either ``"samples"``
    or ``"properties"``) and metadata ``names``.

    Passing ``gradient`` as a ``str`` corresponding to a gradient parameter (for
    instance ``"strain"`` or ``"positions"``) returns the unique indices only for the
    gradient block associated with ``block``. Note that gradient blocks by definition
    have the same properties metadata as their parent :py:class:`TensorBlock`.

    An empty :py:class:`Labels` object is returned if there are no indices in the
    (gradient) blocks of ``tensor`` corresponding to the specified ``axis`` and
    ``names``. This will have length zero but the names will be the same as passed in
    ``names``.

    :param block: the :py:class:`TensorBlock` to find unique indices for.
    :param axis: a str, either ``"samples"`` or ``"properties"``, corresponding to the
        ``axis`` along which the named unique metadata should be found.
    :param names: a ``str``, ``list`` of ``str``, or ``tuple`` of ``str`` corresponding
        to the name(s) of the metadata along the specified ``axis`` for which the unique
        indices should be found.
    :param gradient: a ``str`` corresponding to the gradient parameter name for the
        gradient blocks to find the unique metadata for. If :py:obj:`None` (default),
        the unique metadata of the regular :py:class:`TensorBlock` objects will be
        calculated.

    :return: a sorted :py:class:`Labels` object containing the unique metadata for the
        input ``block`` or its gradient for the specified parameter. Each element in the
        returned :py:class:`Labels` object has len(``names``) entries.
    """
    # Parse input args
    if not torch_jit_is_scripting():
        if not check_isinstance(block, TensorBlock):
            raise TypeError(
                f"`block` must be a metatensor TensorBlock, not {type(block)}"
            )

    names = (
        [names]
        if isinstance(names, str)
        else (list(names) if isinstance(names, tuple) else names)
    )
    _check_args(axis, names, gradient)

    # Make a list of the blocks to find unique indices for
    if gradient is None:
        blocks = [block]
    else:
        blocks = [block.gradient(gradient)]

    return _unique_from_blocks(blocks, axis, names)
