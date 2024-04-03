from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    torch_jit_is_scripting,
    torch_jit_script,
)


@torch_jit_script
def is_contiguous_block(block: TensorBlock) -> bool:
    """
    Checks whether the values array and gradients values arrays (if present) of an input
    :py:class:`TensorBlock` are contiguous.

    Note that arrays of :py:class:`Labels` objects are not checked for contiguity.

    :param block: the input :py:class:`TensorBlock`.

    :return: bool, true if all values arrays contiguous, false otherwise.
    """
    check_contiguous: bool = True
    if not _dispatch.is_contiguous_array(block.values):
        check_contiguous = False

    for _param, gradient in block.gradients():
        if not is_contiguous_block(gradient):
            check_contiguous = False

    return check_contiguous


@torch_jit_script
def is_contiguous(tensor: TensorMap) -> bool:
    """
    Checks whether all values arrays and gradients values arrays (if present) in all
    :py:class:`TensorBlock` of an input :py:class:`TensorMap` are contiguous.

    Note that arrays of :py:class:`Labels` objects are not checked for contiguity.

    :param tensor: the input :py:class:`TensorMap`.

    :return: bool, true if all values arrays contiguous, false otherwise.
    """
    check_contiguous: bool = True
    for _key, block in tensor.items():
        if not is_contiguous_block(block):
            check_contiguous = False

    return check_contiguous


@torch_jit_script
def make_contiguous_block(block: TensorBlock) -> TensorBlock:
    """
    Returns a new :py:class:`TensorBlock` where the values and gradient values (if
    present) arrays are mades to be contiguous.

    :param block: the input :py:class:`TensorBlock`.

    :return: a new :py:class:`TensorBlock` where the values and gradients arrays (if
        present) are contiguous.
    """
    contiguous_block = TensorBlock(
        values=_dispatch.make_contiguous_array(block.values.copy()),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )
    for param, gradient in block.gradients():
        new_gradient = TensorBlock(
            values=_dispatch.make_contiguous_array(gradient.values.copy()),
            samples=gradient.samples,
            components=gradient.components,
            properties=gradient.properties,
        )
        contiguous_block.add_gradient(param, new_gradient)

    return contiguous_block


@torch_jit_script
def make_contiguous(tensor: TensorMap) -> TensorMap:
    """
    Returns a new :py:class:`TensorMap` where all values and gradient values arrays are
    mades to be contiguous.

    :param tensor: the input :py:class:`TensorMap`.

    :return: a new :py:class:`TensorMap` with the same data and metadata as ``tensor``
    and contiguous values of ``tensor``.
    """
    keys: Labels = tensor.keys
    contiguous_blocks: List[TensorBlock] = []
    for _key, block in tensor.items():
        contiguous_block = make_contiguous_block(block)
        contiguous_blocks.append(contiguous_block)

    return TensorMap(keys=keys, blocks=contiguous_blocks)
