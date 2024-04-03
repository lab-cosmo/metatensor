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
        if not _dispatch.is_contiguous_array(gradient.values):
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
