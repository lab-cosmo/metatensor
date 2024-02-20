from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    torch_jit_is_scripting,
    torch_jit_script,
)

def is_contiguous(tensor: TensorMap) -> bool:
    r"""
    Checks contiguity of values and gradients in  a :py:class:`TensorMap`
    
    :param A: the input :py:class:`TensorMap`.

    :return: a boolean indicating (non-)contiguity of values in ``A``.
    """
    check_contiguous = True
    for key, block in tensor.items():
        # Here, call another function: def is_contiguous_block(block: TensorBlock) -> bool
        if not is_contiguous_block(block):
            check_contiguous = False

        for param, gradient in block.gradients():
            if not is_contiguous_block(gradient):
                check_contiguous = False

    return check_contiguous


def is_contiguous_block(block: TensorBlock) -> bool:
    r"""
    Checks contiguity of values in  a :py:class:`TensorBlock`
    
    :param A: the input :py:class:`TensorBlock`.

    :return: a boolean indicating (non-)contiguity of values in ``A``.
    """
    check_contiguous = True
    if not _dispatch.is_contiguous_array(block.values):
        check_contiguous = False

    return check_contiguous


def make_contiguous(tensor: TensorMap) -> TensorMap:
    r"""
    Return a new :py:class:`TensorMap` where the values are made to be contiguous.

    :param A: the input :py:class:`TensorMap`.

    :return: a new :py:class:`TensorMap` with the same metadata as ``A`` and
        contiguous values of ``A``.
    """
    
    keys = tensor.keys
    contiguous_blocks = []

    for key, block in tensor.items():
        contiguous_block = make_contiguous_block(block)
        contiguous_blocks.append(contiguous_block)

    return TensorMap(keys=keys, blocks=contiguous_blocks)


def make_contiguous_block(block: TensorBlock) -> TensorBlock:
    r"""
    Return a new :py:class:`TensorBlock` where the values are made to be contiguous.

    :param A: the input :py:class:`TensorBlock`.

    :return: a new :py:class:`TensorBlock` where values are contiguous.
    """
    contiguous_block = TensorBlock(
        values=_dispatch.make_contiguous_array(block.values.copy()),
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )
    # Make gradients non-contig
    for param, gradient in block.gradients():

        new_gradient = TensorBlock(
            values=_dispatch.make_contiguous_array(gradient.values.copy()),
            samples=gradient.samples,
            components=gradient.components,
            properties=gradient.properties,
        )
        contiguous_block.add_gradient(param, new_gradient)

    return contiguous_block