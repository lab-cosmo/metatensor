from . import _dispatch
from ._backend import (  # torch_jit_is_scripting,; torch_jit_script,
    TensorBlock,
    TensorMap,
)


def is_contiguous(tensor: TensorMap) -> bool:
    """
    Checks contiguity of values and gradients in  a :py:class:`TensorMap`

    :param tensor: the input :py:class:`TensorMap`.

    :return: a boolean indicating (non-)contiguity of values in ``tensor``. Returns true
        if all values and gradients arrays in the input ``tensor`` are contiguous, false
        otherwise.
    """
    check_contiguous = True
    for _key, block in tensor.items():
        if not is_contiguous_block(block):
            check_contiguous = False

        for _param, gradient in block.gradients():
            if not is_contiguous_block(gradient):
                check_contiguous = False

    return check_contiguous


def is_contiguous_block(block: TensorBlock) -> bool:
    """
    :param block: the input :py:class:`TensorBlock`.

    :return: a boolean indicating (non-)contiguity of values in `TensorBlock` supplied
        as input.
    """
    check_contiguous = True
    if not _dispatch.is_contiguous_array(block.values):
        check_contiguous = False

    return check_contiguous


def make_contiguous(tensor: TensorMap) -> TensorMap:
    """
    Return a new :py:class:`TensorMap` where the values are made to be contiguous.

    :param tensor: the input :py:class:`TensorMap`.

    :return: a new :py:class:`TensorMap` with the same data and metadata as ``tensor``
    and contiguous values of ``tensor``.
    """

    keys = tensor.keys
    contiguous_blocks = []

    for _key, block in tensor.items():
        contiguous_block = make_contiguous_block(block)
        contiguous_blocks.append(contiguous_block)

    return TensorMap(keys=keys, blocks=contiguous_blocks)


def make_contiguous_block(block: TensorBlock) -> TensorBlock:
    """
    Return a new :py:class:`TensorBlock` where the values are made to be contiguous.

    :param block: the input :py:class:`TensorBlock`.

    :return: a new :py:class:`TensorBlock` where all the values and gradients arrays (if
        present) are contiguous.
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
