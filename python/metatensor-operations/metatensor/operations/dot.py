from typing import List

from . import _dispatch
from ._backend import TensorBlock, TensorMap, torch_jit_script
from ._utils import _check_same_keys_raise


def _dot_block(block_1: TensorBlock, block_2: TensorBlock) -> TensorBlock:
    if not block_1.properties == block_2.properties:
        raise ValueError("TensorBlocks in `dot` should have the same properties")

    if len(block_2.components) > 0:
        raise ValueError("the second TensorMap in `dot` should not have components")

    if len(block_2.gradients_list()) > 0:
        raise ValueError("the second TensorMap in `dot` should not have gradients")

    # values = block_1.values @ block_2.values.T
    values = _dispatch.dot(block_1.values, block_2.values)

    result_block = TensorBlock(
        values=values,
        samples=block_1.samples,
        components=block_1.components,
        properties=block_2.samples,
    )

    for parameter, gradient in block_1.gradients():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        # gradient_values = gradient.values @ block_2.values.T
        gradient_values = _dispatch.dot(gradient.values, block_2.values)

        result_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient_values,
                samples=gradient.samples,
                components=gradient.components,
                properties=result_block.properties,
            ),
        )

    return result_block


@torch_jit_script
def dot(tensor_1: TensorMap, tensor_2: TensorMap) -> TensorMap:
    """Compute the dot product of two :py:class:`TensorMap`.

    The two :py:class:`TensorMap` must have the same ``keys``. The resulting
    :py:class:`TensorMap` will have the same keys as the input and each block
    will be the dot product of the two corresponding :py:class:`TensorBlock` in
    the input.

    :py:class:`TensorBlocks` corresponding to the same key must have the same
    ``properties``. The resulting :py:class:`TensorBlocks` of the dot product of
    two :py:class:`TensorBlocks` has ``result_block.values = block_1.values @
    block_2.values.T``

    >>> import numpy as np
    >>> from metatensor import Labels
    >>> block_1 = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 3],
    ...             [4, 5, 6],
    ...         ]
    ...     ),
    ...     samples=Labels(["system"], np.array([[0], [1]])),
    ...     components=[],
    ...     properties=Labels(["properties"], np.array([[0], [1], [2]])),
    ... )
    >>> block_2 = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 3],
    ...             [4, 5, 6],
    ...         ]
    ...     ),
    ...     samples=Labels(["system"], np.array([[0], [1]])),
    ...     components=[],
    ...     properties=Labels(["properties"], np.array([[0], [1], [2]])),
    ... )
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    >>> A = TensorMap(keys, [block_1])
    >>> B = TensorMap(keys, [block_2])
    >>> tensor_dot = dot(A, B)
    >>> print(tensor_dot.block(0))
    TensorBlock
        samples (2): ['system']
        components (): []
        properties (2): ['system']
        gradients: None
    >>> print(tensor_dot.block(0).samples)
    Labels(
        system
          0
          1
    )
    >>> print(tensor_dot.block(0).values)
    [[14 32]
     [32 77]]

    :param A: first :py:class:`TensorMap` to multiply
    :param B: second :py:class:`TensorMap` to multiply

    :return: a :py:class:`TensorMap` with the same keys of ``A`` and ``B``, and
            where each :py:class:`TensorBlock` has: the ``sample`` equal to the
            ``sample`` of ``A``; the ``properties`` equal to the ``sample`` of
            ``B``; and the ``components`` equal to the ``components`` of ``A``

    """
    _check_same_keys_raise(tensor_1, tensor_2, "dot")

    blocks: List[TensorBlock] = []
    for key, block_1 in tensor_1.items():
        block_2 = tensor_2.block(key)
        blocks.append(_dot_block(block_1=block_1, block_2=block_2))

    return TensorMap(tensor_1.keys, blocks)
