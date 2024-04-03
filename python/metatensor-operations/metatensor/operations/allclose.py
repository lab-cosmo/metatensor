from . import _dispatch
from ._backend import TensorBlock, TensorMap, torch_jit_script
from ._utils import (
    NotEqualError,
    _check_blocks_impl,
    _check_same_gradients_impl,
    _check_same_keys_impl,
)


def _allclose_impl(
    tensor_1: TensorMap, tensor_2: TensorMap, rtol: float, atol: float, equal_nan: bool
) -> str:
    """Abstract function to perform an allclose operation between two TensorMaps."""
    message = _check_same_keys_impl(tensor_1, tensor_2, "allclose")
    if message != "":
        return f"the tensor maps have different keys: {message}"

    for key, block_1 in tensor_1.items():
        message = _allclose_block_impl(
            block_1=block_1,
            block_2=tensor_2.block(key),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
        if message != "":
            return f"blocks for key {key.print()} are different: {message}"
    return ""


def _allclose_block_impl(
    block_1: TensorBlock,
    block_2: TensorBlock,
    rtol: float,
    atol: float,
    equal_nan: bool,
) -> str:
    """Abstract function to perform an allclose operation between two TensorBlocks."""
    if not block_1.values.shape == block_2.values.shape:
        return "values shapes are different"

    if not _dispatch.allclose(
        block_1.values, block_2.values, rtol=rtol, atol=atol, equal_nan=equal_nan
    ):
        return "values are not allclose"

    check_blocks_message = _check_blocks_impl(
        block_1,
        block_2,
        fname="allclose",
    )
    if check_blocks_message != "":
        return check_blocks_message

    check_same_gradient_message = _check_same_gradients_impl(
        block_1,
        block_2,
        check=["samples", "properties", "components"],
        fname="allclose",
    )
    if check_same_gradient_message != "":
        return check_same_gradient_message

    for parameter, gradient1 in block_1.gradients():
        gradient2 = block_2.gradient(parameter)

        if not _dispatch.allclose(
            gradient1.values,
            gradient2.values,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        ):
            return f"gradient '{parameter}' values are not allclose"
    return ""


@torch_jit_script
def allclose(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    rtol: float = 1e-13,
    atol: float = 1e-12,
    equal_nan: bool = False,
) -> bool:
    """
    Compare two :py:class:`TensorMap`.

    This function returns :py:obj:`True` if the two tensors have the same keys
    (potentially in different order) and all the :py:class:`TensorBlock` have the same
    (and in the same order) samples, components, properties, and their values matrices
    pass the numpy-like ``allclose`` test with the provided ``rtol``, and ``atol``.

    The :py:class:`TensorMap` contains gradient data, then this function only returns
    :py:obj:`True` if all the gradients also have the same samples, components,
    properties and their data matrices pass the numpy-like ``allclose`` test with the
    provided ``rtol``, and ``atol``.

    In practice this function calls :py:func:`allclose_raise`, returning :py:obj:`True`
    if no exception is raised, :py:obj:`False` otherwise.

    :param tensor_1: first :py:class:`TensorMap`
    :param tensor_2: second :py:class:`TensorMap`
    :param rtol: relative tolerance for ``allclose``
    :param atol: absolute tolerance for ``allclose``
    :param equal_nan: should two ``NaN`` be considered equal?

    Examples
    --------
    >>> import numpy as np
    >>> from metatensor import Labels, TensorBlock

    Create simple block

    >>> block_1 = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 4],
    ...             [3, 5, 6],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels(["properties"], np.array([[0], [1], [2]])),
    ... )

    Create a second block that is equivalent to ``block_1``.

    >>> block_2 = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 4],
    ...             [3, 5, 6],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels(["properties"], np.array([[0], [1], [2]])),
    ... )

    Create tensors from blocks, using keys with different names

    >>> keys1 = Labels(names=["key1"], values=np.array([[0]]))
    >>> keys2 = Labels(names=["key2"], values=np.array([[0]]))
    >>> tensor_1 = TensorMap(keys1, [block_1])
    >>> tensor_2 = TensorMap(keys2, [block_2])

    Call :py:func:`metatensor.allclose()`, which should fail as the blocks have
    different keys associated with them.

    >>> allclose(tensor_1, tensor_2)
    False

    Create a third tensor, which differs from ``tensor_1`` only by ``1e-5`` in a single
    block value.

    >>> block3 = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1 + 1e-5, 2, 4],
    ...             [3, 5, 6],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels(["properties"], np.array([[0], [1], [2]])),
    ... )

    Create tensors from blocks, using a key with same name as ``block_1``.

    >>> keys3 = Labels(names=["key1"], values=np.array([[0]]))
    >>> tensor3 = TensorMap(keys3, [block3])

    Call :py:func:`metatensor.allclose()`, which should return False because the default
    ``rtol`` is ``1e-13``, and the difference in the first value between the blocks of
    the two tensors is ``1e-5``.

    >>> allclose(tensor_1, tensor3)
    False

    Calling allclose again with the optional argument ``rtol=1e-5`` should return
    :py:obj:`True`, as the difference in the first value between the blocks of the two
    tensors is within the tolerance limit

    >>> allclose(tensor_1, tensor3, rtol=1e-5)
    True
    """
    return not bool(
        _allclose_impl(
            tensor_1=tensor_1,
            tensor_2=tensor_2,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )
    )


@torch_jit_script
def allclose_raise(
    tensor_1: TensorMap,
    tensor_2: TensorMap,
    rtol: float = 1e-13,
    atol: float = 1e-12,
    equal_nan: bool = False,
):
    """
    Compare two :py:class:`TensorMap`, raising :py:class:`NotEqualError` if they
    are not the same.

    The message associated with the exception will contain more information on
    where the two :py:class:`TensorMap` differ. See :py:func:`allclose` for more
    information on which :py:class:`TensorMap` are considered equal.

    :raises: :py:class:`NotEqualError` if the blocks are different

    :param tensor_1: first :py:class:`TensorMap`
    :param tensor_2: second :py:class:`TensorMap`
    :param rtol: relative tolerance for ``allclose``
    :param atol: absolute tolerance for ``allclose``
    :param equal_nan: should two ``NaN`` be considered equal?

    Examples
    --------
    >>> import numpy as np
    >>> import metatensor
    >>> from metatensor import Labels, TensorBlock

    Create simple block, with one py:obj:`np.nan` value.

    >>> block_1 = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 4],
    ...             [3, 5, np.nan],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels(["properties"], np.array([[0], [1], [2]])),
    ... )

    Create a second block that differs from ``block_1`` by ``1e-5`` in its first value.

    >>> block_2 = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1 + 1e-5, 2, 4],
    ...             [3, 5, np.nan],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels(["properties"], np.array([[0], [1], [2]])),
    ... )

    Create tensors from blocks, using same keys

    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    >>> tensor_1 = TensorMap(keys, [block_1])
    >>> tensor_2 = TensorMap(keys, [block_2])

    Call :py:func:`metatensor.allclose_raise()`, which should raise
    :py:class:`metatensor.NotEqualError` because:

    1. The two ``NaN`` are not considered `equal`.
    2. The difference between the first value in the blocks is greater than the default
       ``rtol`` of ``1e-13``.

    If this is executed yourself, you will see a nested exception explaining that the
    ``values`` of the two blocks are not `allclose`.

    >>> allclose_raise(tensor_1, tensor_2)
    Traceback (most recent call last):
        ...
    metatensor.operations._utils.NotEqualError: blocks for key (key=0) are different: \
values are not allclose

    call :py:func:`metatensor.allclose_raise()` again, but use ``equal_nan=True`` and
    ``rtol=1e-5`` This passes, as the two ``NaN`` are now considered equal, and the
    difference between the first value of the blocks of the two tensors is within the
    ``rtol`` limit of ``1e-5``.

    >>> allclose_raise(tensor_1, tensor_2, equal_nan=True, rtol=1e-5)
    """
    message = _allclose_impl(
        tensor_1=tensor_1, tensor_2=tensor_2, rtol=rtol, atol=atol, equal_nan=equal_nan
    )
    if message != "":
        raise NotEqualError(message)


@torch_jit_script
def allclose_block(
    block_1: TensorBlock,
    block_2: TensorBlock,
    rtol: float = 1e-13,
    atol: float = 1e-12,
    equal_nan: bool = False,
) -> bool:
    """
    Compare two :py:class:`TensorBlock`.

    This function returns :py:obj:`True` if the two :py:class:`TensorBlock` have the
    same samples, components, properties and their values matrices must pass the
    numpy-like ``allclose`` test with the provided ``rtol``, and ``atol``.

    If the :py:class:`TensorBlock` contains gradients, then the gradient must
    also have same (and in the same order) samples, components, properties
    and their data matrices must pass the numpy-like ``allclose`` test with the
    provided ``rtol``, and ``atol``.

    In practice this function calls :py:func:`allclose_block_raise`, returning
    :py:obj:`True` if no exception is raised, :py:obj:`False` otherwise.

    :param block_1: first :py:class:`TensorBlock`
    :param block_2: second :py:class:`TensorBlock`
    :param rtol: relative tolerance for ``allclose``
    :param atol: absolute tolerance for ``allclose``
    :param equal_nan: should two ``NaN`` be considered equal?


    Examples
    --------
    >>> import numpy as np
    >>> from metatensor import Labels, TensorBlock

    Create simple block

    >>> block_1 = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 4],
    ...             [3, 5, 6],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels(["property_1"], np.array([[0], [1], [2]])),
    ... )

    Recreate ``block_1``, but change first value in the block from ``1`` to ``1.00001``.

    >>> block_2 = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1 + 1e-5, 2, 4],
    ...             [3, 5, 6],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels(["property_1"], np.array([[0], [1], [2]])),
    ... )

    Call :py:func:`metatensor.allclose_block()`, which should return :py:obj:`False`
    because the default ``rtol`` is ``1e-13``, and the difference in the first value
    between the two blocks is ``1e-5``.

    >>> allclose_block(block_1, block_2)
    False

    Calling :py:func:`metatensor.allclose_block()` with the optional argument
    ``rtol=1e-5`` should return :py:obj:`True`, as the difference in the first value
    between the two blocks is within the tolerance limit.

    >>> allclose_block(block_1, block_2, rtol=1e-5)
    True
    """
    return not bool(
        _allclose_block_impl(
            block_1=block_1, block_2=block_2, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
    )


@torch_jit_script
def allclose_block_raise(
    block_1: TensorBlock,
    block_2: TensorBlock,
    rtol: float = 1e-13,
    atol: float = 1e-12,
    equal_nan: bool = False,
):
    """
    Compare two :py:class:`TensorBlock`, raising :py:class:`NotEqualError` if
    they are not the same.

    The message associated with the exception will contain more information on
    where the two :py:class:`TensorBlock` differ. See :py:func:`allclose_block`
    for more information on which :py:class:`TensorBlock` are considered equal.

    :raises: :py:class:`NotEqualError` if the blocks are different

    :param block_1: first :py:class:`TensorBlock`
    :param block_2: second :py:class:`TensorBlock`
    :param rtol: relative tolerance for ``allclose``
    :param atol: absolute tolerance for ``allclose``
    :param equal_nan: should two ``NaN`` be considered equal?

    Examples
    --------
    >>> import numpy as np
    >>> import metatensor
    >>> from metatensor import Labels, TensorBlock

    Create simple block

    >>> block_1 = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 4],
    ...             [3, 5, 6],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels(["property_1"], np.array([[0], [1], [2]])),
    ... )

    Recreate ``block_1``, but rename properties label ``'property_1'`` to
    ``'property_2'``.

    >>> block_2 = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 4],
    ...             [3, 5, 6],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels(["property_2"], np.array([[0], [1], [2]])),
    ... )

    Call :py:func:`metatensor.allclose_block_raise()`, which should raise
    :py:func:`metatensor.NotEqualError` because the properties of the two blocks are not
    `equal`.

    >>> allclose_block_raise(block_1, block_2)
    Traceback (most recent call last):
        ...
    metatensor.operations._utils.NotEqualError: inputs to 'allclose' should have the \
same properties, but they are not the same or not in the same order
    """
    message = _allclose_block_impl(
        block_1=block_1, block_2=block_2, rtol=rtol, atol=atol, equal_nan=equal_nan
    )
    if message != "":
        raise NotEqualError(message)
