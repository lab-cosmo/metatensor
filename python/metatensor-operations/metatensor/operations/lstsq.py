import warnings
from typing import List, Optional

from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    torch_jit_annotate,
    torch_jit_script,
)
from ._utils import (
    _check_blocks_raise,
    _check_same_gradients_raise,
    _check_same_keys_raise,
)


def _lstsq_block(
    X: TensorBlock, Y: TensorBlock, rcond: Optional[float], driver: Optional[str] = None
) -> TensorBlock:
    _check_blocks_raise(X, Y, check=["samples", "components"], fname="lstsq")
    _check_same_gradients_raise(X, Y, check=["samples", "components"], fname="lstsq")

    # reshape components together with the samples
    X_n_properties = X.values.shape[-1]
    X_values = X.values.reshape(-1, X_n_properties)

    Y_n_properties = Y.values.shape[-1]
    Y_values = Y.values.reshape(-1, Y_n_properties)

    for parameter, X_gradient in X.gradients():
        X_gradient_values = X_gradient.values.reshape(-1, X_n_properties)
        X_values = _dispatch.concatenate((X_values, X_gradient_values), axis=0)

        Y_gradient = Y.gradient(parameter)
        Y_gradient_values = Y_gradient.values.reshape(-1, Y_n_properties)
        Y_values = _dispatch.concatenate((Y_values, Y_gradient_values), axis=0)

    weights = _dispatch.lstsq(X_values, Y_values, rcond=rcond, driver=driver)

    return TensorBlock(
        values=weights.T,
        samples=Y.properties,
        components=torch_jit_annotate(List[Labels], []),
        properties=X.properties,
    )


@torch_jit_script
def lstsq(
    X: TensorMap,
    Y: TensorMap,
    rcond: Optional[float],
    driver: Optional[str] = None,
) -> TensorMap:
    r"""
    Solve a linear system using two :py:class:`TensorMap`.

    The least-squares solution ``w_b`` for the linear system :math:`X_b w_b =
    Y_b` is solved for all blocks :math:`b` in ``X`` and ``Y``. ``X`` and ``Y``
    must have the same keys. The returned :py:class:`TensorMap` ``w`` has the
    same keys as ``X`` and ``Y``, and stores in each block the least-squares
    solutions :math:`w_b`.

    If a block has multiple components, they are moved to the "samples" axis
    before solving the linear system.

    If gradients are present, they must be present in both ``X`` and ``Y``.
    Gradients are concatenated with the block values along the "samples" axis,
    :math:`A_b = [X_b, {\nabla} X_b]`, :math:`B_b = [Y_b, {\nabla} Y_b]`, and
    the linear system :math:`A_b w_b = B_b` is solved for :math:`w_b` using
    least-squares.

    .. note::
      The solutions :math:`w_b` differ from the output of numpy or torch in that they
      are already transposed. Be aware of that if you want to manually access
      the values of blocks of ``w`` (see also the example below).

    :param X:
        a :py:class:`TensorMap` containing the "coefficient" matrices

    :param Y:
        a :py:class:`TensorMap` containing the "dependent variable" values

    :param rcond:
        Cut-off ratio for small singular values of a. The singular value
        :math:`{\sigma}_i` is treated as zero if smaller than
        :math:`r_{cond}{\sigma}_1`, where :math:`{\sigma}_1` is the biggest
        singular value of :math:`X_b`. :py:obj:`None` choses the default value
        for numpy or PyTorch.

    :param driver:
        Used only in torch (ignored if numpy is used), see
        https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html for a
        full description

    :return: a :py:class:`TensorMap` with the same keys of ``Y`` and ``X``, and
        where each :py:class:`TensorBlock` has: the ``sample`` equal to the
        ``properties`` of ``Y``; and the ``properties`` equal to the
        ``properties`` of ``X``.

    >>> import numpy as np
    >>> from metatensor import Labels, TensorBlock, TensorMap
    >>> import metatensor
    >>> values_X = np.array(
    ...     [
    ...         [1.0, 2.0],
    ...         [3.0, 1.0],
    ...     ]
    ... )
    >>> values_Y = np.array(
    ...     [
    ...         [1.0, 0.0],
    ...         [0.0, 1.0],
    ...     ]
    ... )
    >>> samples = Labels("system", np.array([[0], [1]]))
    >>> components = []
    >>> properties = Labels("properties", np.array([[0], [1]]))
    >>> keys = Labels(names="key", values=np.array([[0]]))
    >>> block_X = TensorBlock(values_X, samples, components, properties)
    >>> block_Y = TensorBlock(values_Y, samples, components, properties)
    >>> X = TensorMap(keys, [block_X])
    >>> Y = TensorMap(keys, [block_Y])
    >>> w = metatensor.lstsq(X, Y, rcond=1e-10)

    We take the transpose here

    >>> y = X.block(0).values @ w.block(0).values.T

    Set small entries in y to 0, they are numerical noise

    >>> mask = np.abs(y) < 1e-15
    >>> y[mask] = 0.0
    >>> print(y)
    [[1. 0.]
     [0. 1.]]
    """
    if rcond is None:
        warnings.warn(
            "WARNING rcond is set to None, which will trigger the default "
            "behavior which is different between numpy and torch lstsq function, "
            "and might depend on the version you are using.",
            stacklevel=1,
        )

    _check_same_keys_raise(X, Y, "lstsq")

    blocks: List[TensorBlock] = []
    for key, X_block in X.items():
        Y_block = Y.block(key)
        blocks.append(_lstsq_block(X_block, Y_block, rcond=rcond, driver=driver))

    return TensorMap(X.keys, blocks)
