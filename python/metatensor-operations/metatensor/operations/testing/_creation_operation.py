from typing import Union

import numpy as np

from .. import _dispatch
from .._classes import Labels, TensorBlock, TensorMap
from .._dispatch import TorchTensor
from ..block_from_array import block_from_array


def cartesian_cubic(
    cartesian_vector: Union[np.ndarray, TorchTensor], compute_grad: bool = False
) -> TensorMap:
    """
    Creates a tensor map from a set of Cartesian vectors together with gradients if
    :param compute_grad: is `True` according to the function:

    .. math::

        f(x, y, z) = x^3 + y^3 + z^3

        \\nabla f = (3x^2, 3y^2, 3z^2)

    :param cartesian_vector: Set of Cartesian vectors with shape (n_samples, 3)
    :param compute_grad: Specifies if the returned tensor map should contain the
        gradients
    """

    cartesian_vector_cubic = cartesian_vector**3
    values = _dispatch.sum(cartesian_vector_cubic, axis=1).reshape(-1, 1)
    if compute_grad:
        values_grad = _dispatch.zeros_like(cartesian_vector, (len(values), 3, 1))
        values_grad[:, 0] = 3 * cartesian_vector[:, 0:1] ** 2
        values_grad[:, 1] = 3 * cartesian_vector[:, 1:2] ** 2
        values_grad[:, 2] = 3 * cartesian_vector[:, 2:3] ** 2

    block = block_from_array(values)
    if compute_grad:
        block.add_gradient(
            parameter="positions",
            gradient=TensorBlock(
                values=values_grad,
                samples=Labels.range("sample", len(values)),
                components=[Labels.range("cartesian", 3)],
                properties=block.properties,
            ),
        )
    return TensorMap(Labels.range("_", 1), [block])


def cartesian_linear(
    cartesian_vector: Union[np.ndarray, TorchTensor], compute_grad: bool = False
) -> TensorMap:
    """
    Creates a tensor map from a set of Cartesian vectors together with gradients if
    :param compute_grad: is `True` according to the function:

    .. math::

        f(x, y, z) = 3x + 2y + 8*z + 4

        \\nabla f = (3, 2, 8)

    :param cartesian_vector: Set of Cartesian vectors with shape (n_samples, 3)
    :param compute_grad: Specifies if the returned tensor map should contain the
        gradients
    """

    cartesian_vector_linear = (
        3 * cartesian_vector[:, 0]
        + 2 * cartesian_vector[:, 1]
        + 8 * cartesian_vector[:, 2]
        + 4
    )
    values = cartesian_vector_linear.reshape(-1, 1)
    if compute_grad:
        values_grad = _dispatch.zeros_like(cartesian_vector, (len(values), 3, 1))
        values_grad[:, 0] = 3 * _dispatch.ones_like(cartesian_vector, (len(values), 1))
        values_grad[:, 1] = 2 * _dispatch.ones_like(cartesian_vector, (len(values), 1))
        values_grad[:, 2] = 8 * _dispatch.ones_like(cartesian_vector, (len(values), 1))

    block = block_from_array(values)
    if compute_grad:
        block.add_gradient(
            parameter="positions",
            gradient=TensorBlock(
                values=values_grad,
                samples=Labels.range("sample", len(values)),
                components=[Labels.range("cartesian", 3)],
                properties=block.properties,
            ),
        )
    return TensorMap(Labels.range("_", 1), [block])
