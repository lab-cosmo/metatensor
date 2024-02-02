from typing import Callable, Union

import numpy as np
from numpy.testing import assert_allclose

from .. import _dispatch
from .._classes import TensorMap
from .._dispatch import TorchTensor


def finite_differences(
    function: Callable[[Union[np.ndarray, TorchTensor], bool], TensorMap],
    input_array: Union[np.ndarray, TorchTensor],
    parameter: str = "positions",
    displacement: float = 1e-6,
    rtol: float = 1e-5,
    atol: float = 1e-16,
) -> None:
    """
    Check that analytical gradients with respect to the :param parameter: agree with a
    finite difference calculation of the gradients. The callable must be able to return
    the analtyical gradients optionally if input argument `compute_grad` is true so it
    can be tested here. The dimension of the gradients are supposed to be in the
    components.  For example if the gradients are taken with respect to Cartesian
    coordinates the :param function: outputs a tensor map of gradients_with 3
    components.

    :param function: a function that outputs a tensor map (with gradients if specified
        by input parameter `compute_grad`) from the :param input_array:.
    :param input_array: an input for which the analytical and numerical gradients are
        tested
    :param parameter: the parameter of the gradient that is checked
    :param displacement: distance each atom will be displaced in each direction when
        computing finite differences
    :param max_relative: Maximal relative error. ``10 * displacement`` is a good
        starting point
    :param atol: Threshold below which all values are considered zero. This should be
        very small (1e-16) to prevent false positives (if all values & gradients are
        below that threshold, tests will pass even with wrong gradients)
    :raises AssertionError: if the two gradients are not equal up to specified precision
    """
    reference = function(input_array, compute_grad=True)
    dim_gradients = len(reference[0].gradient(parameter).components)
    for spatial in range(dim_gradients):
        input_pos = _dispatch.copy(input_array)
        input_pos[:, spatial] += displacement / 2
        updated_pos = function(input_pos)

        input_neg = _dispatch.copy(input_array)
        input_neg[:, spatial] -= displacement / 2
        updated_neg = function(input_neg)

        assert updated_pos.keys == reference.keys
        assert updated_neg.keys == reference.keys

        for key, block in reference.items():
            gradients = block.gradient(parameter)

            block_pos = updated_pos.block(key)
            block_neg = updated_neg.block(key)

            for gradient_i, sample_labels in enumerate(gradients.samples):
                sample_i = sample_labels[0]

                # check that the sample is the same in both descriptors
                assert block_pos.samples[sample_i] == block.samples[sample_i]
                assert block_neg.samples[sample_i] == block.samples[sample_i]

                value_pos = block_pos.values[sample_i]
                value_neg = block_neg.values[sample_i]
                gradient = gradients.values[gradient_i, spatial]

                assert value_pos.shape == gradient.shape
                assert value_neg.shape == gradient.shape

                finite_difference = (value_pos - value_neg) / displacement

                assert_allclose(finite_difference, gradient, rtol=rtol, atol=atol)
