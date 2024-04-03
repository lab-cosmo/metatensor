"""
Reduction over samples
======================

These functions allow to reduce over the sample indices of a :py:class:`TensorMap` or
:py:class:`TensorBlock` objects, generating a new :py:class:`TensorMap` or
:py:class:`TensorBlock` in which the values sharing the same indices for the indicated
``sample_names`` have been combined in a single entry. The functions differ by the type
of reduction operation, but otherwise operate in the same way. The reduction operation
loops over the samples in each block/map, and combines all those that only differ by the
values of the indices associated with the names listed in the ``sample_names`` argument.
One way to see these operations is that the sample indices describe the non-zero entries
in a *sparse* array, and the reduction acts much like :func:`numpy.sum`, where
``sample_names`` plays the same role as the ``axis`` argument. Whenever gradients are
present, the reduction is performed also on the gradients.

See also :py:func:`metatensor.sum_over_samples_block` and
:py:func:`metatensor.sum_over_samples` for a detailed discussion with examples.

TensorMap operations
--------------------

.. autofunction:: metatensor.sum_over_samples

.. autofunction:: metatensor.mean_over_samples

.. autofunction:: metatensor.var_over_samples

.. autofunction:: metatensor.std_over_samples

TensorBlock operations
----------------------

.. autofunction:: metatensor.sum_over_samples_block

.. autofunction:: metatensor.mean_over_samples_block

.. autofunction:: metatensor.var_over_samples_block

.. autofunction:: metatensor.std_over_samples_block
"""

from typing import List, Optional, Union

import numpy as np

from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    torch_jit_is_scripting,
    torch_jit_script,
)


def _reduce_over_samples_block(
    block: TensorBlock,
    sample_names: Union[List[str], str],
    reduction: str,
    remaining_samples: Optional[List[str]] = None,
) -> TensorBlock:
    """
    Create a new :py:class:`TensorBlock` reducing the ``properties`` among the
    selected ``samples``.

    The output :py:class:`TensorBlocks` have the same components of the input
    one. "sum", "mean", "std" or "var" reductions can be performed.

    :param block:
        input block
    :param sample_names:
        names of samples to reduce over. it is ignored if remaining_samples is given
    :param remaining_samples:
        names of samples that should remain after reducing reduce over.
        it is computed automatically from sample_names if missing or set to None
    :param reduction:
        how to reduce, only available values are "mean", "sum", "std" or "var"
    """
    if isinstance(sample_names, str):
        sample_names_list = [sample_names]
    else:
        sample_names_list = sample_names

    block_samples = block.samples

    if remaining_samples is None:
        remaining_sample_names: List[str] = []
        for s_name in block_samples.names:
            if s_name in sample_names_list:
                continue
            remaining_sample_names.append(s_name)
    else:
        remaining_sample_names = remaining_samples

    for sample in remaining_sample_names:
        assert sample in block_samples.names

    assert reduction in ["sum", "mean", "var", "std"]
    # get the indices of the selected sample
    sample_selected = [
        block_samples.names.index(sample) for sample in remaining_sample_names
    ]

    # checks if it is a zero sample TensorBlock
    if len(block.samples) == 0:
        # Here is different from the general case where we use Labels.single() if
        # if len(remaining_sample_names) == 0
        # Labels.single() cannot be used because Labels.single() has not
        # an np.empty() array as values but has one values, it has dimension (1,...)
        # we want (0,...).
        # here if len(remaining_sample_names) == 0 ->
        # Labels([], shape=(0, 0), dtype=int32)

        samples_label = Labels(
            remaining_sample_names,
            _dispatch.zeros_like(block.values, [0, len(remaining_sample_names)]),
        )

        result_block = TensorBlock(
            values=block.values,
            samples=samples_label,
            components=block.components,
            properties=block.properties,
        )

        # The gradient does not change because the only thing that matters for
        # the gradients are the samples to which they are connected, but in this
        # case there are no samples in the TensorBlock
        for parameter, gradient in block.gradients():
            if len(gradient.gradients_list()) != 0:
                raise NotImplementedError("gradients of gradients are not supported")

            result_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=gradient.properties,
                ),
            )

        return result_block

    # get which samples will still be there after reduction
    if len(remaining_sample_names) == 0:
        new_samples = _dispatch.zeros_like(block_samples.values, shape=(1, 0))
        index = _dispatch.zeros_like(
            block_samples.values, shape=(block_samples.values.shape[0],)
        )
    else:
        new_samples, index = _dispatch.unique_with_inverse(
            block_samples.values[:, sample_selected], axis=0
        )
        index = index.reshape(-1)

    block_values = block.values
    other_shape = block_values.shape[1:]
    values_result = _dispatch.zeros_like(
        block_values, shape=(new_samples.shape[0],) + other_shape
    )

    _dispatch.index_add(
        values_result,
        block_values,
        index,
    )

    # define values_mean for torchscript (won't be used unless there are gradients)
    values_mean = _dispatch.empty_like(values_result, [0])

    if reduction == "mean" or reduction == "std" or reduction == "var":
        bincount = _dispatch.bincount(index)
        values_result = values_result / bincount.reshape(
            (-1,) + (1,) * len(other_shape)
        )
        if reduction == "std" or reduction == "var":
            values_result2 = _dispatch.zeros_like(
                block_values, shape=(new_samples.shape[0],) + other_shape
            )
            _dispatch.index_add(
                values_result2,
                block_values**2,
                index,
            )
            values_result2 = values_result2 / bincount.reshape(
                (-1,) + (1,) * len(other_shape)
            )
            # I need the mean values in the derivatives
            if len(block.gradients_list()) > 0:
                values_mean = _dispatch.copy(values_result)
            values_result = values_result2 - values_result**2
            if reduction == "std":
                values_result = _dispatch.sqrt(values_result)

    # check if the reduce operation reduce all the samples
    if len(remaining_sample_names) == 0:
        samples_label = Labels(
            names="_",
            values=_dispatch.zeros_like(block_samples.values, shape=(1, 1)),
        )
    else:
        samples_label = Labels(
            remaining_sample_names,
            new_samples,
        )

    result_block = TensorBlock(
        values=values_result,
        samples=samples_label,
        components=block.components,
        properties=block.properties,
    )

    for parameter, gradient in block.gradients():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        # check if all gradients are zeros
        if len(gradient.samples) == 0:
            # The gradients does not change because, if they are all zeros, the
            # gradients after reducing operation is still zero.

            # For any function of the TensorBlock values x(t):
            # f(x(t))-> df(x(t))/dx * dx/dt
            # and dx/dt == 0.
            result_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=gradient.properties,
                ),
            )
            continue

        gradient_samples = gradient.samples
        # here we need to copy because we want to modify the samples array
        samples = _dispatch.copy(gradient_samples.values)

        # change the first columns of the samples array with the mapping
        # between samples and gradient.samples
        samples[:, 0] = index[_dispatch.to_index_array(samples[:, 0])]

        new_gradient_samples, index_gradient = _dispatch.unique_with_inverse(
            samples[:, :], axis=0
        )
        index_gradient = index_gradient.reshape(-1)

        gradient_values = gradient.values
        other_shape = gradient_values.shape[1:]
        gradient_values_result = _dispatch.zeros_like(
            gradient_values,
            shape=(new_gradient_samples.shape[0],) + other_shape,
        )
        _dispatch.index_add(gradient_values_result, gradient_values, index_gradient)

        if reduction == "mean" or reduction == "var" or reduction == "std":
            bincount = _dispatch.bincount(index_gradient)
            gradient_values_result = gradient_values_result / bincount.reshape(
                (-1,) + (1,) * len(other_shape)
            )
            if reduction == "std" or reduction == "var":
                values_times_gradient_values = _dispatch.zeros_like(gradient_values)

                for i in range(gradient.samples.values.shape[0]):
                    s = gradient.samples.entry(i)
                    values_times_gradient_values[i] = (
                        gradient_values[i] * block_values[int(s[0])]
                    )

                values_grad_result = _dispatch.zeros_like(
                    gradient_values,
                    shape=(new_gradient_samples.shape[0],) + other_shape,
                )
                _dispatch.index_add(
                    values_grad_result,
                    values_times_gradient_values,
                    index_gradient,
                )

                values_grad_result = values_grad_result / bincount.reshape(
                    (-1,) + (1,) * len(other_shape)
                )
                if reduction == "var":
                    for i, s in enumerate(new_gradient_samples):
                        gradient_values_result[i] = (
                            gradient_values_result[i] * values_mean[int(s[0])]
                        )
                    gradient_values_result = 2 * (
                        values_grad_result - gradient_values_result
                    )
                else:  # std
                    for i, s in enumerate(new_gradient_samples):
                        sample = int(s[0])
                        if torch_jit_is_scripting():
                            gradient_values_result[i] = (
                                values_grad_result[i]
                                - (gradient_values_result[i] * values_mean[sample])
                            ) / values_result[sample]
                        else:
                            # only numpy raise a warning for division by zero
                            with np.errstate(divide="ignore", invalid="ignore"):
                                gradient_values_result[i] = (
                                    values_grad_result[i]
                                    - (gradient_values_result[i] * values_mean[sample])
                                ) / values_result[sample]

                        gradient_values_result[i] = _dispatch.nan_to_num(
                            gradient_values_result[i], nan=0.0, posinf=0.0, neginf=0.0
                        )

        # no check for the len of the gradient sample is needed because there
        # always will be at least one sample in the gradient
        result_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient_values_result,
                samples=Labels(gradient_samples.names, new_gradient_samples),
                components=gradient.components,
                properties=gradient.properties,
            ),
        )

    return result_block


def _reduce_over_samples(
    tensor: TensorMap, sample_names: Union[List[str], str], reduction: str
) -> TensorMap:
    """
    Create a new :py:class:`TensorMap` with the same keys as as the input
    ``tensor``, and each :py:class:`TensorBlock` is obtained summing the
    corresponding input :py:class:`TensorBlock` over the ``sample_names``
    indices.

    "sum", "mean", "std" or "var" reductions can be performed.

    :param tensor: input :py:class:`TensorMap`
    :param sample_names: names of samples to reduce over
    :param reduction: how to reduce, only available values are "mean", "sum",
    "std" or "var"
    """
    if isinstance(sample_names, str):
        sample_names_list = [sample_names]
    else:
        sample_names_list = sample_names

    for sample in sample_names_list:
        if sample not in tensor.sample_names:
            raise ValueError(
                f"one of the requested sample name ({sample}) is not part of "
                "this TensorMap"
            )

    remaining_samples: List[str] = []
    for s_name in tensor.sample_names:
        if s_name in sample_names_list:
            continue
        remaining_samples.append(s_name)

    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        blocks.append(
            _reduce_over_samples_block(
                block=block,
                sample_names=sample_names_list,
                reduction=reduction,
                remaining_samples=remaining_samples,
            )
        )
    return TensorMap(tensor.keys, blocks)


@torch_jit_script
def sum_over_samples_block(
    block: TensorBlock, sample_names: Union[List[str], str]
) -> TensorBlock:
    """Sum a :py:class:`TensorBlock`, combining the samples
    according to ``sample_names``.

    This function creates a new :py:class:`TensorBlock` in which each sample is
    obtained summing over the ``sample_names`` indices, so that the resulting
    :py:class:`TensorBlock` does not have those indices.

    ``sample_names`` indicates over which dimensions in the samples the sum is
    performed. It accept either a single string or a list of the string with the
    sample names corresponding to the directions along which the sum is
    performed. A single string is equivalent to a list with a single element:
    ``sample_names = "atom"`` is the same as ``sample_names = ["atom"]``.

    :param block:
        input :py:class:`TensorBlock`
    :param sample_names:
        names of samples to sum over

    :returns:
        a :py:class:`TensorBlock` containing the reduced values and sample labels

    >>> from metatensor import Labels, TensorBlock, TensorMap
    >>> block = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 4],
    ...             [3, 5, 6],
    ...             [7, 8, 9],
    ...             [10, 11, 12],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...                 [1, 0],
    ...                 [1, 1],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels.range("properties", 3),
    ... )
    >>> block_sum = sum_over_samples_block(block, sample_names="atom")
    >>> print(block_sum.samples)
    Labels(
        system
          0
          1
    )
    >>> print(block_sum.values)
    [[ 4  7 10]
     [17 19 21]]
    """

    return _reduce_over_samples_block(
        block=block, sample_names=sample_names, reduction="sum"
    )


@torch_jit_script
def sum_over_samples(
    tensor: TensorMap, sample_names: Union[List[str], str]
) -> TensorMap:
    """
    Sum a :py:class:`TensorMap`, combining the samples according to ``sample_names``.

    This function creates a new :py:class:`TensorMap` with the same keys
    as as the input ``tensor``. Each :py:class:`TensorBlock` is obtained summing the
    corresponding input :py:class:`TensorBlock` over the ``sample_names``
    indices, essentially calling :py:func:`sum_over_samples_block` over each
    block in ``tensor``.

    ``sample_names`` indicates over which dimensions in the samples the sum is
    performed. It accept either a single string or a list of the string with the
    sample names corresponding to the directions along which the sum is
    performed. A single string is equivalent to a list with a single element:
    ``sample_names = "atom"`` is the same as ``sample_names = ["atom"]``.

    :param tensor:
        input :py:class:`TensorMap`
    :param sample_names:
        names of samples to sum over

    :returns:
        a :py:class:`TensorMap` containing the reduced values and sample labels

    >>> from metatensor import Labels, TensorBlock, TensorMap
    >>> block = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 4],
    ...             [3, 5, 6],
    ...             [7, 8, 9],
    ...             [10, 11, 12],
    ...         ]
    ...     ),
    ...     samples=Labels(
    ...         ["system", "atom"],
    ...         np.array(
    ...             [
    ...                 [0, 0],
    ...                 [0, 1],
    ...                 [1, 0],
    ...                 [1, 1],
    ...             ]
    ...         ),
    ...     ),
    ...     components=[],
    ...     properties=Labels.range("properties", 3),
    ... )
    >>> keys = Labels(names=["key"], values=np.array([[0]]))
    >>> tensor = TensorMap(keys, [block])
    >>> tensor_sum = sum_over_samples(tensor, sample_names="atom")
    >>> # only 'system' is left as a sample
    >>> print(tensor_sum.block(0))
    TensorBlock
        samples (2): ['system']
        components (): []
        properties (3): ['properties']
        gradients: None
    >>> print(tensor_sum.block(0).samples)
    Labels(
        system
          0
          1
    )
    >>> print(tensor_sum.block(0).values)
    [[ 4  7 10]
     [17 19 21]]
    """

    return _reduce_over_samples(
        tensor=tensor, sample_names=sample_names, reduction="sum"
    )


@torch_jit_script
def mean_over_samples_block(
    block: TensorBlock, sample_names: Union[List[str], str]
) -> TensorBlock:
    """Averages a :py:class:`TensorBlock`, combining the samples according
    to ``sample_names``.

    See also :py:func:`sum_over_samples_block` and :py:func:`mean_over_samples`

    :param block:
        input :py:class:`TensorBlock`
    :param sample_names:
        names of samples to average over

    :returns:
        a :py:class:`TensorBlock` containing the reduced values and sample labels
    """

    return _reduce_over_samples_block(
        block=block, sample_names=sample_names, reduction="sum"
    )


@torch_jit_script
def mean_over_samples(
    tensor: TensorMap, sample_names: Union[str, List[str]]
) -> TensorMap:
    """Compute the mean of a :py:class:`TensorMap`, combining the samples according to
    ``sample_names``.

    This function creates a new :py:class:`TensorMap` with the same keys as
    as the input ``tensor``, and each :py:class:`TensorBlock` is obtained
    averaging the corresponding input :py:class:`TensorBlock` over the ``sample_names``
    indices.

    ``sample_names`` indicates over which dimensions in the samples the mean is
    performed. It accept either a single string or a list of the string with the
    sample names corresponding to the directions along which the mean is performed.
    A single string is equivalent to a list with a single element:
    ``sample_names = "atom"`` is the same as ``sample_names = ["atom"]``.

    For a general discussion of reduction operations and a usage example see the
    doc for :py:func:`sum_over_samples`.

    :param tensor: input :py:class:`TensorMap`
    :param sample_names: names of samples to average over
    """

    return _reduce_over_samples(
        tensor=tensor, sample_names=sample_names, reduction="mean"
    )


@torch_jit_script
def std_over_samples_block(
    block: TensorBlock, sample_names: Union[List[str], str]
) -> TensorBlock:
    """Computes the standard deviation for a :py:class:`TensorBlock`,
    combining the samples according to ``sample_names``.

    See also :py:func:`sum_over_samples_block` and :py:func:`std_over_samples`

    :param block:
        input :py:class:`TensorBlock`
    :param sample_names:
        names of samples to compute the standard deviation for

    :returns:
        a :py:class:`TensorBlock` containing the reduced values and sample labels
    """

    return _reduce_over_samples_block(
        block=block, sample_names=sample_names, reduction="std"
    )


@torch_jit_script
def std_over_samples(
    tensor: TensorMap, sample_names: Union[str, List[str]]
) -> TensorMap:
    r"""Compute the standard deviation of a :py:class:`TensorMap`, combining the samples
    according to ``sample_names``.

    This function creates a new :py:class:`TensorMap` with the same keys as
    as the input ``tensor``, and each :py:class:`TensorBlock` is obtained
    performing the std deviation of the corresponding input :py:class:`TensorBlock`
    over the ``sample_names`` indices.

    ``sample_names`` indicates over which dimensions in the samples the mean is
    performed. It accept either a single string or a list of the string with the
    sample names corresponding to the directions along which the mean is performed.
    A single string is equivalent to a list with a single element:
    ``sample_names = "atom"`` is the same as ``sample_names = ["atom"]``.

    For a general discussion of reduction operations and a usage example see the
    doc for :py:func:`sum_over_samples()`.

    The gradient is implemented as follows:

    .. math::

        \nabla[Std(X)] = 0.5(\nabla[Var(X)])/Std(X)
        = (E[X \nabla X] - E[X]E[\nabla X])/Std(X)

    :param tensor: input :py:class:`TensorMap`
    :param sample_names: names of samples to perform the standart deviation over
    """

    return _reduce_over_samples(
        tensor=tensor, sample_names=sample_names, reduction="std"
    )


@torch_jit_script
def var_over_samples_block(
    block: TensorBlock, sample_names: Union[List[str], str]
) -> TensorBlock:
    """Computes the variance for a :py:class:`TensorBlock`,
    combining the samples according to ``sample_names``.

    See also :py:func:`sum_over_samples_block` and :py:func:`std_over_samples`

    :param block:
        input :py:class:`TensorBlock`
    :param sample_names:
        names of samples to compute the variance for

    :returns:
        a :py:class:`TensorBlock` containing the reduced values and sample labels
    """

    return _reduce_over_samples_block(
        block=block, sample_names=sample_names, reduction="var"
    )


@torch_jit_script
def var_over_samples(
    tensor: TensorMap, sample_names: Union[str, List[str]]
) -> TensorMap:
    r"""Compute the variance of a :py:class:`TensorMap`, combining the
    samples according to ``sample_names``.

    This function creates a new :py:class:`TensorMap` with the same keys as
    as the input ``tensor``, and each :py:class:`TensorBlock` is obtained
    performing the variance of the corresponding input :py:class:`TensorBlock`
    over the ``sample_names`` indices.

    ``sample_names`` indicates over which dimensions in the samples the mean is
    performed. It accept either a single string or a list of the string with the
    sample names corresponding to the directions along which the mean is performed.
    A single string is equivalent to a list with a single element:
    ``sample_names = "atom"`` is the same as ``sample_names = ["atom"]``.

    For a general discussion of reduction operations and a usage example see the
    doc for :py:func:`sum_over_samples`.

    The gradient is implemented as follow:

    .. math::

        \nabla[Var(X)] = 2(E[X \nabla X] - E[X]E[\nabla X])

    :param tensor: input :py:class:`TensorMap`
    :param sample_names: names of samples to perform the variance over
    """

    return _reduce_over_samples(
        tensor=tensor, sample_names=sample_names, reduction="var"
    )
