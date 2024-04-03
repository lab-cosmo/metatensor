from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    check_isinstance,
    torch_jit_is_scripting,
    torch_jit_script,
)
from ._dispatch import TorchTensor


def _slice_block(block: TensorBlock, axis: str, labels: Labels) -> TensorBlock:
    if axis == "samples":
        # only keep the same names as `labels`
        all_samples = block.samples.view(labels.names)
        # create an arrays of bools indicating which samples indices to keep
        samples_mask = _dispatch.bool_array_like(
            [all_samples.entry(i) in labels for i in range(len(all_samples))],
            block.values,
        )
        new_values = _dispatch.mask(block.values, 0, samples_mask)
        new_samples = Labels(
            block.samples.names,
            _dispatch.mask(block.samples.values, 0, samples_mask),
        )

        new_block = TensorBlock(
            values=new_values,
            samples=new_samples,
            components=block.components,
            properties=block.properties,
        )

        # Create a map from the previous samples indexes to the new sample indexes
        # to update the gradient samples

        # sample_map contains at position old_sample the index of the
        # corresponding new sample
        sample_map = _dispatch.int_array_like(
            int_list=[-1] * len(samples_mask),
            like=samples_mask,
        )
        last = 0
        for i, picked in enumerate(samples_mask):
            if picked:
                sample_map[i] = last
                last += 1

        for parameter, gradient in block.gradients():
            if len(gradient.gradients_list()) != 0:
                raise NotImplementedError("gradients of gradients are not supported")

            sample_column = gradient.samples.column("sample")
            if not isinstance(gradient.samples.values, TorchTensor) and isinstance(
                samples_mask, TorchTensor
            ):
                # Torch complains if `sample_column` is numpy since it tries to convert
                # it to a Tensor, but the numpy array is read-only. Making a copy
                # removes the read-only marker
                sample_column = sample_column.copy()

            # Create a samples filter for the Gradient TensorBlock
            grad_samples_mask = samples_mask[_dispatch.to_index_array(sample_column)]

            new_grad_samples_values = _dispatch.mask(
                gradient.samples.values, 0, grad_samples_mask
            )

            if new_grad_samples_values.shape[0] != 0:
                # update the "sample" column of the gradient samples
                # to refer to the new samples
                new_grad_samples_values[:, 0] = sample_map[
                    _dispatch.to_index_array(new_grad_samples_values[:, 0])
                ]

                new_grad_samples = Labels(
                    names=gradient.samples.names,
                    values=new_grad_samples_values,
                )
            else:
                new_grad_samples = Labels(
                    names=gradient.samples.names,
                    values=_dispatch.empty_like(
                        gradient.samples.values, [0, gradient.samples.values.shape[1]]
                    ),
                )

            new_grad_values = _dispatch.mask(gradient.values, 0, grad_samples_mask)
            # Add sliced gradient to the TensorBlock
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=new_grad_values,
                    samples=new_grad_samples,
                    components=gradient.components,
                    properties=new_block.properties,
                ),
            )

    else:
        assert axis == "properties"
        # only keep the same names as `labels`
        all_properties = block.properties.view(list(labels.names))
        # create an arrays of bools indicating which samples indices to keep
        properties_mask = _dispatch.bool_array_like(
            [all_properties.entry(i) in labels for i in range(len(all_properties))],
            block.values,
        )
        new_values = _dispatch.mask(
            block.values, len(block.values.shape) - 1, properties_mask
        )
        new_properties = Labels(
            block.properties.names,
            _dispatch.mask(block.properties.values, 0, properties_mask),
        )

        new_block = TensorBlock(
            values=new_values,
            samples=block.samples,
            components=block.components,
            properties=new_properties,
        )

        # Slice each Gradient TensorBlock and add to the new_block.
        for parameter, gradient in block.gradients():
            if len(gradient.gradients_list()) != 0:
                raise NotImplementedError("gradients of gradients are not supported")

            assert axis == "properties"
            new_grad_values = _dispatch.mask(
                gradient.values, len(gradient.values.shape) - 1, properties_mask
            )
            new_grad_samples = gradient.samples

            # Add sliced gradient to the TensorBlock
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=new_grad_values,
                    samples=new_grad_samples,
                    components=gradient.components,
                    properties=new_block.properties,
                ),
            )

    return new_block


def _check_args(
    block: TensorBlock,
    axis: str,
    labels: Labels,
):
    """
    Checks the arguments passed to :py:func:`slice` and :py:func:`slice_block`.
    """
    # check axis
    if axis not in ["samples", "properties"]:
        raise ValueError(
            f"``axis``: {axis} is not known as a slicing axis. Please use"
            "'samples' or 'properties'"
        )

    if not torch_jit_is_scripting():
        if not check_isinstance(labels, Labels):
            raise TypeError(f"`labels` must be metatensor Labels, not {type(labels)}")

    if axis == "samples":
        s_names = block.samples.names
        for name in labels.names:
            if name not in s_names:
                raise ValueError(
                    f"invalid sample name '{name}' which is not part of the input"
                )
    else:
        assert axis == "properties"
        p_names = block.properties.names
        for name in labels.names:
            if name not in p_names:
                raise ValueError(
                    f"invalid property name '{name}' which is not part of the input"
                )


@torch_jit_script
def slice(tensor: TensorMap, axis: str, labels: Labels) -> TensorMap:
    """
    Slice a :py:class:`TensorMap` along either the ``"samples"`` or ``"properties"`
    ``axis``. ``labels`` is a :py:class:`Labels` objects that specifies the
    samples/properties (respectively) names and indices that should be sliced, i.e. kept
    in the output :py:class:`TensorMap`.

    This function will return a :py:class:`TensorMap` whose blocks are of equal or
    smaller dimensions (due to slicing) than those of the input. However, the returned
    :py:class:`TensorMap` will be returned with the same number of blocks and the
    corresponding keys as the input. If any block upon slicing is reduced to nothing,
    i.e. in the case that it has none of the specified ``labels`` along the
    ``"samples"`` or ``"properties"`` ``axis``, an empty block (i.e. a block with one of
    the dimension set to 0) will used for this key, and a warning will be emitted.

    See the documentation for the :py:func:`slice_block` function to see how an
    individual :py:class:`TensorBlock` is sliced.

    :param tensor: the input :py:class:`TensorMap` to be sliced.
    :param axis: a :py:class:`str` indicating the axis along which slicing should occur.
        Should be either "samples" or "properties".
    :param labels: a :py:class:`Labels` object containing the names and indices of the
        "samples" or "properties" to keep in each of the sliced :py:class:`TensorBlock`
        of the output :py:class:`TensorMap`.

    :return: a :py:class:`TensorMap` that corresponds to the sliced input tensor.
    """
    # Check input args
    if not torch_jit_is_scripting():
        if not check_isinstance(tensor, TensorMap):
            raise TypeError(
                f"`tensor` must be a metatensor TensorMap, not {type(tensor)}"
            )

    _check_args(tensor.block(0), axis=axis, labels=labels)

    return TensorMap(
        keys=tensor.keys,
        blocks=[
            _slice_block(tensor[tensor.keys.entry(i)], axis, labels)
            for i in range(len(tensor.keys))
        ],
    )


@torch_jit_script
def slice_block(block: TensorBlock, axis: str, labels: Labels) -> TensorBlock:
    """
    Slices an input :py:class:`TensorBlock` along either the ``"samples"`` or
    ``"properties"`` ``axis``. ``labels`` is a :py:class:`Labels` objects that specify
    the sample/property names and indices that should be sliced, i.e. kept in the output
    :py:class:`TensorBlock`.

    If none of the entries in ``labels`` can be found in the ``block``, the dimension
    corresponding to ``axis`` will be sliced to 0, and the returned block with have a
    shape of either ``(0, n_components, n_properties)`` or ``(n_samples, n_components,
    0)``.

    :param block: the input :py:class:`TensorBlock` to be sliced.
    :param axis: a :py:class:`str` indicating the axis along which slicing should occur.
        Should be either "samples" or "properties".
    :param labels: a :py:class:`Labels` object containing the names and indices of the
        "samples" or "properties" to keep in the sliced output :py:class:`TensorBlock`.

    :return new_block: a :py:class:`TensorBlock` that corresponds to the sliced input.
    """
    if not torch_jit_is_scripting():
        if not check_isinstance(block, TensorBlock):
            raise TypeError(
                f"`block` must be a metatensor TensorBlock, not {type(block)}"
            )

    _check_args(block, axis=axis, labels=labels)

    return _slice_block(
        block,
        axis=axis,
        labels=labels,
    )
