import functools
import operator
from typing import List

import numpy as np

from ..block import TensorBlock
from ..labels import Labels
from ..tensor import TensorMap
from . import _dispatch
from .equal_metadata import _check_blocks, _check_maps, _check_same_gradients


def join(tensor_maps: List[TensorMap], axis: str):
    """Join a sequence of :py:class:`TensorMap` along an axis.

    The ``axis`` parameter specifies the type join. For example, if
    ``axis='properties'`` it will be the joined along the properties of the
    TensorMaps and if ``axis='samples'`` it will be the along the samples.

    If ``axis='properties'`` is specified, the components of the TensorBlocks
    have to be empty.

    .. code-block:: python

        joined_tensor = join([tensor_a,tensor_b],axis="samples")
        joined_tensor = join([tensor_a,tensor_b],axis="properties")

    Possible clashes of the meta data like the property names are resolved by
    one of the three following strategies:

    1. If the names are the same and the values are unique we keep the names and
       join the values vertically::

        labels_1 = ["n"], [0, 2, 3] labels_2 = ["n"], [1, 4, 5]

       will lead to::

        joined_labels = ["n"],
                       [0, 2, 3, 1, 4, 5]

    2. If the names are the same but the values are not unique a new dimension
       ``"tensor"`` is added to the names::

            labels_1 = ["n"], [0, 2, 3]
            labels_2 = ["n"], [0, 2]

       will lead to:

            joined_labels = ["tensor", "n"],
                            [[0, 0], [0, 2], [0, 3], [1, 0], [1, 2]]

       In ``joined_labels`` the label values ``[0, 0], [0, 2], [0, 3]``
       correspond to ``properties_1`` ``[1, 0], [1, 2]`` correspond to
       ``properties_2``.

    3. If names of the labels are different we change the names to ("tensor",
       "property"). This case is only supposed to happen when joining in the
       property dimension, hence the choice of names. For example::

            properties_1 = ["a"], [0, 2, 3]
            properties_2 = ["b", "c"], [[0, 0], [1, 2]]

       will lead to:

            joined_properties = ["tensor", "property"],
                         [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]

    :param tensormaps: sequence of :py:class:`TensorMap` for join
    :param axis: A string indicating how the tensormaps are stacked. Allowed
                 values are ``'properties'`` or ``'samples'``.

    :return: The stacked :py:class:`TensorMap` with more properties or samples
             than the input TensorMap.
    """

    if not isinstance(tensor_maps, (list, tuple)):
        raise TypeError(
            "the `TensorMap`s to join must be provided as a list or a tuple"
        )

    if len(tensor_maps) < 1:
        raise ValueError("provide at least one `TensorMap` for joining")

    if len(tensor_maps) == 1:
        return tensor_maps[0]

    for ts_to_join in tensor_maps[1:]:
        _check_maps(tensor_maps[0], ts_to_join, "join")

    blocks = []
    for key in tensor_maps[0].keys:
        blocks_to_join = [ts.block(key) for ts in tensor_maps]

        if axis == "properties":
            blocks.append(_join_blocks_along_properties(blocks_to_join))
        elif axis == "samples":
            blocks.append(_join_blocks_along_samples(blocks_to_join))
        else:
            raise ValueError(
                "Only `'properties'` or `'samples'` are "
                "valid values for the `axis` parameter."
            )

    return TensorMap(tensor_maps[0].keys, blocks)


def _join_labels(labels: List[Labels]) -> Labels:
    """Join a sequence of :py:class:`Labels`"""
    names_list = [label.names for label in labels]
    values_list = [label.tolist() for label in labels]
    tensor_values = np.repeat(
        a=np.arange(len(values_list)), repeats=[len(value) for value in values_list]
    )

    # We use functools to flatten a list of sublists::
    #
    #   [('a', 'b', 'c'), ('a', 'b')] -> ['a', 'b', 'c', 'a', 'b']
    #
    # A nested list with sublist of different shapes can not be handled by np.unique.
    unique_names = np.unique(functools.reduce(operator.concat, names_list))

    # Label names are unique: We can do an equal check only checking the lengths.
    names_are_same = np.all(
        len(unique_names) == np.array([len(names) for names in names_list])
    )

    if names_are_same:
        unique_values = np.unique(np.vstack(values_list), axis=0)
        values_are_unique = len(unique_values) == len(np.vstack(values_list))
        if values_are_unique:
            new_names = names_list[0]
            new_values = np.vstack(values_list)
        else:
            new_names = ["tensor"] + list(names_list[0])
            new_values = np.hstack(
                [tensor_values.reshape(-1, 1), np.vstack(values_list)]
            )
    else:
        new_names = ["tensor", "property"]
        property_values = np.hstack([np.arange(len(values)) for values in values_list])
        new_values = np.vstack([tensor_values, property_values]).T

    return Labels(names=new_names, values=new_values)


def _join_blocks_along_properties(blocks: List[TensorBlock]) -> TensorBlock:
    """Join a sequence of :py:class:`TensorBlock` along their properties."""
    first_block = blocks[0]

    fname = "_join_blocks_along_properties"
    for block in blocks[1:]:
        _check_blocks(first_block, block, ["components", "samples"], fname)
        _check_same_gradients(first_block, block, ["components", "samples"], fname)

    properties = _join_labels([block.properties for block in blocks])
    result_block = TensorBlock(
        values=_dispatch.hstack([block.values for block in blocks]),
        samples=first_block.samples,
        components=first_block.components,
        properties=properties,
    )

    for parameter, first_gradient in first_block.gradients():
        # Different blocks might contain different gradient samples
        gradient_sample_values = np.unique(
            np.hstack([block.gradient(parameter).samples for block in blocks])
        ).tolist()
        gradient_samples = Labels(
            names=first_gradient.samples.names, values=np.array(gradient_sample_values)
        )

        gradient_data = np.zeros(
            (len(gradient_samples),)
            + first_gradient.data.shape[1:-1]
            + (len(properties),)
        )

        properties_start = 0
        for block in blocks:
            gradient = block.gradient(parameter)

            # find the correct sample position to put the data
            for i_grad, sample_grad in enumerate(gradient.samples):
                i_sample = gradient_samples.position(sample_grad)
                gradient_data[
                    i_sample,
                    ...,
                    properties_start : properties_start + len(gradient.properties),
                ] = gradient.data[i_grad]

            properties_start += len(gradient.properties)

        result_block.add_gradient(
            parameter, gradient_data, gradient_samples, first_gradient.components
        )

    return result_block


def _join_blocks_along_samples(blocks: List[TensorBlock]) -> TensorBlock:
    """Join a sequence of :py:class:`TensorBlock` along their samples."""
    first_block = blocks[0]

    fname = "_join_blocks_along_samples"
    for block in blocks[1:]:
        _check_blocks(first_block, block, ["components", "properties"], fname)
        _check_same_gradients(first_block, block, ["components", "properties"], fname)

    result_block = TensorBlock(
        values=_dispatch.vstack([block.values for block in blocks]),
        samples=_join_labels([block.samples for block in blocks]),
        components=first_block.components,
        properties=first_block.properties,
    )

    for parameter, first_gradient in first_block.gradients():
        gradient_data = [first_gradient.data]
        gradient_samples = [np.array(first_gradient.samples.tolist())]

        for block in blocks[1:]:
            gradient = block.gradient(parameter)
            gradient_data.append(gradient.data)

            samples = np.array(gradient.samples.tolist())
            # update the "sample" dimension in gradient samples
            samples[:, 0] += 1 + gradient_samples[-1][:, 0].max()
            gradient_samples.append(samples)

        gradient_data = _dispatch.vstack(gradient_data)
        gradient_samples = np.vstack(gradient_samples)

        gradients_samples = Labels(first_gradient.samples.names, gradient_samples)

        result_block.add_gradient(
            parameter,
            gradient_data,
            gradients_samples,
            first_gradient.components,
        )

    return result_block
