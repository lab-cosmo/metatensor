"""
Manipulating TensorMap dimensions
=================================

Functions for manipulating dimensions of an :py:class:`metatensor.TensorMap` (i.e.
changing the columns of the :py:class:`metatensor.Labels` within).

.. autofunction:: metatensor.append_dimension

.. autofunction:: metatensor.insert_dimension

.. autofunction:: metatensor.permute_dimensions

.. autofunction:: metatensor.remove_dimension

.. autofunction:: metatensor.rename_dimension
"""

from typing import List, Union

from . import _dispatch
from ._backend import Array, TensorBlock, TensorMap, torch_jit_script


def _check_axis(axis: str):
    if axis not in ["keys", "samples", "properties"]:
        raise ValueError(
            f"'{axis}' is not a valid axis. Choose from 'keys', 'samples' or "
            "'properties'."
        )


@torch_jit_script
def insert_dimension(
    tensor: TensorMap,
    axis: str,
    index: int,
    name: str,
    values: Union[Array, int],
) -> TensorMap:
    """Insert a :py:class:`metatensor.Labels` dimension along the given axis before the
    given index.

    For ``axis=="samples"`` a new dimension is `not` appended to gradients.

    :param tensor: the input :py:class:`TensorMap`.
    :param axis: axis for which the ``name`` should be inserted. Allowed are ``"keys"``,
        ``"properties"`` or ``"samples"``.
    :param index: index before the new dimension is inserted.
    :param name: the name to be inserted
    :param values: values to be inserted. This can be an array (``np.array`` or
        ``torch.Tensor`` according to whether ``metatensor`` or ``metatensor.torch`` is
        being used); or an integer. In the latter case, the new dimension will have the
        given integer value for all entries in the labels

    :raises ValueError: if ``axis`` is a not valid value

    :return: a new :py:class:`metatensor.TensorMap` with inserted labels dimension.

    Examples
    --------

    >>> import numpy as np
    >>> import metatensor
    >>> values = np.array([[1, 2], [3, 4]])
    >>> block = metatensor.block_from_array(values)
    >>> keys = metatensor.Labels(["foo"], np.array([[0]]))
    >>> tensor = metatensor.TensorMap(keys=keys, blocks=[block])
    >>> tensor
    TensorMap with 1 blocks
    keys: foo
           0
    >>> metatensor.insert_dimension(
    ...     tensor,
    ...     axis="keys",
    ...     index=0,
    ...     name="bar",
    ...     values=np.array([1]),
    ... )
    TensorMap with 1 blocks
    keys: bar  foo
           1    0
    """
    _check_axis(axis)

    keys = tensor.keys

    if isinstance(values, int):
        values = _dispatch.int_array_like([values], keys.values)
        label_values = values
        values_was_int = True
    else:
        label_values = values
        values_was_int = False

    if axis == "keys":
        if values_was_int:
            label_values = _dispatch.concatenate([values] * len(keys), axis=0)

        keys = keys.insert(index=index, name=name, values=label_values)

    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        samples = block.samples
        properties = block.properties

        if axis == "samples":
            if values_was_int:
                label_values = _dispatch.concatenate([values] * len(samples), axis=0)

            samples = samples.insert(index=index, name=name, values=label_values)

        elif axis == "properties":
            if values_was_int:
                label_values = _dispatch.concatenate([values] * len(properties), axis=0)

            properties = properties.insert(index=index, name=name, values=label_values)

        new_block = TensorBlock(
            values=block.values,
            samples=samples,
            components=block.components,
            properties=properties,
        )

        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=properties,
                ),
            )

        blocks.append(new_block)

    return TensorMap(keys=keys, blocks=blocks)


@torch_jit_script
def append_dimension(
    tensor: TensorMap,
    axis: str,
    name: str,
    values: Union[Array, int],
) -> TensorMap:
    """Append a :py:class:`metatensor.Labels` dimension along the given axis.

    For ``axis=="samples"`` the new dimension is `not` appended to gradients.

    :param tensor: the input :py:class:`TensorMap`.
    :param axis: axis for which the ``name`` should be appended. Allowed are ``"keys"``,
        ``"properties"`` or ``"samples"``.
    :param name: name of the dimension be appended
    :param values: values of the dimension to be appended (``np.array`` or
        ``torch.Tensor`` according to whether ``metatensor`` or ``metatensor.torch`` is
        being used); or an integer. In the latter case, the new dimension will have the
        given integer value for all entries in the labels

    :raises ValueError: if ``axis`` is a not valid value

    :return: a new :py:class:`metatensor.TensorMap` with appended labels dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import metatensor
    >>> values = np.array([[1, 2], [3, 4]])
    >>> block = metatensor.block_from_array(values)
    >>> keys = metatensor.Labels(["foo"], np.array([[0]]))
    >>> tensor = metatensor.TensorMap(keys=keys, blocks=[block])
    >>> tensor
    TensorMap with 1 blocks
    keys: foo
           0
    >>> metatensor.append_dimension(
    ...     tensor,
    ...     axis="keys",
    ...     name="bar",
    ...     values=np.array([1]),
    ... )
    TensorMap with 1 blocks
    keys: foo  bar
           0    1
    """
    if axis == "keys":
        index = len(tensor.keys.names)
        return insert_dimension(tensor, axis, index, name, values)
    elif axis == "samples":
        index = len(tensor.sample_names)
        return insert_dimension(tensor, axis, index, name, values)
    elif axis == "properties":
        index = len(tensor.property_names)
        return insert_dimension(tensor, axis, index, name, values)
    else:
        raise ValueError(
            f"'{axis}' is not a valid axis. Choose from 'keys', 'samples' or "
            "'properties'."
        )


@torch_jit_script
def permute_dimensions(
    tensor: TensorMap, axis: str, dimensions_indexes: List[int]
) -> TensorMap:
    """Permute dimensions of a :py:class:`Labels` of the given axis according to a
    :py:class:`list` of indexes.

    Values of ``dimensions_indexes`` have to be same as the indexes of
    :py:class:`Labels` but can be in a different order.

    For ``axis=="samples"`` gradients samples dimensions are not permuted.

    :param tensor: the input :py:class:`TensorMap`.
    :param axis: axis for which the ``name`` should be inserted. Allowed are ``"keys"``,
                 ``"properties"`` or ``"samples"``.
    :param dimensions_indexes: desired ordering of the dimensions

    :raises ValueError: if ``axis`` is a not valid value

    :return: a new :py:class:`metatensor.TensorMap` with the labels dimension permuted

    Examples
    --------
    >>> import numpy as np
    >>> import metatensor
    >>> values = np.array([[1, 2], [3, 4]])
    >>> block = metatensor.block_from_array(values)
    >>> keys = metatensor.Labels(["foo", "bar", "baz"], np.array([[42, 10, 3]]))
    >>> tensor = metatensor.TensorMap(keys=keys, blocks=[block])
    >>> tensor
    TensorMap with 1 blocks
    keys: foo  bar  baz
          42   10    3

    Move the last (second) dimension to the first position.

    >>> metatensor.permute_dimensions(tensor, axis="keys", dimensions_indexes=[2, 0, 1])
    TensorMap with 1 blocks
    keys: baz  foo  bar
           3   42   10
    """
    _check_axis(axis)

    keys = tensor.keys
    if axis == "keys":
        keys = keys.permute(dimensions_indexes=dimensions_indexes)

    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        samples = block.samples
        properties = block.properties

        if axis == "samples":
            samples = samples.permute(dimensions_indexes=dimensions_indexes)
        elif axis == "properties":
            properties = properties.permute(dimensions_indexes=dimensions_indexes)

        new_block = TensorBlock(
            values=block.values,
            samples=samples,
            components=block.components,
            properties=properties,
        )

        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=properties,
                ),
            )

        blocks.append(new_block)

    return TensorMap(keys=keys, blocks=blocks)


@torch_jit_script
def remove_dimension(tensor: TensorMap, axis: str, name: str) -> TensorMap:
    """Remove a :py:class:`metatensor.Labels` dimension along the given axis.

    Removal can only be performed if the resulting :py:class:`metatensor.Labels`
    instance will be unique.

    For ``axis=="samples"`` the dimension is not removed from gradients.

    :param tensor: the input :py:class:`TensorMap`.
    :param axis: axis for which ``name`` should be removed. Allowed are ``"keys"``,
                 ``"properties"`` or ``"samples"``.
    :param name: the :py:class:`metatensor.Labels` name to be removed

    :raises ValueError: if ``axis`` is a not valid value
    :raises ValueError: if the only dimension should be removed
    :raises ValueError: if name is not part of the axis

    :return: a new :py:class:`metatensor.TensorMap` with removed labels dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import metatensor
    >>> values = np.array([[1, 2], [3, 4]])
    >>> block = metatensor.block_from_array(values)
    >>> keys = metatensor.Labels(["key", "extra"], np.array([[0, 0]]))
    >>> tensor = metatensor.TensorMap(keys=keys, blocks=[block])
    >>> tensor
    TensorMap with 1 blocks
    keys: key  extra
           0     0
    >>> metatensor.remove_dimension(tensor, axis="keys", name="extra")
    TensorMap with 1 blocks
    keys: key
           0

    Removing a dimension can only be performed if the resulting :py:class:`Labels` will
    contain unique entries.

    >>> from metatensor import MetatensorError
    >>> block = metatensor.block_from_array(values)
    >>> keys = metatensor.Labels(["key", "extra"], np.array([[0, 0], [0, 1]]))
    >>> tensor = metatensor.TensorMap(keys=keys, blocks=[block.copy(), block.copy()])
    >>> tensor
    TensorMap with 2 blocks
    keys: key  extra
           0     0
           0     1
    >>> try:
    ...     metatensor.remove_dimension(tensor, axis="keys", name="extra")
    ... except MetatensorError as e:
    ...     print(e)
    ...
    invalid parameter: can not have the same label value multiple time: [0] is already present at position 0
    """  # noqa E501
    _check_axis(axis)

    keys = tensor.keys
    if axis == "keys":
        keys = keys.remove(name=name)

    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        samples = block.samples
        properties = block.properties

        if axis == "samples":
            samples = samples.remove(name)
        elif axis == "properties":
            properties = properties.remove(name)

        new_block = TensorBlock(
            values=block.values,
            samples=samples,
            components=block.components,
            properties=properties,
        )

        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=properties,
                ),
            )

        blocks.append(new_block)

    return TensorMap(keys=keys, blocks=blocks)


@torch_jit_script
def rename_dimension(tensor: TensorMap, axis: str, old: str, new: str) -> TensorMap:
    """Rename a :py:class:`metatensor.Labels` dimension name for a given axis.

    :param tensor: the input :py:class:`TensorMap`.
    :param axis: axis for which the names should be appended. Allowed are ``"keys"``,
                 ``"properties"`` or ``"samples"``.
    :param old: name to be replaced
    :param new: name after the replacement

    :raises ValueError: if ``axis`` is a not valid value

    :return: a `new` :py:class:`metatensor.TensorMap` with renamed labels dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import metatensor
    >>> values = np.array([[1, 2], [3, 4]])
    >>> block = metatensor.block_from_array(values)
    >>> keys = metatensor.Labels(["foo"], np.array([[0]]))
    >>> tensor = metatensor.TensorMap(keys=keys, blocks=[block])
    >>> tensor
    TensorMap with 1 blocks
    keys: foo
           0
    >>> metatensor.rename_dimension(tensor, axis="keys", old="foo", new="bar")
    TensorMap with 1 blocks
    keys: bar
           0

    """
    _check_axis(axis)

    keys = tensor.keys
    if axis == "keys":
        keys = keys.rename(old, new)

    blocks: List[TensorBlock] = []
    for block in tensor.blocks():
        samples = block.samples
        properties = block.properties

        if axis == "samples":
            samples = samples.rename(old, new)
        elif axis == "properties":
            properties = properties.rename(old, new)

        new_block = TensorBlock(
            values=block.values,
            samples=samples,
            components=block.components,
            properties=properties,
        )

        for parameter, gradient in block.gradients():
            gradient_samples = gradient.samples
            if axis == "samples" and old in gradient_samples.names:
                gradient_samples = gradient_samples.rename(old, new)

            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient_samples,
                    components=gradient.components,
                    properties=properties,
                ),
            )

        blocks.append(new_block)

    return TensorMap(keys=keys, blocks=blocks)
