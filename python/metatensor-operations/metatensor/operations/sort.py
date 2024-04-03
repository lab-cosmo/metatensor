from typing import List, Union

from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    torch_jit_annotate,
    torch_jit_is_scripting,
    torch_jit_script,
)


def _sort_single_gradient_block(
    block: TensorBlock,
    gradient_block: TensorBlock,
    axes: List[str],
    descending: bool,
) -> TensorBlock:
    """
    Sorts a single gradient tensor block given the tensor block which the gradients are
    attached to. This function does not check the user input.  This is different from
    `_sort_single_block` because we need to update the sample differently (since
    gradient samples are pointers into the values samples).
    """

    sample_names = gradient_block.samples.names
    sample_values = gradient_block.samples.values

    component_names: List[List[str]] = []
    components_values = []
    for component in gradient_block.components:
        component_names.append(component.names)
        components_values.append(component.values)

    property_names = gradient_block.properties.names
    properties_values = gradient_block.properties.values

    values = gradient_block.values
    if "samples" in axes:
        # we first need to get the mapping induced by the sorting in its parent block
        # so we can change the sample column label entries so it matches the ones of
        # the parent block
        block_sample_values = block.samples.values
        # sample index -> sample labels
        sorted_idx = _dispatch.argsort_labels_values(
            block_sample_values, reverse=descending
        )
        # obtain inverse mapping sample labels -> sample index
        sorted_idx_inverse = _dispatch.empty_like(sorted_idx, shape=(len(sorted_idx),))
        sorted_idx_inverse[sorted_idx] = _dispatch.int_array_like(
            list(range(len(sorted_idx))), sorted_idx
        )
        # adapt sample column in gradient samples to the one of the sorted values of
        # the gradient_block the gradient is attached to
        sample_values = _dispatch.copy(sample_values)
        sample_values[:, 0] = sorted_idx_inverse[
            _dispatch.to_index_array(sample_values[:, 0])
        ]

        # sort the samples in gradient regularly moving the rows considering all columns
        sorted_idx = _dispatch.argsort_labels_values(sample_values, reverse=descending)
        sample_values = sample_values[sorted_idx]
        values = values[sorted_idx]
    if "components" in axes:
        for i, _ in enumerate(gradient_block.components):
            sorted_idx = _dispatch.argsort_labels_values(
                components_values[i], reverse=descending
            )
            components_values[i] = components_values[i][sorted_idx]
            values = _dispatch.take(values, sorted_idx, axis=i + 1)
    if "properties" in axes:
        sorted_idx = _dispatch.argsort_labels_values(
            properties_values, reverse=descending
        )
        properties_values = properties_values[sorted_idx]
        values = _dispatch.take(values, sorted_idx, axis=-1)

    samples_labels = Labels(names=sample_names, values=sample_values)
    properties_labels = Labels(names=property_names, values=properties_values)
    components_labels = [
        Labels(names=component_names[i], values=components_values[i])
        for i in range(len(component_names))
    ]

    return TensorBlock(
        values=values,
        samples=samples_labels,
        components=components_labels,
        properties=properties_labels,
    )


def _sort_single_block(
    block: TensorBlock,
    axes: List[str],
    descending: bool,
) -> TensorBlock:
    """
    Sorts a single TensorBlock without the user input checking and sorting of gradients
    """

    sample_names = block.samples.names
    sample_values = block.samples.values

    component_names: List[List[str]] = []
    components_values = []
    for component in block.components:
        component_names.append(component.names)
        components_values.append(component.values)

    property_names = block.properties.names
    properties_values = block.properties.values

    values = block.values
    if "samples" in axes:
        sorted_idx = _dispatch.argsort_labels_values(sample_values, reverse=descending)
        sample_values = sample_values[sorted_idx]
        values = values[sorted_idx]
    if "components" in axes:
        for i, _ in enumerate(block.components):
            sorted_idx = _dispatch.argsort_labels_values(
                components_values[i], reverse=descending
            )
            components_values[i] = components_values[i][sorted_idx]
            values = _dispatch.take(values, sorted_idx, axis=i + 1)
    if "properties" in axes:
        sorted_idx = _dispatch.argsort_labels_values(
            properties_values, reverse=descending
        )
        properties_values = properties_values[sorted_idx]
        values = _dispatch.take(values, sorted_idx, axis=-1)

    samples_labels = Labels(names=sample_names, values=sample_values)
    properties_labels = Labels(names=property_names, values=properties_values)
    components_labels = [
        Labels(names=component_names[i], values=components_values[i])
        for i in range(len(component_names))
    ]

    return TensorBlock(
        values=values,
        samples=samples_labels,
        components=components_labels,
        properties=properties_labels,
    )


@torch_jit_script
def sort_block(
    block: TensorBlock,
    axes: Union[str, List[str]] = "all",
    descending: bool = False,
) -> TensorBlock:
    """
    Rearrange the values of a block according to the order given by the sorted metadata
    of the given axes.

    This function creates copies of the metadata on the CPU to sort the metadata.

    :param axes: axes to sort. The labels entries along these axes will be sorted in
        lexicographic order, and the arrays values will be reordered accordingly.
        Possible values are ``'samples'``, ``'components'``, ``'properties'`` and
        ``'all'`` to sort everything.

    :param descending: if false, the order is ascending

    :return: sorted tensor block

    >>> import numpy as np
    >>> import metatensor
    >>> from metatensor import TensorBlock, TensorMap, Labels
    >>> block = TensorBlock(
    ...     values=np.arange(9).reshape(3, 3),
    ...     samples=Labels(["system", "atom"], np.array([[0, 3], [0, 1], [0, 2]])),
    ...     components=[],
    ...     properties=Labels(["n", "l"], np.array([[2, 0], [3, 0], [1, 0]])),
    ... )
    >>> print(block)
    TensorBlock
        samples (3): ['system', 'atom']
        components (): []
        properties (3): ['n', 'l']
        gradients: None
    >>> # sorting axes one by one
    >>> block_sorted_stepwise = metatensor.sort_block(block, axes=["properties"])
    >>> # properties (last dimension of the array) are sorted
    >>> block_sorted_stepwise.values
    array([[2, 0, 1],
           [5, 3, 4],
           [8, 6, 7]])
    >>> block_sorted_stepwise = metatensor.sort_block(
    ...     block_sorted_stepwise, axes=["samples"]
    ... )
    >>> # samples (first dimension of the array) are sorted
    >>> block_sorted_stepwise.values
    array([[5, 3, 4],
           [8, 6, 7],
           [2, 0, 1]])
    >>> # sorting both samples and properties at the same time
    >>> sorted_block = metatensor.sort_block(block)
    >>> np.all(sorted_block.values == block_sorted_stepwise.values)
    True
    >>> # This function can also sort gradients:
    >>> sorted_block.add_gradient(
    ...     parameter="g",
    ...     gradient=TensorBlock(
    ...         values=np.arange(18).reshape(3, 2, 3),
    ...         samples=Labels(["sample"], np.array([[1], [2], [0]])),
    ...         components=[Labels.range("direction", 2)],
    ...         properties=sorted_block.properties,
    ...     ),
    ... )
    >>> sorted_grad_block = metatensor.sort_block(sorted_block)
    >>> sorted_grad_block.gradient("g").samples == Labels.range("sample", 3)
    True
    >>> sorted_grad_block.gradient("g").properties == sorted_block.properties
    True
    >>> # the components (middle dimensions) are also sorted:
    >>> sorted_grad_block.gradient("g").values
    array([[[12, 13, 14],
            [15, 16, 17]],
    <BLANKLINE>
           [[ 0,  1,  2],
            [ 3,  4,  5]],
    <BLANKLINE>
           [[ 6,  7,  8],
            [ 9, 10, 11]]])

    """
    if isinstance(axes, str):
        if axes == "all":
            axes_list = ["samples", "components", "properties"]
        else:
            axes_list = [axes]
    elif isinstance(axes, list):
        axes_list = axes
    else:
        if torch_jit_is_scripting():
            extra = ""
        else:
            extra = f", not {type(axes)}"

        raise TypeError("'axes' should be a string or list of strings" + extra)

    for axis in axes_list:
        if axis not in ["samples", "components", "properties"]:
            raise ValueError(
                "`axes` must be one of 'samples', 'components' or 'properties', "
                f"not '{axis}'"
            )

    result_block = _sort_single_block(block, axes_list, descending)

    for parameter, gradient in block.gradients():
        if len(gradient.gradients_list()) != 0:
            raise NotImplementedError("gradients of gradients are not supported")

        result_block.add_gradient(
            parameter=parameter,
            gradient=_sort_single_gradient_block(
                block, gradient, axes_list, descending
            ),
        )

    return result_block


@torch_jit_script
def sort(
    tensor: TensorMap,
    axes: Union[str, List[str]] = "all",
    descending: bool = False,
) -> TensorMap:
    """
    Sort the ``tensor`` according to the key values and the blocks for each specified
    axis in ``axes`` according to the label values along these axes.

    Each block is sorted separately, see :py:func:`sort_block` for more information

    Note: This function duplicates metadata on the CPU for the purpose of sorting.

    :param axes: axes to sort. The labels entries along these axes will be sorted in
        lexicographic order, and the arrays values will be reordered accordingly.
        Possible values are ``'keys'``, ``'samples'``, ``'components'``,
        ``'properties'`` and ``'all'`` to sort everything.
    :param descending: if false, the order is ascending
    :return: sorted tensor map

    >>> import numpy as np
    >>> import metatensor
    >>> from metatensor import TensorBlock, TensorMap, Labels
    >>> block_1 = TensorBlock(
    ...     values=np.arange(9).reshape(3, 3),
    ...     samples=Labels(["system", "atom"], np.array([[0, 3], [0, 1], [0, 2]])),
    ...     components=[],
    ...     properties=Labels(["n", "l"], np.array([[1, 0], [2, 0], [0, 0]])),
    ... )
    >>> block_2 = TensorBlock(
    ...     values=np.arange(3).reshape(1, 3),
    ...     samples=Labels(["system", "atom"], np.array([[0, 0]])),
    ...     components=[],
    ...     properties=Labels(["n", "l"], np.array([[1, 0], [2, 0], [0, 0]])),
    ... )
    >>> tensor = TensorMap(
    ...     keys=Labels(["types"], np.array([[1], [0]])), blocks=[block_2, block_1]
    ... )
    >>> metatensor.sort(tensor, axes="keys")
    TensorMap with 2 blocks
    keys: types
            0
            1
    """
    if isinstance(axes, str):
        axes_list: List[str] = []
        if axes == "all":
            axes_list = ["samples", "components", "properties"]
            sort_keys = True
        elif axes == "keys":
            axes_list = torch_jit_annotate(List[str], [])
            sort_keys = True
        else:
            axes_list = [axes]
            sort_keys = False

    elif isinstance(axes, list):
        axes_list = axes

        if "keys" in axes_list:
            keys_index = axes_list.index("keys")
            sort_keys = True
            axes_list.pop(keys_index)
        else:
            sort_keys = False

    else:
        if torch_jit_is_scripting():
            extra = ""
        else:
            extra = f", not {type(axes)}"

        raise TypeError("'axes' should be a string or list of strings" + extra)

    # Do we need to sort the keys?
    if sort_keys:
        sorted_idx = _dispatch.argsort_labels_values(
            tensor.keys.values, reverse=descending
        )
        new_keys = Labels(
            names=tensor.keys.names,
            values=tensor.keys.values[sorted_idx],
        )
    else:
        new_keys = tensor.keys
        sorted_idx = _dispatch.int_array_like(
            int_list=list(range(len(new_keys))),
            like=new_keys.values,
        )

    # Do any required sorting on the blocks
    new_blocks: List[TensorBlock] = []
    for i in sorted_idx:
        new_blocks.append(
            sort_block(
                block=tensor.block(tensor.keys[int(i)]),
                axes=axes_list,
                descending=descending,
            )
        )

    return TensorMap(new_keys, new_blocks)
