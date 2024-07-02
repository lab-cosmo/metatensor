import numpy as np

from . import _dispatch
from ._backend import Labels, TensorBlock, torch_jit_is_scripting, torch_jit_script


try:
    import torch

    TorchScriptClass = torch.ScriptClass
except ImportError:

    class TorchScriptClass:
        pass


@torch_jit_script
def block_from_array(array) -> TensorBlock:
    """
    Creates a simple TensorBlock from an array.

    The metadata in the resulting :py:class:`TensorBlock` is filled with ranges
    of integers. This function should be seen as a quick way of creating a
    :py:class:`TensorBlock` from arbitrary data. However, the metadata generated
    in this way has little meaning.

    :param array: An array with two or more dimensions. This can either be a
        :py:class:`numpy.ndarray` or a :py:class:`torch.Tensor`.

    :return: A :py:class:`TensorBlock` whose values correspond to the provided
        ``array``. The metadata names are set to ``"sample"`` for samples;
        ``"component_1"``, ``"component_2"``, ... for components; and
        ``property`` for properties. The number of ``component`` labels is
        adapted to the dimensionality of the input array. The metadata
        associated with each label is a range of integers going from 0 to the
        size of the corresponding axis. The returned :py:class:`TensorBlock` has
        no gradients.


    >>> import numpy as np
    >>> import metatensor
    >>> # Construct a simple 4D array:
    >>> array = np.linspace(0, 10, 42).reshape((7, 3, 1, 2))
    >>> # Transform it into a TensorBlock:
    >>> tensor_block = metatensor.block_from_array(array)
    >>> print(tensor_block)
    TensorBlock
        samples (7): ['sample']
        components (3, 1): ['component_1', 'component_2']
        properties (2): ['property']
        gradients: None
    >>> # The data inside the TensorBlock will correspond to the provided array:
    >>> print(np.all(array == tensor_block.values))
    True
    """

    shape = array.shape
    n_dimensions = len(shape)
    if n_dimensions < 2:
        raise ValueError(
            f"the array provided to `block_from_array` \
            must have at least two dimensions. Too few provided: {n_dimensions}"
        )

    if torch_jit_is_scripting():
        # we are using metatensor-torch
        labels_array_like = torch.empty(0)
    else:
        if isinstance(Labels, TorchScriptClass):
            # we are using metatensor-torch
            labels_array_like = torch.empty(0)
        else:
            # we are using metatensor-core
            labels_array_like = np.empty(0)

    samples = Labels(
        names=["sample"],
        values=_dispatch.int_array_like(
            list(range(shape[0])), labels_array_like
        ).reshape(-1, 1),
    )
    components = [
        Labels(
            names=[f"component_{component_index+1}"],
            values=_dispatch.int_array_like(
                list(range(axis_size)), labels_array_like
            ).reshape(-1, 1),
        )
        for component_index, axis_size in enumerate(shape[1:-1])
    ]
    properties = Labels(
        names=["property"],
        values=_dispatch.int_array_like(
            list(range(shape[-1])), labels_array_like
        ).reshape(-1, 1),
    )

    device = _dispatch.get_device(array)
    samples = samples.to(device)
    components = [component.to(device) for component in components]
    properties = properties.to(device)

    return TensorBlock(array, samples, components, properties)
