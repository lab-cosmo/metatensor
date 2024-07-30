from typing import List

from . import _dispatch
from ._backend import (
    Labels,
    TensorBlock,
    TensorMap,
    check_isinstance,
    torch_jit_is_scripting,
    torch_jit_script,
)


@torch_jit_script
def drop_blocks(tensor: TensorMap, keys: Labels, copy: bool = False) -> TensorMap:
    """
    Drop specified key/block pairs from a TensorMap.

    :param tensor: the :py:class:`TensorMap` to drop the key-block pair from.

    :param keys: selection of keys to remove from the input ``tensor``. This can
        contain any subset of the names of the tensor's keys.

    :param copy: if :py:obj:`True`, the returned :py:class:`TensorMap` is constructed by
        copying the blocks from the input `tensor`. If :py:obj:`False` (default), the
        values of the blocks in the output :py:class:`TensorMap` reference the same data
        as the input `tensor`. The latter can be useful for limiting memory usage, but
        should be used with caution when manipulating the underlying data.
    """
    # Check arg types
    if not torch_jit_is_scripting():
        if not check_isinstance(tensor, TensorMap):
            raise TypeError(
                f"`tensor` must be a metatensor TensorMap, not {type(tensor)}"
            )

        if not check_isinstance(keys, Labels):
            raise TypeError(f"`keys` must be a metatensor Labels, not {type(keys)}")

        if not isinstance(copy, bool):
            raise TypeError(f"`copy` must be a boolean, not {type(copy)}")

    # Find the indices of keys to remove
    tensor_keys = tensor.keys
    to_remove_indices: List[int] = tensor_keys.select(keys).tolist()

    # Create the new TensorMap
    new_blocks: List[TensorBlock] = []
    new_keys_values = []
    for i in range(len(tensor_keys)):
        if i in to_remove_indices:
            continue

        new_keys_values.append(tensor_keys.entry(i).values)
        block = tensor[i]

        if copy:
            new_blocks.append(block.copy())
        else:
            # just increase the reference count on everything
            new_block = TensorBlock(
                values=block.values,
                samples=block.samples,
                components=block.components,
                properties=block.properties,
            )

            for parameter, gradient in block.gradients():
                if len(gradient.gradients_list()) != 0:
                    raise NotImplementedError(
                        "gradients of gradients are not supported"
                    )

                new_block.add_gradient(
                    parameter=parameter,
                    gradient=TensorBlock(
                        values=gradient.values,
                        samples=gradient.samples,
                        components=gradient.components,
                        properties=new_block.properties,
                    ),
                )

            new_blocks.append(new_block)

    if len(new_keys_values) != 0:
        new_keys = Labels(tensor_keys.names, _dispatch.stack(new_keys_values, 0))
    else:
        new_keys = Labels(
            names=tensor_keys.names,
            values=_dispatch.empty_like(
                tensor_keys.values, (0, len(tensor_keys.names))
            ),
        )

    return TensorMap(keys=new_keys, blocks=new_blocks)
