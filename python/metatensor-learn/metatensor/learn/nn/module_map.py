from copy import deepcopy
from typing import List, Optional, Union

import torch
from torch.nn import Module, ModuleList

from metatensor.operations import _dispatch

from .._backend import Labels, LabelsEntry, TensorBlock, TensorMap


@torch.jit.interface
class ModuleMapInterface(torch.nn.Module):
    """
    This interface required for TorchScript to index the :py:class:`torch.nn.ModuleDict`
    with non-literals in ModuleMap. Any module that is used with ModuleMap must
    implement this interface to be TorchScript compilable.

    Note that the *typings and argument names must match exactly* so that an interface
    is correctly implemented.

    Reference
    ---------
    https://github.com/pytorch/pytorch/pull/45716
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass


class ModuleMap(ModuleList):
    """
    A class that imitates :py:class:`torch.nn.ModuleDict`. In its forward function the
    module at position `i` given on construction by :param modules: is applied to the
    tensor block that corresponding to the`i`th key in :param in_keys:.

    :param in_keys:
        A :py:class:`metatensor.Labels` object with the keys of the module map that are
        assumed to be in the input tensor map in the :py:meth:`forward` function.

    :param modules:
        A sequence of modules applied in the :py:meth:`forward` function on the input
        :py:class:`TensorMap`. Each module corresponds to one :py:class:`LabelsEntry` in
        :param in_keys: that determines on which :py:class:`TensorBlock` the module is
        applied on.  :param modules: and :param in_keys: must match in length.

    :param out_properties:
        A list of labels that is used to determine the properties labels of the
        output.  Because a module could change the number of properties, the labels of
        the properties cannot be persevered. By default the output properties are
        relabeled using Labels.range with "_" as key.


        >>> import torch
        >>> import numpy as np
        >>> from copy import deepcopy
        >>> from metatensor import Labels, TensorBlock, TensorMap
        >>> from metatensor.learn.nn import ModuleMap

        Create simple block

        >>> block_1 = TensorBlock(
        ...     values=torch.tensor(
        ...         [
        ...             [1.0, 2.0, 4.0],
        ...             [3.0, 5.0, 6.0],
        ...         ]
        ...     ),
        ...     samples=Labels(
        ...         ["system", "atom"],
        ...         np.array(
        ...             [
        ...                 [0, 0],
        ...                 [0, 1],
        ...             ]
        ...         ),
        ...     ),
        ...     components=[],
        ...     properties=Labels(["properties"], np.array([[0], [1], [2]])),
        ... )
        >>> block_2 = TensorBlock(
        ...     values=torch.tensor(
        ...         [
        ...             [5.0, 8.0, 2.0],
        ...             [1.0, 2.0, 8.0],
        ...         ]
        ...     ),
        ...     samples=Labels(
        ...         ["system", "atom"],
        ...         np.array(
        ...             [
        ...                 [0, 0],
        ...                 [0, 1],
        ...             ]
        ...         ),
        ...     ),
        ...     components=[],
        ...     properties=Labels(["properties"], np.array([[3], [4], [5]])),
        ... )
        >>> keys = Labels(names=["key"], values=np.array([[0], [1]]))
        >>> tensor = TensorMap(keys, [block_1, block_2])

        Create modules

        >>> linear = torch.nn.Linear(3, 1, bias=False)
        >>> with torch.no_grad():
        ...     _ = linear.weight.copy_(torch.tensor([1.0, 1.0, 1.0]))
        ...
        >>> modules = [linear, deepcopy(linear)]
        >>> # you could also extend the module by some nonlinear activation function

        Create ModuleMap from this ModucDict and apply it

        >>> module_map = ModuleMap(tensor.keys, modules)
        >>> out = module_map(tensor)
        >>> out
        TensorMap with 2 blocks
        keys: key
               0
               1
        >>> out[0].values
        tensor([[ 7.],
                [14.]], grad_fn=<MmBackward0>)
        >>> out[1].values
        tensor([[15.],
                [11.]], grad_fn=<MmBackward0>)

        Let's look at the metadata

        >>> tensor[0]
        TensorBlock
            samples (2): ['system', 'atom']
            components (): []
            properties (3): ['properties']
            gradients: None
        >>> out[0]
        TensorBlock
            samples (2): ['system', 'atom']
            components (): []
            properties (1): ['_']
            gradients: None

        It got completely lost because we cannot know in general what the output is.
        You can add in the initialization of the ModuleMap a TensorMap that contains
        the intended output Labels.
    """

    def __init__(
        self,
        in_keys: Labels,
        modules: List[Module],
        out_properties: Optional[List[Labels]] = None,
    ):
        super().__init__(modules)
        if len(in_keys) != len(modules):
            raise ValueError(
                "in_keys and modules must match in length, but found "
                f"{len(in_keys) != len(modules)} [len(in_keys) != len(modules)]"
            )
        self._in_keys: Labels = in_keys
        self._out_properties = out_properties

    @classmethod
    def from_module(
        cls,
        in_keys: Labels,
        module: Module,
        out_properties: Optional[List[Labels]] = None,
    ):
        """
        A wrapper around one :py:class:`torch.nn.Module` applying the same type of
        module on each tensor block.

        :param in_keys:
            A :py:class:`metatensor.Labels` object that determines the keys of the
            module map that are ass the TensorMaps that are assumed to be in the input
            tensor map in the :py:meth:`forward` function.
        :param module:
            The module that is applied on each block.
        :param out_properties:
            A list of labels that is used to determine the properties labels of
            the output.  Because a module could change the number of properties, the
            labels of the properties cannot be persevered. By default the output
            properties are relabeled using Labels.range with "_" as key.

        >>> import torch
        >>> import numpy as np
        >>> from metatensor import Labels, TensorBlock, TensorMap
        >>> block_1 = TensorBlock(
        ...     values=torch.tensor(
        ...         [
        ...             [1.0, 2.0, 4.0],
        ...             [3.0, 5.0, 6.0],
        ...         ]
        ...     ),
        ...     samples=Labels(
        ...         ["system", "atom"],
        ...         np.array(
        ...             [
        ...                 [0, 0],
        ...                 [0, 1],
        ...             ]
        ...         ),
        ...     ),
        ...     components=[],
        ...     properties=Labels(["properties"], np.array([[0], [1], [2]])),
        ... )
        >>> block_2 = TensorBlock(
        ...     values=torch.tensor(
        ...         [
        ...             [5.0, 8.0, 2.0],
        ...             [1.0, 2.0, 8.0],
        ...         ]
        ...     ),
        ...     samples=Labels(
        ...         ["system", "atom"],
        ...         np.array(
        ...             [
        ...                 [0, 0],
        ...                 [0, 1],
        ...             ]
        ...         ),
        ...     ),
        ...     components=[],
        ...     properties=Labels(["properties"], np.array([[0], [1], [2]])),
        ... )
        >>> keys = Labels(names=["key"], values=np.array([[0], [1]]))
        >>> tensor = TensorMap(keys, [block_1, block_2])
        >>> linear = torch.nn.Linear(3, 1, bias=False)
        >>> with torch.no_grad():
        ...     _ = linear.weight.copy_(torch.tensor([1.0, 1.0, 1.0]))
        ...
        >>> # you could also extend the module by some nonlinear activation function
        >>> from metatensor.learn.nn import ModuleMap
        >>> module_map = ModuleMap.from_module(tensor.keys, linear)
        >>> out = module_map(tensor)
        >>> out[0].values
        tensor([[ 7.],
                [14.]], grad_fn=<MmBackward0>)
        >>> out[1].values
        tensor([[15.],
                [11.]], grad_fn=<MmBackward0>)
        """
        module = deepcopy(module)
        modules = []
        for _ in range(len(in_keys)):
            modules.append(deepcopy(module))

        return cls(in_keys, modules, out_properties)

    def forward(self, tensor: TensorMap) -> TensorMap:
        """
        Apply the modules on each block in ``tensor``. ``tensor`` must have the same
        set of keys as the modules used to initialize this :py:class:`ModuleMap`.

        :param tensor:
            input tensor map
        """
        out_blocks: List[TensorBlock] = []
        for key, block in tensor.items():
            out_block = self.forward_block(key, block)

            for parameter, gradient in block.gradients():
                if len(gradient.gradients_list()) != 0:
                    raise NotImplementedError(
                        "gradients of gradients are not supported"
                    )
                out_block.add_gradient(
                    parameter=parameter,
                    gradient=self.forward_block(key, gradient),
                )
            out_blocks.append(out_block)

        return TensorMap(tensor.keys, out_blocks)

    def forward_block(self, key: LabelsEntry, block: TensorBlock) -> TensorBlock:
        position = self._in_keys.position(key)
        if position is None:
            raise KeyError(f"key {key} not found in modules.")
        module_idx: int = position
        module: ModuleMapInterface = self[module_idx]

        out_values = module.forward(block.values)
        if self._out_properties is None:
            # we do not use range because of metatensor/issues/410
            properties = Labels(
                "_",
                _dispatch.int_array_like(
                    list(range(out_values.shape[-1])), block.samples.values
                ).reshape(-1, 1),
            )
        else:
            properties = self._out_properties[module_idx]
        return TensorBlock(
            values=out_values,
            properties=properties,
            components=block.components,
            samples=block.samples,
        )

    @torch.jit.export
    def get_module(self, key: LabelsEntry):
        """
        :param key:
            key of module which should be returned

        :return module:
            returns he torch.nn.Module corresponding to the :param key:
        """
        # type annotation in function signature had to be removed because of TorchScript
        position = self._in_keys.position(key)
        if position is None:
            raise KeyError(f"key {key} not found in modules.")
        module_idx: int = position
        module: ModuleMapInterface = self[module_idx]
        return module

    @property
    def in_keys(self) -> Labels:
        """
        A list of labels that defines the initialized keys with corresponding modules
        of this module map.
        """
        return self._in_keys

    @property
    def out_properties(self) -> Union[None, List[Labels]]:
        """
        A list of labels that is used to determine properties labels of the
        output of forward function.
        """
        return self._out_properties

    def repr_as_module_dict(self) -> str:
        """
        Returns a string that is easier to read that the standard __repr__ showing the
        mapping from label entry key to module.
        """
        representation = "ModuleMap(\n"
        for i, key in enumerate(self._in_keys):
            representation += f"  ({key!r}): {self[i]!r}\n"
        representation += ")"
        return representation
