from copy import deepcopy
from typing import List, Optional, Sequence, Union

import torch
from torch.nn import Module, ModuleList

from .._classes import Labels, LabelsEntry, TensorBlock, TensorMap


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


class ModuleMap(Module):
    """
    A class that imitates :py:class:`torch.nn.ModuleDict`. In its forward function the
    module at position `i` given on construction by :param modules: is applied to the
    tensor block that corresponding to the`i`th key in :param in_keys:.

    :param in_keys:
        A :py:class:`metatensor.Labels` object that determines the keys of the module
        map that are ass the TensorMaps that are assumed to be in the input tensor map
        in the :py:meth:`forward` function.

    :param modules:
        A sequence of modules applied in the :py:meth:`forward` function on the input
        :py:class:`TensorMap`. Each module corresponds to one :py:class:`LabelsEntry` in
        :param in_keys: that determines on which :py:class:`TensorBlock` the module is
        applied on.  :param modules: and :param in_keys: must match in length.

    :param out_tensor:
        A tensor map that is used to determine the properties labels of the output.
        Because an arbitrary module can change the number of properties, the labels of
        the properties cannot be persevered. By default the output properties are
        relabeled using Labels.range.


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
        ...         ["structure", "center"],
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
        ...         ["structure", "center"],
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

        Lets look at the metadata

        >>> tensor[0]
        TensorBlock
            samples (2): ['structure', 'center']
            components (): []
            properties (3): ['properties']
            gradients: None
        >>> out[0]
        TensorBlock
            samples (2): ['structure', 'center']
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
        modules: Sequence[Module],
        out_tensor: Optional[TensorMap] = None,
    ):
        super().__init__()
        if len(in_keys) != len(modules):
            raise ValueError(
                "in_keys and modules must match in length, but found "
                f"{len(in_keys) != len(modules)} [len(in_keys) != len(modules)]"
            )
        self._in_keys: Labels = in_keys
        self._modules_list = ModuleList(modules)
        # copy to prevent undefined behavior due to inplace changes
        if out_tensor is not None:
            out_tensor = out_tensor.copy()
        self._out_tensor = out_tensor
        # containts the orderd Labels object

    @classmethod
    def from_module(
        cls,
        in_keys: Labels,
        module: Module,
        many_to_one: bool = True,
        out_tensor: Optional[TensorMap] = None,
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
        :param many_to_one:
            Specifies if a separate module for each block is used. If `False` the module
            is deep copied for each key in the :py:attr:`in_keys`. otherwise the same
            module is used over all keys connecting the optimization of the weights.
        :param out_tensor:
            A tensor map that is used to determine the properties labels of the output.
            Because an arbitrary module can change the number of properties, the labels
            of the properties cannot be persevered. By default the output properties are
            relabeled using Labels.range.

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
        ...         ["structure", "center"],
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
        ...         ["structure", "center"],
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
            if many_to_one:
                modules.append(module)
            else:
                modules.append(deepcopy(module))

        return cls(in_keys, modules, out_tensor)

    def forward(self, tensor: TensorMap) -> TensorMap:
        """
        Takes a tensor map and applies the modules on each key it.

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
        module = self.get_module(key)

        out_values = module.forward(block.values)
        if self._out_tensor is None:
            properties = Labels.range("_", out_values.shape[-1])
        else:
            properties = self._out_tensor.block(key).properties
        return TensorBlock(
            values=out_values,
            properties=properties,
            components=block.components,
            samples=block.samples,
        )

    @property
    def modules(self):
        """
        :return modules:
            torch.nn.ModulesList that was given by initialization. It is not a copy, so
            its utility functionalities can be used to e.g. move the module map to GPU
        """
        # type annotation in function signature had to be removed because of TorchScript
        return self._modules_list

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
        module: ModuleMapInterface = self._modules_list[module_idx]
        return module

    @property
    def in_keys(self) -> Labels:
        """ """
        return self._in_keys

    @property
    def out_tensor(self) -> Optional[TensorMap]:
        """
        The tensor map that is used to determine properties labels of the output of
        forward function.
        """
        return self._out_tensor


class Linear(ModuleMap):
    """
    :param in_tensor:
        A tensor map that will be accepted in the :py:meth:`forward` function. It is
        used to determine the keys input shape, device and dtype of the input to create
        linear modules for tensor maps.

    :param out_tensor:
        A tensor map that is used to determine the properties labels and shape of the
        output tensor map.  Because a linear module can change the number of
        properties, the labels of the properties cannot be persevered.

    :param bias:
        See :py:class:`torch.nn.Linear` for bool as input. For each TensorMap key the
        bias can be also individually tuend by using a TensorMap with one value for the
        bool.
    """

    def __init__(
        self,
        in_tensor: TensorMap,
        out_tensor: TensorMap,
        bias: Union[bool, TensorMap] = True,
    ):
        if isinstance(bias, bool):
            blocks = [
                TensorBlock(
                    values=torch.tensor(bias).reshape(1, 1),
                    samples=Labels.range("_", 1),
                    components=[],
                    properties=Labels.range("_", 1),
                )
                for _ in in_tensor.keys
            ]
            bias = TensorMap(keys=in_tensor.keys, blocks=blocks)
        modules = []
        for key, in_block in in_tensor.items():
            out_block = out_tensor.block(key)
            module = torch.nn.Linear(
                len(in_block.properties),
                len(out_block.properties),
                bias.block(key).values.flatten()[0],
                in_block.values.device,
                in_block.values.dtype,
            )
            modules.append(module)

        super().__init__(in_tensor.keys, modules, out_tensor)

    @classmethod
    def from_module(
        cls,
        in_keys: Labels,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        many_to_one: bool = True,
        out_tensor: Optional[TensorMap] = None,
    ):
        """
        Simplified constructor if the same module should be applied on all blocks in the
        tensor map. It can be chosen if the separate module copies should act on the
        blocks or exactly the same module. This effects the type of computational graph
        created.

        :param in_keys:
            The keys that are assumed to be in the input tensor map in the
            :py:meth:`forward` function.
        :param in_features:
            See :py:class:`torch.nn.Linear`
        :param out_features:
            See :py:class:`torch.nn.Linear`
        :param bias:
            See :py:class:`torch.nn.Linear`
        :param device:
            See :py:class:`torch.nn.Linear`
        :param dtype:
            See :py:class:`torch.nn.Linear`
        :param many_to_one:
            Specifies if a separate module for each block is used. If `False` the module
            is deep copied for each key in the :py:attr:`in_keys`. otherwise the same
            module is used over all keys connecting the optimization of the weights.
        :param out_tensor:
            A tensor map that is used to determine the properties labels of the output.
            Because an arbitrary module can change the number of properties, the labels
            of the properties cannot be persevered. By default the output properties are
            relabeled using Labels.range.
        """
        module = torch.nn.Linear(in_features, out_features, bias, device, dtype)
        return ModuleMap.from_module(in_keys, module, many_to_one, out_tensor)

    @classmethod
    def from_weights(cls, weights: TensorMap, bias: Optional[TensorMap] = None):
        """
        Construct a linear module map from a tensor map.

        :param weights:
            The weight tensor map from which we create the linear modules.  The
            properties of the tensor map describe the input dimension and the samples
            describe the output dimension.

        :param bias:
            The weight tensor map from which we create the linear layers.
        """
        modules = []
        for key, weights_block in weights.items():
            module = torch.nn.Linear(
                len(weights_block.samples),
                len(weights_block.properties),
                bias=False,
                device=weights_block.values.device,
                dtype=weights_block.values.dtype,
            )
            module.weight = torch.nn.Parameter(weights_block.values.T)
            if bias is not None:
                module.bias = torch.nn.Parameter(bias.block(key).values)
            modules.append(module)

        return ModuleMap(weights.keys, modules, weights)

    def forward(self, tensor: TensorMap) -> TensorMap:
        # added to appear in doc, :inherited-members: is not compatible with torch
        return super().forward(tensor)
