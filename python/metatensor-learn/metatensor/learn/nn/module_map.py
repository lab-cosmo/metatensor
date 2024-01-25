from copy import deepcopy
from typing import List, Optional, Union

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


class ModuleMap(ModuleList):
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

    :param out_properties:
        A list of labels that is used to determine the properties labels of the
        output.  Because a module could change the number of properties, the labels of
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
        modules: List[Module],
        out_properties: List[Labels] = None,
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
        out_properties: List[Labels] = None,
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
            properties are relabeled using Labels.range.

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
            modules.append(deepcopy(module))

        return cls(in_keys, modules, out_properties)

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
        position = self._in_keys.position(key)
        if position is None:
            raise KeyError(f"key {key} not found in modules.")
        module_idx: int = position
        module: ModuleMapInterface = self[module_idx]

        out_values = module.forward(block.values)
        if self._out_properties is None:
            properties = Labels.range("_", out_values.shape[-1])
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


class Linear(ModuleMap):
    """
    Construct a module map with only linear modules.

    :param in_keys:
        The keys that are assumed to be in the input tensor map in the
        :py:meth:`forward` function.
    :param in_features:
        Specifies the dimensionality of the input after it has been applied on the input
        tensor map. If a list of integers is given, it specifies the input dimension for
        each block, therefore it should have the same length as :param in_keys:.  If
        only one integer is given, it is assumed that the same be applied on each block.
    :param out_features:
        Specifies the dimensionality of the output after it has been applied on the
        input tensor map. If a list of integers is given, it specifies the output
        dimension for each block, therefore it should have the same length as :param
        in_keys:.  If only one integer is given, it is assumed that the same be applied
        on each block.
    :param bias:
        Specifies if a bias term (offset) should be applied on the input tensor map in
        the forward function. If a list of bools is given, it specifies the bias term
        for each block, therefore it should have the same length as :param in_keys:.  If
        only one bool value is given, it is assumed that the same be applied on each
        block.
    :param device:
        Specifies the torch device of the values. If None the default torch device is
        taken.
    :param dtype:
        Specifies the torch dtype of the values. If None the default torch dtype is
        taken.
    :param out_properties:
        A list of labels that is used to determine the properties labels of the
        output.  Because a module could change the number of properties, the labels of
        the properties cannot be persevered. By default the output properties are
        relabeled using Labels.range.
    """

    def __init__(
        self,
        in_keys: Labels,
        in_features: Union[int, List[int]],
        out_features: Union[int, List[int]],
        bias: Union[bool, List[bool]] = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
        out_properties: List[Labels] = None,
    ):
        if isinstance(in_features, int):
            in_features = [in_features] * len(in_keys)
        elif not (isinstance(in_features, List)):
            raise TypeError(
                "`in_features` must be integer or List of integers, but not"
                f" {type(in_features)}."
            )
        elif len(in_keys) != len(in_features):
            raise ValueError(
                "`in_features` must have same length as `in_keys`, but"
                f" len(in_features) != len(in_keys) [{len(in_features)} !="
                f" {len(in_keys)}]"
            )

        if isinstance(out_features, int):
            out_features = [out_features] * len(in_keys)
        elif not (isinstance(out_features, List)):
            raise TypeError(
                "`out_features` must be integer or List of integers, but not"
                f" {type(out_features)}."
            )
        elif len(in_keys) != len(out_features):
            raise ValueError(
                "`out_features` must have same length as `in_keys`, but"
                f" len(out_features) != len(in_keys) [{len(out_features)} !="
                f" {len(in_keys)}]"
            )

        if isinstance(bias, bool):
            bias = [bias] * len(in_keys)
        elif not (isinstance(bias, List)):
            raise TypeError(
                f"`bias` must be bool or List of bools, but not {type(bias)}."
            )
        elif len(in_keys) != len(bias):
            raise ValueError(
                "`bias` must have same length as `in_keys`, but len(bias) !="
                f" len(in_keys) [{len(bias)} != {len(in_keys)}]"
            )

        modules = []
        for i in range(len(in_keys)):
            module = torch.nn.Linear(
                in_features[i],
                out_features[i],
                bias[i],
                device,
                dtype,
            )
            modules.append(module)

        super().__init__(in_keys, modules, out_properties)

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
