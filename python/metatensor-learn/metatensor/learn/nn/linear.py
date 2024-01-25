from typing import List, Optional, Union

import torch

from .module_map import ModuleMap
from .._classes import Labels, LabelsEntry, TensorBlock, TensorMap

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
