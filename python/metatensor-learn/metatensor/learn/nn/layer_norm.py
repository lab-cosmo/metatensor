"""
Module containing classes :class:`LayerNorm` and :class:`EquiLayerNorm`, i.e.
module maps that apply layer norms in a generic and equivariant way,
respectively.
"""

from typing import List, Optional, Union

import torch

from .._backend import Labels
from .module_map import ModuleMap


class LayerNorm(ModuleMap):
    """
    Construct a module map with only layer norm modules.

    :param in_keys:
        The keys that are assumed to be in the input tensor map in the
        :py:meth:`forward` function.
    :param normalized_shape:
        Specifies the input shape from an expected input of size. If a single
        integer is used, this module will normalize over the last dimension
        which is expected to be of that specific size, for all blocks. If a
        torch.Size is passed, it is treated as the desired shape for all blocks.
        If a list of either of these is passed, each element is treated to each
        respective block, and it should have the same length as :param in_keys:.
    :param eps:
        A value added to the denominator for numerical stability. Default: 1e-5.
        If passed as a list, it is applied to each block and it should have the
        same length as :param in_keys:. If only one value is given, it is
        assumed that the same be applied on each block.
    :param elementwise_affine:
        A boolean value that when set to True, this module has learnable affine
        parameters. Default: True. If passed as a list, it is applied to each
        block and it should have the same length as :param in_keys:. If only one
        value is given, it is assumed that the same be applied on each block.
    :param bias:
        Specifies if a learnable bias term (offset) should be applied on the
        input tensor map in the forward function. If a list of bools is given,
        it specifies the bias term for each block, therefore it should have the
        same length as :param in_keys:.  If only one bool value is given, it is
        assumed that the same be applied on each block.
    :param device:
        Specifies the torch device of the values. If None the default torch
        device is taken. Note: these are only applied to invariant blocks to
        which the layer norm is applied.
    :param dtype:
        Specifies the torch dtype of the values. If None the default torch dtype
        is taken. Note: these are only applied to invariant blocks to
        which the layer norm is applied.
    :param out_properties:
        A list of labels that is used to determine the properties labels of the
        output.  Because a module could change the number of properties, the
        labels of the properties cannot be persevered. By default the output
        properties are relabeled using Labels.range.
    """

    def __init__(
        self,
        in_keys: Labels,
        normalized_shape: Union[
            Union[int, List[int], torch.Size],
            Union[List[int], List[List[int]], List[torch.Size]],
        ],
        eps: Union[float, List[float]] = 1e-5,
        elementwise_affine: Union[bool, List[bool]] = True,
        *,
        bias: Union[bool, List[bool]] = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        out_properties: Optional[List[Labels]] = None,
    ):
        # Check `normalized_shape`
        if isinstance(normalized_shape, int) or isinstance(
            normalized_shape, torch.Size
        ):
            normalized_shape = [normalized_shape] * len(in_keys)
        elif not (isinstance(normalized_shape, List)):
            raise TypeError(
                "`normalized_shape` must be integer or List of integers,"
                f" but not {type(normalized_shape)}."
            )
        elif len(in_keys) != len(normalized_shape):
            raise ValueError(
                "`normalized_shape` must have same length as `in_keys`, but"
                f" len(normalized_shape) != len(in_keys)"
                f" [{len(normalized_shape)} != {len(in_keys)}]"
            )

        for n_shape in normalized_shape:
            if not (
                isinstance(n_shape, int)
                or isinstance(n_shape, List)
                or isinstance(n_shape, torch.Size)
            ):
                raise TypeError(
                    "`normalized_shape` must be integer or List of"
                    f" integers, but not {type(n_shape)}."
                )
        # Check `eps`
        if isinstance(eps, float):
            eps = [eps] * len(in_keys)
        elif not (isinstance(eps, List)):
            raise TypeError(
                "`eps` must be float or List of float, but not"
                f" {type(normalized_shape)}."
            )
        elif len(in_keys) != len(eps):
            raise ValueError(
                "`eps` must have same length as `in_keys`, but"
                f" len(eps) != len(in_keys) [{len(eps)} !="
                f" {len(in_keys)}]"
            )
        # Check `elementwise_affine`
        if isinstance(elementwise_affine, bool):
            elementwise_affine = [elementwise_affine] * len(in_keys)
        elif not (isinstance(elementwise_affine, List)):
            raise TypeError(
                "`elementwise_affine` must be float or List of float,"
                f" but not {type(normalized_shape)}."
            )
        elif len(in_keys) != len(elementwise_affine):
            raise ValueError(
                "`elementwise_affine` must have same length as `in_keys`,"
                f" but len(elementwise_affine) != len(in_keys)"
                f" [{len(elementwise_affine)} != {len(in_keys)}]"
            )
        # Check `bias`
        if isinstance(bias, bool):
            bias = [bias] * len(in_keys)
        elif not (isinstance(bias, List)):
            raise TypeError(
                "`bias` must be float or List of float, but not"
                f" {type(normalized_shape)}."
            )
        elif len(in_keys) != len(bias):
            raise ValueError(
                "`bias` must have same length as `in_keys`, but"
                f" len(bias) != len(in_keys) [{len(bias)} !="
                f" {len(in_keys)}]"
            )

        # Build module list
        modules = []
        for i in range(len(in_keys)):
            module = torch.nn.LayerNorm(
                normalized_shape=normalized_shape[i],
                eps=eps[i],
                elementwise_affine=elementwise_affine[i],
                bias=bias[i],
                device=device,
                dtype=dtype,
            )
            modules.append(module)

        super().__init__(in_keys, modules, out_properties)


class EquiLayerNorm(ModuleMap):
    """
    Construct a module map with only layer norm modules applied to the invariant
    blocks, and the identity applied to covariant blocks.

    :param in_keys:
        The keys that are assumed to be in the input tensor map in the
        :py:meth:`forward` function.
    :param invariant_key_idxs:
        The indices of the invariant keys present in `in_keys` in the input
        tensor map. Only blocks for these keys will have layer norm applied. The
        other blocks will have the identity operator applied.
    :param normalized_shape:
        Specifies the input shape from an expected input of size. If a single
        integer is used, this module will normalize over the last dimension
        which is expected to be of that specific size, for all blocks. If a
        torch.Size is passed, it is treated as the desired shape for all blocks.
        If a list of either of these is passed, each element is treated to each
        respective block, and it should have the same length as :param
        invariant_key_idxs:.
    :param eps:
        A value added to the denominator for numerical stability. Default: 1e-5.
        If passed as a list, it is applied to each block and it should have the
        same length as :param invariant_key_idxs:. If only one value is given,
        it is assumed that the same be applied on each block.
    :param elementwise_affine:
        A boolean value that when set to True, this module has learnable affine
        parameters. Default: True. If passed as a list, it is applied to each
        block and it should have the same length as :param invariant_key_idxs:.
        If only one value is given, it is assumed that the same be applied on
        each block.
    :param bias:
        Specifies if a learnable bias term (offset) should be applied on the
        input tensor map in the forward function. If a list of bools is given,
        it specifies the bias term for each block, therefore it should have the
        same length as :param invariant_key_idxs:.  If only one bool value is
        given, it is assumed that the same be applied on each block.
    :param device:
        Specifies the torch device of the values. If None the default torch
        device is taken. Note: these are only applied to invariant blocks to
        which the layer norm is applied.
    :param dtype:
        Specifies the torch dtype of the values. If None the default torch dtype
        is taken. Note: these are only applied to invariant blocks to which the
        layer norm is applied.
    :param out_properties:
        A list of labels that is used to determine the properties labels of the
        output.  Because a module could change the number of properties, the
        labels of the properties cannot be persevered. By default the output
        properties are relabeled using Labels.range.
    """

    def __init__(
        self,
        in_keys: Labels,
        invariant_key_idxs: List[int],
        normalized_shape: Union[
            Union[int, List[int], torch.Size],
            Union[List[int], List[List[int]], List[torch.Size]],
        ],
        eps: Union[float, List[float]] = 1e-5,
        elementwise_affine: Union[bool, List[bool]] = True,
        *,
        bias: Union[bool, List[bool]] = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        out_properties: Optional[List[Labels]] = None,
    ):
        # Check `normalized_shape`
        if isinstance(normalized_shape, int) or isinstance(
            normalized_shape, torch.Size
        ):
            normalized_shape = [normalized_shape] * len(invariant_key_idxs)
        elif not (isinstance(normalized_shape, List)):
            raise TypeError(
                "`normalized_shape` must be integer or List of integers, but not"
                f" {type(normalized_shape)}."
            )
        elif len(invariant_key_idxs) != len(normalized_shape):
            raise ValueError(
                "`normalized_shape` must have same length as `invariant_key_idxs`,"
                f" but len(normalized_shape) != len(invariant_key_idxs)"
                f" [{len(normalized_shape)} != {len(invariant_key_idxs)}]"
            )
        for n_shape in normalized_shape:
            if not (
                isinstance(n_shape, int)
                or isinstance(n_shape, List)
                or isinstance(n_shape, torch.Size)
            ):
                raise TypeError(
                    "`normalized_shape` must be integer or List of integers, but not"
                    f" {type(n_shape)}."
                )
        # Check `eps`
        if isinstance(eps, float):
            eps = [eps] * len(invariant_key_idxs)
        elif not (isinstance(eps, List)):
            raise TypeError(
                "`eps` must be float or List of float, but not"
                f" {type(normalized_shape)}."
            )
        elif len(invariant_key_idxs) != len(eps):
            raise ValueError(
                "`eps` must have same length as `invariant_key_idxs`, but"
                f" len(eps) != len(invariant_key_idxs) [{len(eps)} !="
                f" {len(invariant_key_idxs)}]"
            )
        # Check `elementwise_affine`
        if isinstance(elementwise_affine, bool):
            elementwise_affine = [elementwise_affine] * len(invariant_key_idxs)
        elif not (isinstance(elementwise_affine, List)):
            raise TypeError(
                "`elementwise_affine` must be float or List of float, but not"
                f" {type(normalized_shape)}."
            )
        elif len(invariant_key_idxs) != len(elementwise_affine):
            raise ValueError(
                "`elementwise_affine` must have same length as `invariant_key_idxs`,"
                f" but len(elementwise_affine) != len(invariant_key_idxs)"
                f" [{len(elementwise_affine)} != {len(invariant_key_idxs)}]"
            )
        # Check `bias`
        if isinstance(bias, bool):
            bias = [bias] * len(invariant_key_idxs)
        elif not (isinstance(bias, List)):
            raise TypeError(
                "`bias` must be float or List of float, but not"
                f" {type(normalized_shape)}."
            )
        elif len(invariant_key_idxs) != len(bias):
            raise ValueError(
                "`bias` must have same length as `invariant_key_idxs`, but"
                f" len(bias) != len(invariant_key_idxs) [{len(bias)} !="
                f" {len(invariant_key_idxs)}]"
            )

        # Build module list
        modules = []
        invariant_idx = 0
        for i in range(len(in_keys)):

            # Invariant block: apply LayerNorm
            if i in invariant_key_idxs:
                for j in range(len(invariant_key_idxs)):
                    if invariant_key_idxs[j] == i:
                        invariant_idx = j

                module = torch.nn.LayerNorm(
                    normalized_shape=normalized_shape[invariant_idx],
                    eps=eps[invariant_idx],
                    elementwise_affine=elementwise_affine[invariant_idx],
                    bias=bias[invariant_idx],
                    device=device,
                    dtype=dtype,
                )

            # Covariant block: apply Identity
            else:
                module = torch.nn.Identity()

            modules.append(module)

        super().__init__(in_keys, modules, out_properties)
