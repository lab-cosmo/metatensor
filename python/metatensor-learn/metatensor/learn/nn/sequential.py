from typing import List

import torch

from .._backend import Labels
from .module_map import ModuleMap


class Sequential(torch.nn.Module):
    """
    A sequential model that applies a list of ModuleMaps to the input in order.

    :param in_keys:
        The keys that are assumed to be in the input tensor map in the
        :py:meth:`forward` function.
    :param args:
        A list of :py:class:`ModuleMap` objects that will be applied in order to
        the input tensor map in the :py:meth:`forward` function.
    """

    def __init__(self, in_keys: Labels, *args: List[ModuleMap]):

        modules = []
        for i in range(len(in_keys)):
            module = torch.nn.Sequential(*[arg[i] for arg in args])
            modules.append(module)

        modules = torch.nn.ModuleList(modules)

        super().__init__(in_keys, modules, out_properties=args[-1].out_properties)
