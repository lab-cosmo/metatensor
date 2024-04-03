import warnings
from typing import List, Optional, Union

import numpy as np
import torch

from . import System


try:
    import ase

    HAS_ASE = True
except ImportError:
    HAS_ASE = False


class IntoSystem:
    """A type that can be converted into a
    :py:class:`metatensor.torch.atomistic.System`.

    This is an abstract class that is used to indicate a class whose objects
    can be converted into a :py:class:`System`. For the moment,
    the only supported type is :py:class:`ase.Atoms`."""

    pass


def systems_to_torch(
    systems: Union[IntoSystem, List[IntoSystem]],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    positions_requires_grad: bool = False,
    cell_requires_grad: bool = False,
) -> Union[System, List[System]]:
    """Converts a system or a list of systems into a
    ``metatensor.torch.atomistic.System`` or a list of such objects.

    :param: systems: The system or list of systems to convert.
    :param: dtype: The dtype of the output tensors. If ``None``, the default
        dtype is used.
    :param: device: The device of the output tensors. If ``None``, the default
        device is used.
    :param: positions_requires_grad: Whether the positions tensors of
        the outputs should require gradients.
    :param: cell_requires_grad: Whether the cell tensors of the outputs
        should require gradients.

    :return: The converted system or list of systems.
    """

    if isinstance(systems, list):
        return [
            _system_to_torch(
                system, dtype, device, positions_requires_grad, cell_requires_grad
            )
            for system in systems
        ]
    else:
        return _system_to_torch(
            systems, dtype, device, positions_requires_grad, cell_requires_grad
        )


def _system_to_torch(
    system: IntoSystem,
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
    positions_requires_grad: bool,
    cell_requires_grad: bool,
) -> System:
    """Converts a system into a ``metatensor.torch.atomistic.System``.

    :param: system: The system to convert.
    :param: dtype: The dtype of the output tensors. If ``None``, the default
        dtype is used.
    :param: device: The device of the output tensors. If ``None``, the default
        device is used.
    :param: positions_requires_grad: Whether the positions tensors of
        the outputs should require gradients.
    :param: cell_requires_grad: Whether the cell tensors of the outputs
        should require gradients.

    :return: The converted system.
    """
    if not HAS_ASE:
        raise RuntimeError("The `ase` package is required to convert systems to torch.")

    if not isinstance(system, ase.Atoms):
        raise ValueError(
            "Only `ase.Atoms` objects can be converted to `System`s "
            f"for now; got {type(system)}."
        )

    if dtype is None:
        # this is necessary because creating torch tensors from numpy arrays
        # takes the dtype from the numpy array, which is not always the default
        # dtype
        dtype = torch.get_default_dtype()

    positions = torch.tensor(
        system.positions,
        requires_grad=positions_requires_grad,
        dtype=dtype,
        device=device,
    )

    if np.all(system.pbc):
        cell = torch.tensor(
            system.cell[:],
            requires_grad=cell_requires_grad,
            dtype=dtype,
            device=device,
        )
    elif not np.any(system.pbc):
        if system.cell is not None and not np.all(system.cell == 0):
            warnings.warn(
                "A conversion to `System` was requested for an `ase.Atoms` object "
                "with non-zero cell vectors but no periodic boundary conditions. "
                "The cell vectors will be set to zero.",
                stacklevel=2,
            )
        cell = torch.zeros(
            (3, 3), requires_grad=cell_requires_grad, dtype=dtype, device=device
        )
    else:
        raise ValueError(
            "The `metatensor.torch.atomistic.System` class does not support "
            "systems with mixed periodic and non-periodic dimensions."
        )

    types = torch.tensor(system.numbers, device=device, dtype=torch.int32)

    return System(positions=positions, cell=cell, types=types)
