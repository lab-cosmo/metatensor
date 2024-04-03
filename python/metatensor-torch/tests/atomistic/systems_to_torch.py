try:
    import ase

    HAS_ASE = True
except ImportError:
    HAS_ASE = False

import numpy as np
import pytest
import torch

from metatensor.torch.atomistic import systems_to_torch


@pytest.mark.skipif(not HAS_ASE, reason="requires ASE")
def test_ase_to_torch_periodic():
    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=4 * np.eye(3),
        pbc=[True, True, True],
    )

    system = systems_to_torch(atoms)

    assert isinstance(system, torch.ScriptObject)

    assert isinstance(system.types, torch.Tensor)
    assert torch.all(system.types == torch.tensor([6, 8]))
    assert system.types.dtype == torch.int32
    assert not system.types.requires_grad

    assert isinstance(system.positions, torch.Tensor)
    assert torch.all(system.positions == torch.tensor([(0, 0, 0), (0, 0, 2)]))
    assert system.positions.dtype == torch.get_default_dtype()
    assert not system.positions.requires_grad

    assert isinstance(system.cell, torch.Tensor)
    assert torch.all(system.cell == 4 * torch.eye(3))
    assert system.cell.dtype == torch.get_default_dtype()
    assert not system.cell.requires_grad

    system = systems_to_torch(atoms, positions_requires_grad=True)

    assert system.positions.requires_grad
    assert not system.cell.requires_grad

    # test a list of ase.Atoms
    systems = systems_to_torch([atoms, atoms])
    assert isinstance(systems[0], torch.ScriptObject)


@pytest.mark.skipif(not HAS_ASE, reason="requires ASE")
def test_ase_to_torch_non_periodic():

    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=None,
        pbc=[False, False, False],
    )

    system = systems_to_torch(atoms)

    assert isinstance(system, torch.ScriptObject)

    assert isinstance(system.types, torch.Tensor)
    assert torch.all(system.types == torch.tensor([6, 8]))
    assert system.types.dtype == torch.int32
    assert not system.types.requires_grad

    assert isinstance(system.positions, torch.Tensor)
    assert torch.all(system.positions == torch.tensor([(0, 0, 0), (0, 0, 2)]))
    assert system.positions.dtype == torch.get_default_dtype()
    assert not system.positions.requires_grad

    assert torch.all(system.cell == 0)

    # test a list of ase.Atoms
    systems = systems_to_torch([atoms, atoms])
    assert isinstance(systems[0], torch.ScriptObject)

    # test the warning when converting a non-periodic
    # system with non-zero cell vectors

    atoms = ase.Atoms(
        "CO",
        positions=[(0, 0, 0), (0, 0, 2)],
        cell=4 * np.eye(3),
        pbc=[False, False, False],
    )

    with pytest.warns(UserWarning, match="non-zero cell vectors"):
        system = systems_to_torch(atoms)

    assert torch.all(system.cell == 0)
