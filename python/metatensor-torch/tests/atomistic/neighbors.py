import pytest
import torch
from packaging import version

from metatensor.torch.atomistic import (
    NeighborsListOptions,
    System,
    register_autograd_neighbors,
)


try:
    import ase

    from metatensor.torch.atomistic.ase_calculator import _compute_ase_neighbors

    HAVE_ASE = True
except ImportError:
    HAVE_ASE = False


def test_neighbors_lists_options():
    options = NeighborsListOptions(3.4, True, "hello")

    assert options.cutoff == 3.4
    assert options.full_list
    assert options.requestors() == ["hello"]

    options.add_requestor("another one")
    assert options.requestors() == ["hello", "another one"]

    # No empty requestors, no duplicated requestors
    options.add_requestor("")
    options.add_requestor("hello")
    assert options.requestors() == ["hello", "another one"]

    assert NeighborsListOptions(3.4, True, "a") == NeighborsListOptions(3.4, True, "b")
    assert NeighborsListOptions(3.4, True) != NeighborsListOptions(3.4, False)
    assert NeighborsListOptions(3.4, True) != NeighborsListOptions(3.5, True)

    expected = "NeighborsListOptions(cutoff=3.400000, full_list=True)"
    assert str(options) == expected

    expected = """NeighborsListOptions
    cutoff: 3.400000
    full_list: True
    requested by:
        - hello
        - another one
"""
    if version.parse(torch.__version__) >= version.parse("2.1"):
        assert repr(options) == expected


@pytest.mark.skipif(not HAVE_ASE, reason="this tests requires ASE neighbors list")
def test_neighbors_autograd():
    torch.manual_seed(0xDEADBEEF)
    n_atoms = 20
    cell_size = 6.0
    positions = cell_size * torch.rand(
        n_atoms, 3, dtype=torch.float64, requires_grad=True
    )
    cell = cell_size * torch.eye(3, dtype=torch.float64, requires_grad=True)

    def compute(positions, cell, options):
        atoms = ase.Atoms(
            "C" * positions.shape[0],
            positions=positions.detach().numpy(),
            cell=cell.detach().numpy(),
            pbc=True,
        )
        neighbors = _compute_ase_neighbors(
            atoms, options, dtype=torch.float64, device="cpu"
        )

        system = System(
            torch.from_numpy(atoms.numbers).to(torch.int32), positions, cell
        )
        register_autograd_neighbors(system, neighbors)

        return neighbors.values

    options = NeighborsListOptions(cutoff=2.0, full_list=False)
    torch.autograd.gradcheck(
        compute,
        (positions, cell, options),
        fast_mode=True,
    )

    options = NeighborsListOptions(cutoff=2.0, full_list=True)
    torch.autograd.gradcheck(
        compute,
        (positions, cell, options),
        fast_mode=True,
    )


@pytest.mark.skipif(not HAVE_ASE, reason="this tests requires ASE neighbors list")
def test_neighbors_autograd_errors():
    n_atoms = 20
    cell_size = 6.0
    positions = cell_size * torch.rand(
        n_atoms, 3, dtype=torch.float64, requires_grad=True
    )
    cell = cell_size * torch.eye(3, dtype=torch.float64, requires_grad=True)

    atoms = ase.Atoms(
        "C" * positions.shape[0],
        positions=positions.detach().numpy(),
        cell=cell.detach().numpy(),
        pbc=True,
    )
    options = NeighborsListOptions(cutoff=2.0, full_list=False)
    neighbors = _compute_ase_neighbors(
        atoms, options, dtype=torch.float64, device="cpu"
    )
    system = System(torch.from_numpy(atoms.numbers).to(torch.int32), positions, cell)
    register_autograd_neighbors(system, neighbors)

    message = (
        "`neighbors` is already part of a computational graph, "
        "detach it before calling `register_autograd_neighbors\\(\\)`"
    )
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors)

    message = (
        "one neighbor pair does not match its metadata: the pair between atom 0 and "
        "atom 4 for the \\[0, 0, 0\\] cell shift should have a distance vector of "
        "\\[0.489917, 1.24926, 0.102936\\] but has a distance vector of "
        "\\[1.46975, 3.74777, 0.308807\\]"
    )
    neighbors = _compute_ase_neighbors(
        atoms, options, dtype=torch.float64, device="cpu"
    )
    neighbors.values[:] *= 3
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors, check_consistency=True)

    neighbors = _compute_ase_neighbors(
        atoms, options, dtype=torch.float64, device="cpu"
    )
    message = (
        "`system` and `neighbors` must have the same dtype, "
        "got torch.float32 and torch.float64"
    )
    system = System(
        torch.from_numpy(atoms.numbers).to(torch.int32),
        positions.to(torch.float32),
        cell.to(torch.float32),
    )
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors, check_consistency=True)

    message = "`system` and `neighbors` must be on the same device, got meta and cpu"
    system = System(
        torch.from_numpy(atoms.numbers).to(torch.int32).to(torch.device("meta")),
        positions.to(torch.device("meta")),
        cell.to(torch.device("meta")),
    )
    with pytest.raises(ValueError, match=message):
        register_autograd_neighbors(system, neighbors, check_consistency=True)
