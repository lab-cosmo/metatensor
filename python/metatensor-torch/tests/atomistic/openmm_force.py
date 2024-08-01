import os

import ase.io
import metatensor_lj_test
import numpy as np
import pytest
from ase.build import bulk

from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
from metatensor.torch.atomistic.openmm_force import get_metatensor_force


try:
    import NNPOps  # noqa: F401
    import openmm
    import openmmtorch  # noqa: F401

    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729


def model():
    return metatensor_lj_test.lennard_jones_model(
        atomic_type=29,
        cutoff=CUTOFF,
        sigma=SIGMA,
        epsilon=EPSILON,
        length_unit="Angstrom",
        energy_unit="eV",
        with_extension=False,
    )


def model_different_units():
    return metatensor_lj_test.lennard_jones_model(
        atomic_type=29,
        cutoff=CUTOFF / ase.units.Bohr,
        sigma=SIGMA / ase.units.Bohr,
        epsilon=EPSILON / ase.units.kJ * ase.units.mol,
        length_unit="Bohr",
        energy_unit="kJ/mol",
        with_extension=False,
    )


def _modify_pdb(path):
    # makes the ASE pdb file compatible with OpenMM
    with open(path, "r") as f:
        lines = f.readlines()
    count = 0
    new_lines = []
    for line in lines:
        if "Cu" in line:
            count += 1
        line = line.replace(" Cu", f"Cu{count}")
        new_lines.append(line)
    with open(path, "w") as f:
        f.writelines(new_lines)


def _check_against_ase(tmpdir, atoms):
    model_path = os.path.join(tmpdir, "model.pt")
    structure_path = os.path.join(tmpdir, "structure.pdb")

    ase.io.write(structure_path, atoms)
    _modify_pdb(structure_path)

    topology = openmm.app.PDBFile(structure_path).getTopology()
    system = openmm.System()
    for atom in topology.atoms():
        system.addParticle(atom.element.mass)
    if atoms.pbc.any():
        system.setDefaultPeriodicBoxVectors(
            atoms.cell[0] * openmm.unit.angstrom,
            atoms.cell[1] * openmm.unit.angstrom,
            atoms.cell[2] * openmm.unit.angstrom,
        )
    force = get_metatensor_force(system, topology, model_path, check_consistency=True)
    system.addForce(force)
    integrator = openmm.VerletIntegrator(0.001 * openmm.unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("CPU")
    properties = {}
    simulation = openmm.app.Simulation(
        topology, system, integrator, platform, properties
    )
    simulation.context.setPositions(openmm.app.PDBFile(structure_path).getPositions())
    state = simulation.context.getState(getForces=True)
    openmm_forces = (
        state.getForces(asNumpy=True).value_in_unit(
            openmm.unit.ev / (openmm.unit.angstrom * openmm.unit.mole)
        )
        / 6.0221367e23
    )

    atoms = ase.io.read(structure_path)
    calculator = MetatensorCalculator(model_path, check_consistency=True)
    atoms.set_calculator(calculator)
    ase_forces = atoms.get_forces()

    assert np.allclose(openmm_forces, ase_forces)


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not available")
def test_diagonal_cell(tmpdir):
    cell = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]
    )
    atoms = bulk("Cu", cubic=False)
    atoms *= (2, 2, 2)
    atoms.positions += np.random.rand(*atoms.positions.shape) * 0.1
    atoms.cell = cell
    atoms.pbc = True
    atoms.wrap()

    m = model()
    m.save(os.path.join(tmpdir, "model.pt"))

    _check_against_ase(tmpdir, atoms)


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not available")
def test_non_diagonal_cell(tmpdir):
    cell = np.array(
        [
            [10.0, 0.0, 0.0],
            [3.0, 10.0, 0.0],
            [3.0, -3.0, 10.0],
        ]
    )
    atoms = bulk("Cu", cubic=False)
    atoms *= (2, 2, 2)
    atoms.positions += np.random.rand(*atoms.positions.shape) * 0.1
    atoms.cell = cell
    atoms.pbc = True
    atoms.wrap()

    m = model()
    m.save(os.path.join(tmpdir, "model.pt"))

    _check_against_ase(tmpdir, atoms)


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not available")
def test_non_diagonal_cell_different_units(tmpdir):
    cell = np.array(
        [
            [100.0, 0.0, 0.0],
            [3.0, 100.0, 0.0],
            [3.0, -3.0, 100.0],
        ]
    )
    atoms = bulk("Cu", cubic=False)
    atoms *= (2, 2, 2)
    atoms.positions += np.random.rand(*atoms.positions.shape) * 0.1
    atoms.cell = cell
    atoms.pbc = True
    atoms.wrap()

    m = model_different_units()
    m.save(os.path.join(tmpdir, "model.pt"))

    _check_against_ase(tmpdir, atoms)


@pytest.mark.skipif(not HAS_OPENMM, reason="OpenMM not available")
def test_no_cell(tmpdir):
    atoms = bulk("Cu", cubic=False)
    atoms *= (2, 2, 2)
    atoms.positions += np.random.rand(*atoms.positions.shape) * 0.1
    atoms.cell = None
    atoms.pbc = False
    atoms.wrap()

    m = model()
    m.save(os.path.join(tmpdir, "model.pt"))

    _check_against_ase(tmpdir, atoms)
