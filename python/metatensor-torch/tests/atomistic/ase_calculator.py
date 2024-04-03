import os
from typing import Dict, List, Optional

import ase.build
import ase.calculators.lj
import ase.md
import ase.units
import numpy as np
import pytest
import torch

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborsListOptions,
    System,
)
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator


class LennardJones(torch.nn.Module):
    """
    Implementation of Lennard-Jones potential using the ``MetatensorAtomisticModule``
    API. This is then compared against the implementation inside ASE.
    """

    def __init__(self, cutoff, epsilon, sigma):
        super().__init__()
        self._nl_options = NeighborsListOptions(cutoff=cutoff, full_list=False)

        self._epsilon = epsilon
        self._sigma = sigma

        # shift the energy to 0 at the cutoff
        self._shift = 4 * epsilon * ((sigma / cutoff) ** 12 - (sigma / cutoff) ** 6)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if "energy" not in outputs:
            return {}

        per_atoms = outputs["energy"].per_atom

        all_energies = []
        for system_i, system in enumerate(systems):
            neighbors = system.get_neighbors_list(self._nl_options)
            all_i = neighbors.samples.column("first_atom").to(torch.long)
            all_j = neighbors.samples.column("second_atom").to(torch.long)

            energy = torch.zeros(len(system), dtype=system.positions.dtype)

            distances = neighbors.values.reshape(-1, 3)

            sigma_r_6 = (self._sigma / torch.linalg.vector_norm(distances, dim=1)) ** 6
            sigma_r_12 = sigma_r_6 * sigma_r_6
            e = 4.0 * self._epsilon * (sigma_r_12 - sigma_r_6) - self._shift

            # We only compute each pair once (full_list=False in self._nl_options),
            # and assign half of the energy to each atom
            energy = energy.index_add(0, all_i, e, alpha=0.5)
            energy = energy.index_add(0, all_j, e, alpha=0.5)

            if selected_atoms is not None:
                current_system_mask = selected_atoms.column("system") == system_i
                current_atoms = selected_atoms.column("atom")
                current_atoms = current_atoms[current_system_mask].to(torch.long)
                energy = energy[current_atoms]

            if per_atoms:
                all_energies.append(energy)
            else:
                all_energies.append(energy.sum(0, keepdim=True))

        if per_atoms:
            if selected_atoms is None:
                samples_list: List[List[int]] = []
                for s, system in enumerate(systems):
                    for a in range(len(system)):
                        samples_list.append([s, a])

                samples = Labels(["system", "atom"], torch.tensor(samples_list))
            else:
                samples = selected_atoms
        else:
            samples = Labels(["system"], torch.arange(len(systems)).reshape(-1, 1))

        block = TensorBlock(
            values=torch.vstack(all_energies).reshape(-1, 1),
            samples=samples,
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels(["energy"], torch.tensor([[0]])),
        )
        return {
            "energy": TensorMap(Labels("_", torch.tensor([[0]])), [block]),
        }

    def requested_neighbors_lists(self) -> List[NeighborsListOptions]:
        return [self._nl_options]


CUTOFF = 5.0
SIGMA = 1.5808
EPSILON = 0.1729


@pytest.fixture
def model():
    model = LennardJones(cutoff=CUTOFF, sigma=SIGMA, epsilon=EPSILON)
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="Angstrom",
        interaction_range=CUTOFF,
        atomic_types=[28],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
                per_atom=True,
                explicit_gradients=[],
            ),
        },
        supported_devices=["cpu"],
        dtype="float64",
    )

    metadata = ModelMetadata()
    return MetatensorAtomisticModel(model, metadata, capabilities)


@pytest.fixture
def model_different_units():
    model = LennardJones(
        cutoff=CUTOFF / ase.units.Bohr,
        sigma=SIGMA / ase.units.Bohr,
        epsilon=EPSILON / ase.units.kJ * ase.units.mol,
    )
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="Bohr",
        interaction_range=CUTOFF,
        atomic_types=[28],
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="kJ/mol",
                per_atom=True,
                explicit_gradients=[],
            ),
        },
        supported_devices=["cpu"],
        dtype="float64",
    )

    metadata = ModelMetadata()
    return MetatensorAtomisticModel(model, metadata, capabilities)


@pytest.fixture
def atoms():
    np.random.seed(0xDEADBEEF)

    atoms = ase.build.make_supercell(
        ase.build.bulk("Ni", "fcc", a=3.6, cubic=True), 2 * np.eye(3)
    )
    atoms.positions += 0.2 * np.random.rand(*atoms.positions.shape)

    return atoms


def check_against_ase_lj(atoms, calculator):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )

    atoms.calc = calculator

    assert np.allclose(ref.get_potential_energy(), atoms.get_potential_energy())
    assert np.allclose(ref.get_potential_energies(), atoms.get_potential_energies())
    assert np.allclose(ref.get_forces(), atoms.get_forces())
    assert np.allclose(ref.get_stress(), atoms.get_stress())


def test_python_model(model, model_different_units, atoms):
    check_against_ase_lj(atoms, MetatensorCalculator(model, check_consistency=True))
    check_against_ase_lj(
        atoms, MetatensorCalculator(model_different_units, check_consistency=True)
    )


def test_torch_script_model(model, model_different_units, atoms):
    model = torch.jit.script(model)
    check_against_ase_lj(atoms, MetatensorCalculator(model, check_consistency=True))

    model_different_units = torch.jit.script(model_different_units)
    check_against_ase_lj(
        atoms, MetatensorCalculator(model_different_units, check_consistency=True)
    )


def test_exported_model(tmpdir, model, model_different_units, atoms):
    path = os.path.join(tmpdir, "exported-model.pt")
    model.export(path)
    check_against_ase_lj(atoms, MetatensorCalculator(path, check_consistency=True))

    model_different_units.export(path)
    check_against_ase_lj(atoms, MetatensorCalculator(path, check_consistency=True))


def test_get_properties(model, atoms):
    atoms.calc = MetatensorCalculator(model, check_consistency=True)

    properties = atoms.get_properties(["energy", "energies", "forces", "stress"])

    assert np.all(properties["energies"] == atoms.get_potential_energies())
    assert np.all(properties["energy"] == atoms.get_potential_energy())
    assert np.all(properties["forces"] == atoms.get_forces())
    assert np.all(properties["stress"] == atoms.get_stress())


def test_selected_atoms(tmpdir, model, atoms):
    ref = atoms.copy()
    ref.calc = ase.calculators.lj.LennardJones(
        sigma=SIGMA, epsilon=EPSILON, rc=CUTOFF, ro=CUTOFF, smooth=False
    )

    path = os.path.join(tmpdir, "exported-model.pt")
    model.export(path)
    calculator = MetatensorCalculator(path, check_consistency=True)

    first_mask = [a % 2 == 0 for a in range(len(atoms))]
    first_half = Labels(
        ["system", "atom"],
        torch.tensor([[0, a] for a in range(len(atoms)) if a % 2 == 0]),
    )

    second_mask = [a % 2 == 1 for a in range(len(atoms))]
    second_half = Labels(
        ["system", "atom"],
        torch.tensor([[0, a] for a in range(len(atoms)) if a % 2 == 1]),
    )

    # check per atom energy
    requested = {"energy": ModelOutput(per_atom=True)}
    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=first_half)
    first_energies = outputs["energy"].block().values.numpy().reshape(-1)

    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=second_half)
    second_energies = outputs["energy"].block().values.numpy().reshape(-1)

    expected = ref.get_potential_energies()
    assert np.allclose(expected[first_mask], first_energies)
    assert np.allclose(expected[second_mask], second_energies)

    # check total energy
    requested = {"energy": ModelOutput(per_atom=False)}
    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=first_half)
    first_energies = outputs["energy"].block().values.numpy().reshape(-1)

    outputs = calculator.run_model(atoms, outputs=requested, selected_atoms=second_half)
    second_energies = outputs["energy"].block().values.numpy().reshape(-1)

    expected = ref.get_potential_energy()
    assert np.allclose(expected, first_energies + second_energies)


def test_serialize_ase(tmpdir, model, atoms):
    # Run some tests with a different dtype
    model._capabilities.dtype = "float32"

    calculator = MetatensorCalculator(model)

    message = (
        "can not save metatensor model in ASE `todict`, please initialize "
        "`MetatensorCalculator` with a path to a saved model file if you need to use "
        "`todict"
    )
    with pytest.raises(RuntimeError, match=message):
        calculator.todict()

    # save with exported model
    path = os.path.join(tmpdir, "exported-model.pt")
    model.export(path)

    calculator = MetatensorCalculator(path)
    data = calculator.todict()
    _ = MetatensorCalculator.fromdict(data)

    # check the standard trajectory format of ASE, which uses `todict`/`fromdict`
    atoms.calc = MetatensorCalculator(path)
    with tmpdir.as_cwd():
        dyn = ase.md.VelocityVerlet(
            atoms,
            timestep=2 * ase.units.fs,
            trajectory="file.traj",
        )
        dyn.run(10)
        dyn.close()

        atoms = ase.io.read("file.traj", "-1")
        assert atoms.calc is not None
