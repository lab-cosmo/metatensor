from typing import Iterable, List, Optional

import torch

from metatensor.torch import Labels
from metatensor.torch.atomistic import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
)


try:
    import openmm
    import openmmtorch

    HAS_OPENMM = True
except ImportError:

    class MLPotential:
        pass

    class MLPotentialImpl:
        pass

    class MLPotentialImplFactory:
        pass

    HAS_OPENMM = False


def get_metatensor_force(
    system: openmm.System,
    topology: openmm.app.Topology,
    path: str,
    extensions_directory: Optional[str] = None,
    forceGroup: int = 0,
    atoms: Optional[Iterable[int]] = None,
) -> openmm.System:

    if not HAS_OPENMM:
        raise ImportError(
            "Could not import openmm. If you want to use metatensor with "
            "openmm, please install openmm-ml with conda."
        )

    model = load_atomistic_model(path, extensions_directory=extensions_directory)

    # Get the atomic numbers of the ML region.
    all_atoms = list(topology.atoms())
    atomic_numbers = [atom.element.atomic_number for atom in all_atoms]

    if atoms is None:
        selected_atoms = None
    else:
        selected_atoms = Labels(
            names=["system", "atom"],
            values=torch.tensor(
                [[0, selected_atom] for selected_atom in atoms],
                dtype=torch.int32,
            ),
        )

    class MetatensorForce(torch.nn.Module):

        def __init__(
            self,
            model: torch.jit._script.RecursiveScriptModule,
            atomic_numbers: List[int],
            selected_atoms: Optional[Labels],
        ) -> None:
            super(MetatensorForce, self).__init__()

            self.model = model
            self.register_buffer(
                "atomic_numbers", torch.tensor(atomic_numbers, dtype=torch.int32)
            )
            self.evaluation_options = ModelEvaluationOptions(
                length_unit="nm",
                outputs={
                    "energy": ModelOutput(
                        quantity="energy",
                        unit="kJ/mol",
                        per_atom=False,
                    ),
                },
                selected_atoms=selected_atoms,
            )

        def forward(
            self, positions: torch.Tensor, cell: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            # move labels if necessary
            selected_atoms = self.evaluation_options.selected_atoms
            if selected_atoms is not None:
                if selected_atoms.device != positions.device:
                    self.evaluation_options.selected_atoms = selected_atoms.to(
                        positions.device
                    )

            if cell is None:
                cell = torch.zeros(
                    (3, 3), dtype=positions.dtype, device=positions.device
                )

            # create System
            system = System(
                types=self.atomic_numbers,
                positions=positions,
                cell=cell,
            )

            energy = (
                self.model([system], self.evaluation_options, check_consistency=True)[
                    "energy"
                ]
                .block()
                .values.reshape(())
            )
            return energy

    metatensor_force = MetatensorForce(
        model,
        atomic_numbers,
        selected_atoms,
    )

    # torchscript everything
    module = torch.jit.script(metatensor_force)

    # create the OpenMM force
    force = openmmtorch.TorchForce(module)
    isPeriodic = (
        topology.getPeriodicBoxVectors() is not None
    ) or system.usesPeriodicBoundaryConditions()
    force.setUsesPeriodicBoundaryConditions(isPeriodic)
    force.setForceGroup(forceGroup)

    return force
