from typing import Iterable, List, Optional

import torch

from metatensor.torch import Labels, TensorBlock
from metatensor.torch.atomistic import (
    ModelEvaluationOptions,
    ModelOutput,
    NeighborListOptions,
    System,
    load_atomistic_model,
)


try:
    import openmm
    import openmmtorch
    from NNPOps.neighbors import getNeighborPairs

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
    check_consistency: bool = False,
) -> openmm.System:

    if not HAS_OPENMM:
        raise ImportError(
            "Could not import openmm and/or nnpops. If you want to use metatensor with "
            "openmm, please install openmm-torch and nnpops with conda."
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
            check_consistency: bool,
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

            requested_nls = self.model.requested_neighbor_lists()
            if len(requested_nls) > 1:
                raise ValueError(
                    "The model requested more than one neighbor list. "
                    "Currently, only models with a single neighbor list are supported "
                    "by the OpenMM interface."
                )
            elif len(requested_nls) == 1:
                self.requested_neighbor_list = requested_nls[0]
            else:
                # no neighbor list requested
                self.requested_neighbor_list = None

            self.check_consistency = check_consistency

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
            system = _attach_neighbors(system, self.requested_neighbor_list)

            energy = (
                self.model(
                    [system],
                    self.evaluation_options,
                    check_consistency=self.check_consistency,
                )["energy"]
                .block()
                .values.reshape(())
            )
            return energy

    metatensor_force = MetatensorForce(
        model,
        atomic_numbers,
        selected_atoms,
        check_consistency,
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


def _attach_neighbors(
    system: System, requested_nl_options: NeighborListOptions
) -> System:

    if requested_nl_options is None:
        return system

    cell: Optional[torch.Tensor] = None
    if not torch.all(system.cell == 0.0):
        cell = system.cell

    # Get the neighbor pairs, shifts and edge indices.
    neighbors, interatomic_vectors, _, _ = getNeighborPairs(
        system.positions,
        requested_nl_options.engine_cutoff("nm"),
        -1,
        cell,
    )
    mask = neighbors[0] >= 0
    neighbors = neighbors[:, mask]
    neighbors = neighbors.flip(0)  # [neighbor, center] -> [center, neighbor]
    interatomic_vectors = interatomic_vectors[mask, :]

    if requested_nl_options.full_list:
        neighbors = torch.concatenate((neighbors, neighbors.flip(0)), dim=1)
        interatomic_vectors = torch.concatenate(
            (interatomic_vectors, -interatomic_vectors)
        )

    if cell is not None:
        interatomic_vectors_unit_cell = (
            system.positions[neighbors[1]] - system.positions[neighbors[0]]
        )
        cell_shifts = (
            interatomic_vectors_unit_cell - interatomic_vectors
        ) @ torch.linalg.inv(cell)
        cell_shifts = torch.round(cell_shifts).to(torch.int32)
    else:
        cell_shifts = torch.zeros(
            (neighbors.shape[1], 3),
            dtype=torch.int32,
            device=system.positions.device,
        )

    neighbor_list = TensorBlock(
        values=interatomic_vectors.reshape(-1, 3, 1),
        samples=Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=torch.concatenate([neighbors.T, cell_shifts], dim=-1),
        ),
        components=[
            Labels(
                names=["xyz"],
                values=torch.arange(
                    3, dtype=torch.int32, device=system.positions.device
                ).reshape(-1, 1),
            )
        ],
        properties=Labels(
            names=["distance"],
            values=torch.tensor(
                [[0]], dtype=torch.int32, device=system.positions.device
            ),
        ),
    )

    system.add_neighbor_list(requested_nl_options, neighbor_list)
    return system
