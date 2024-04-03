from typing import Dict, List, Optional

import torch

from .. import Labels, TensorMap
from . import ModelOutput, System


def _check_outputs(
    systems: List[System],
    requested: Dict[str, ModelOutput],
    selected_atoms: Optional[Labels],
    outputs: Dict[str, TensorMap],
    expected_dtype: torch.dtype,
):
    """
    Check that the outputs of a model conform to the expected structure for metatensor
    atomistic models.

    This function checks conformance with the reference documentation in
    https://lab-cosmo.github.io/metatensor/latest/atomistic/outputs.html
    """

    for name, output in outputs.items():
        if requested.get(name) is None:
            raise ValueError(
                f"the model produced an output named '{name}', which was not requested"
            )

        if len(output) != 0:
            output_dtype = output.block_by_id(0).values.dtype
            if output_dtype != expected_dtype:
                raise ValueError(
                    f"wrong dtype for the {name} output: "
                    f"the model promised {expected_dtype}, we got {output_dtype}"
                )

    for name, request in requested.items():
        value = outputs.get(name)
        if value is None:
            raise ValueError(
                f"the model did not produce the '{name}' output, which was requested"
            )

        if name == "energy":
            _check_energy(systems, request, selected_atoms, energy=value)
        else:
            continue


def _check_energy(
    systems: List[System],
    request: ModelOutput,
    selected_atoms: Optional[Labels],
    energy: TensorMap,
):
    """Check the "energy" output metadata"""
    if energy.keys != Labels("_", torch.tensor([[0]])):
        raise ValueError(
            "invalid keys for 'energy' output: expected `Labels('_', [[0]])`"
        )

    energy_block = energy.block_by_id(0)

    if request.per_atom:
        expected_samples_names = ["system", "atom"]
    else:
        expected_samples_names = ["system"]

    if energy_block.samples.names != expected_samples_names:
        raise ValueError(
            "invalid sample names for 'energy' output: "
            f"expected {expected_samples_names}, got {energy_block.samples.names}"
        )

    # check samples values from systems & selected_atoms
    if request.per_atom:
        expected_values: List[List[int]] = []
        for s, system in enumerate(systems):
            for a in range(len(system)):
                expected_values.append([s, a])

        expected_samples = Labels(
            ["system", "atom"],
            torch.tensor(expected_values, device=energy_block.values.device),
        )
        if selected_atoms is not None:
            expected_samples = expected_samples.intersection(selected_atoms)

        if len(expected_samples.union(energy_block.samples)) != len(expected_samples):
            raise ValueError(
                "invalid samples entries for 'energy' output, they do not match the "
                f"`systems` and `selected_atoms`. Expected samples:\n{expected_samples}"
            )

    else:
        expected_samples = Labels("system", torch.arange(len(systems)).reshape(-1, 1))
        if selected_atoms is not None:
            selected_systems = Labels(
                "system", torch.unique(selected_atoms.column("system")).reshape(-1, 1)
            )
            expected_samples = expected_samples.intersection(selected_systems)

        if len(expected_samples.union(energy_block.samples)) != len(expected_samples):
            raise ValueError(
                "invalid samples entries for 'energy' output, they do not match the "
                f"`systems` and `selected_atoms`. Expected samples:\n{expected_samples}"
            )

    if len(energy_block.components) != 0:
        raise ValueError(
            "invalid components for 'energy' output: components should be empty"
        )

    if energy_block.properties != Labels("energy", torch.tensor([[0]])):
        raise ValueError(
            "invalid properties for 'energy' output: expected `Labels('energy', [[0]])`"
        )

    for parameter, gradient in energy_block.gradients():
        if parameter not in ["strain", "positions"]:
            raise ValueError(f"invalid gradient for 'energy' output: {parameter}")

        # strain gradient checks
        if parameter == "strain":
            if gradient.samples.names != ["sample"]:
                raise ValueError(
                    "invalid samples for 'energy' output 'strain' gradients: "
                    f"expected the names to be ['sample'], got {gradient.samples.names}"
                )

            if len(gradient.components) != 2:
                raise ValueError(
                    "invalid components for 'energy' output 'strain' gradients: "
                    "expected two components"
                )

            if gradient.components[0] != Labels("xyz_1", torch.tensor([[0], [1], [2]])):
                raise ValueError(
                    "invalid components for 'energy' output 'strain' gradients: "
                    "expected Labels('xyz_1', [[0], [1], [2]]) for the first component"
                )

            if gradient.components[1] != Labels("xyz_2", torch.tensor([[0], [1], [2]])):
                raise ValueError(
                    "invalid components for 'energy' output 'strain' gradients: "
                    "expected Labels('xyz_2', [[0], [1], [2]]) for the second component"
                )

        # positions gradient checks
        if parameter == "positions":
            if gradient.samples.names != ["sample", "system", "atom"]:
                raise ValueError(
                    "invalid samples for 'energy' output 'positions' gradients: "
                    "expected the names to be ['sample', 'system', 'atom'], "
                    f"got {gradient.samples.names}"
                )

            if len(gradient.components) != 1:
                raise ValueError(
                    "invalid components for 'energy' output 'positions' gradients: "
                    "expected one component"
                )

            if gradient.components[0] != Labels("xyz", torch.tensor([[0], [1], [2]])):
                raise ValueError(
                    "invalid components for 'energy' output 'positions' gradients: "
                    "expected Labels('xyz', [[0], [1], [2]]) for the first component"
                )
