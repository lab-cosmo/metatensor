from typing import Dict, List, Optional

import torch

from .. import Labels, TensorMap, dtype_name
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
    https://docs.metatensor.org/latest/atomistic/outputs.html
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
                    f"the model promised {dtype_name(expected_dtype)}, "
                    f"we got {dtype_name(output_dtype)}"
                )

    for name, request in requested.items():
        value = outputs.get(name)
        if value is None:
            raise ValueError(
                f"the model did not produce the '{name}' output, which was requested"
            )

        if name == "energy":
            _check_energy_like(
                "energy",
                value,
                systems,
                request,
                selected_atoms,
            )
        elif name == "energy_ensemble":
            _check_energy_like(
                "energy_ensemble",
                value,
                systems,
                request,
                selected_atoms,
            )
        else:
            # this is a non-standard output, there is nothing to check
            continue


def _check_energy_like(
    name: str,
    value: TensorMap,
    systems: List[System],
    request: ModelOutput,
    selected_atoms: Optional[Labels],
):
    """
    Check either "energy" or "energy_ensemble" output metadata
    """

    assert name in ["energy", "energy_ensemble"]

    if value.keys != Labels("_", torch.tensor([[0]])):
        raise ValueError(
            f"invalid keys for '{name}' output: expected `Labels('_', [[0]])`"
        )

    device = value.device
    energy_block = value.block_by_id(0)

    if request.per_atom:
        expected_samples_names = ["system", "atom"]
    else:
        expected_samples_names = ["system"]

    if energy_block.samples.names != expected_samples_names:
        raise ValueError(
            f"invalid sample names for '{name}' output: "
            f"expected {expected_samples_names}, got {energy_block.samples.names}"
        )

    # check samples values from systems & selected_atoms
    if request.per_atom:
        expected_values: List[List[int]] = []
        for s, system in enumerate(systems):
            for a in range(len(system)):
                expected_values.append([s, a])

        expected_samples = Labels(
            ["system", "atom"], torch.tensor(expected_values, device=device)
        )
        if selected_atoms is not None:
            expected_samples = expected_samples.intersection(selected_atoms)

        if len(expected_samples.union(energy_block.samples)) != len(expected_samples):
            raise ValueError(
                f"invalid samples entries for '{name}' output, they do not match the "
                f"`systems` and `selected_atoms`. Expected samples:\n{expected_samples}"
            )

    else:
        expected_samples = Labels(
            "system", torch.arange(len(systems), device=device).reshape(-1, 1)
        )
        if selected_atoms is not None:
            selected_systems = Labels(
                "system", torch.unique(selected_atoms.column("system")).reshape(-1, 1)
            )
            expected_samples = expected_samples.intersection(selected_systems)

        if len(expected_samples.union(energy_block.samples)) != len(expected_samples):
            raise ValueError(
                f"invalid samples entries for '{name}' output, they do not match the "
                f"`systems` and `selected_atoms`. Expected samples:\n{expected_samples}"
            )

    if len(energy_block.components) != 0:
        raise ValueError(
            f"invalid components for '{name}' output: components should be empty"
        )

    # the only difference between energy & energy_ensemble is in the properties
    if name == "energy":
        expected_properties = Labels("energy", torch.tensor([[0]], device=device))
        message = "`Labels('energy', [[0]])`"
    else:
        assert name == "energy_ensemble"
        n_ensemble_members = energy_block.values.shape[-1]
        expected_properties = Labels(
            "energy", torch.arange(n_ensemble_members, device=device).reshape(-1, 1)
        )
        message = "`Labels('energy', [[0], ..., [n]])`"

    if energy_block.properties != expected_properties:
        raise ValueError(f"invalid properties for '{name}' output: expected {message}")

    for parameter, gradient in energy_block.gradients():
        if parameter not in ["strain", "positions"]:
            raise ValueError(f"invalid gradient for '{name}' output: {parameter}")

        xyz = torch.tensor([[0], [1], [2]], device=device)
        # strain gradient checks
        if parameter == "strain":
            if gradient.samples.names != ["sample"]:
                raise ValueError(
                    f"invalid samples for '{name}' output 'strain' gradients: "
                    f"expected the names to be ['sample'], got {gradient.samples.names}"
                )

            if len(gradient.components) != 2:
                raise ValueError(
                    f"invalid components for '{name}' output 'strain' gradients: "
                    "expected two components"
                )

            if gradient.components[0] != Labels("xyz_1", xyz):
                raise ValueError(
                    f"invalid components for '{name}' output 'strain' gradients: "
                    "expected Labels('xyz_1', [[0], [1], [2]]) for the first component"
                )

            if gradient.components[1] != Labels("xyz_2", xyz):
                raise ValueError(
                    f"invalid components for '{name}' output 'strain' gradients: "
                    "expected Labels('xyz_2', [[0], [1], [2]]) for the second component"
                )

        # positions gradient checks
        if parameter == "positions":
            if gradient.samples.names != ["sample", "system", "atom"]:
                raise ValueError(
                    f"invalid samples for '{name}' output 'positions' gradients: "
                    "expected the names to be ['sample', 'system', 'atom'], "
                    f"got {gradient.samples.names}"
                )

            if len(gradient.components) != 1:
                raise ValueError(
                    f"invalid components for '{name}' output 'positions' gradients: "
                    "expected one component"
                )

            if gradient.components[0] != Labels("xyz", xyz):
                raise ValueError(
                    f"invalid components for '{name}' output 'positions' gradients: "
                    "expected Labels('xyz', [[0], [1], [2]]) for the first component"
                )
