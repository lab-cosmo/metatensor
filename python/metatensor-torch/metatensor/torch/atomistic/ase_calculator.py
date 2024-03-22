import logging
import os
import pathlib
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .. import Labels, TensorBlock
from . import (
    MetatensorAtomisticModel,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    check_atomistic_model,
    register_autograd_neighbors,
)


import ase  # isort: skip
import ase.neighborlist  # isort: skip
import ase.calculators.calculator  # isort: skip
from ase.calculators.calculator import (  # isort: skip
    InputError,
    PropertyNotImplementedError,
    all_properties as ALL_ASE_PROPERTIES,
)


if os.environ.get("METATENSOR_IMPORT_FOR_SPHINX", "0") == "0":
    # this can not be imported when building the documentation
    from .. import sum_over_samples  # isort: skip

FilePath = Union[str, bytes, pathlib.PurePath]

LOGGER = logging.getLogger(__name__)


STR_TO_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class MetatensorCalculator(ase.calculators.calculator.Calculator):
    """
    The :py:class:`MetatensorCalculator` class implements ASE's
    :py:class:`ase.calculators.calculator.Calculator` API using metatensor atomistic
    models to compute energy, forces and any other supported property.

    This class can be initialized with any :py:class:`MetatensorAtomisticModel`, and
    used to run simulations using ASE's MD facilities.
    """

    def __init__(
        self,
        model: Union[FilePath, MetatensorAtomisticModel],
        check_consistency=False,
        device=None,
    ):
        """
        :param model: model to use for the calculation. This can be a file path, a
            Python instance of :py:class:`MetatensorAtomisticModel`, or the output of
            :py:func:`torch.jit.script` on :py:class:`MetatensorAtomisticModel`.
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        :param device: torch device to use for the calculation. If ``None``, we will try
            the options in the model's ``supported_device`` in order.
        """
        super().__init__()

        self.parameters = {
            "check_consistency": check_consistency,
        }

        # Load the model
        if isinstance(model, (str, bytes, pathlib.PurePath)):
            if not os.path.exists(model):
                raise InputError(f"given model path '{model}' does not exist")

            self.parameters["model_path"] = str(model)
            check_atomistic_model(model)
            model = torch.jit.load(model)

        elif isinstance(model, torch.jit.RecursiveScriptModule):
            if model.original_name != "MetatensorAtomisticModel":
                raise InputError(
                    "torch model must be 'MetatensorAtomisticModel', "
                    f"got '{model.original_name}' instead"
                )
        elif isinstance(model, MetatensorAtomisticModel):
            # nothing to do
            pass
        else:
            raise TypeError(f"unknown type for model: {type(model)}")

        self.parameters["device"] = str(device) if device is not None else None
        # check if the model supports the requested device
        capabilities = model.capabilities()
        if device is None:
            device = _find_best_device(capabilities.supported_devices)
        else:
            device = torch.device(device)
            device_is_supported = False

            for supported in capabilities.supported_devices:
                try:
                    supported = torch.device(supported)
                except RuntimeError as e:
                    warnings.warn(
                        "the model contains an invalid device in `supported_devices`: "
                        f"{e}",
                        stacklevel=2,
                    )
                    continue

                if supported.type == device.type:
                    device_is_supported = True
                    break

            if not device_is_supported:
                raise ValueError(
                    f"This model does not support the requested device ({device}), "
                    "the following devices are supported: "
                    f"{capabilities.supported_devices}"
                )

        if capabilities.dtype in STR_TO_DTYPE:
            self._dtype = STR_TO_DTYPE[capabilities.dtype]
        else:
            raise ValueError(
                f"found unexpected dtype in model capabilities: {capabilities.dtype}"
            )

        self._device = device
        self._model = model.to(device=self._device)

        # We do our own check to verify if a property is implemented in `calculate()`,
        # so we pretend to be able to compute all properties ASE knows about.
        self.implemented_properties = ALL_ASE_PROPERTIES

    def todict(self):
        if "model_path" not in self.parameters:
            raise RuntimeError(
                "can not save metatensor model in ASE `todict`, please initialize "
                "`MetatensorCalculator` with a path to a saved model file if you need "
                "to use `todict`"
            )

        return self.parameters

    @classmethod
    def fromdict(cls, data):
        return MetatensorCalculator(
            model=data["model_path"],
            check_consistency=data["check_consistency"],
            device=data["device"],
        )

    def metadata(self) -> ModelMetadata:
        """Get the metadata of the underlying model"""
        return self._model.metadata()

    def run_model(
        self,
        atoms: ase.Atoms,
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorBlock]:
        """
        Run the model on the given ``atoms``, computing properties according to the
        ``outputs`` and ``selected_atoms`` options.

        The output of the model is returned directly, and as such the blocks' ``values``
        will be :py:class:`torch.Tensor`.

        This is intended as an easy way to run metatensor models on
        :py:class:`ase.Atoms` when the model can predict properties not supported by the
        usual ASE's calculator interface.

        All the parameters have the same meaning as the corresponding ones in
        :py:meth:`metatensor.torch.atomistic.ModelInterface.forward`.

        :param atoms: system on which to run the model
        :param outputs: outputs of the model that should be predicted
        :param selected_atoms: subset of atoms on which to run the calculation
        """
        types, positions, cell = _ase_to_torch_data(
            atoms=atoms, dtype=self._dtype, device=self._device
        )
        system = System(types, positions, cell)

        # Compute the neighbors lists requested by the model using ASE NL
        for options in self._model.requested_neighbors_lists():
            neighbors = _compute_ase_neighbors(
                atoms, options, dtype=self._dtype, device=self._device
            )
            register_autograd_neighbors(
                system,
                neighbors,
                check_consistency=self.parameters["check_consistency"],
            )
            system.add_neighbors_list(options, neighbors)

        options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs=outputs,
            selected_atoms=selected_atoms,
        )
        return self._model(
            systems=[system],
            options=options,
            check_consistency=self.parameters["check_consistency"],
        )

    def calculate(
        self,
        atoms: ase.Atoms,
        properties: List[str],
        system_changes: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Compute some ``properties`` with this calculator, and return them in the format
        expected by ASE.

        This is not intended to be called directly by users, but to be an implementation
        detail of ``atoms.get_energy()`` and related functions. See
        :py:meth:`ase.calculators.calculator.Calculator.calculate` for more information.
        """
        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        outputs = _ase_properties_to_metatensor_outputs(properties)
        capabilities = self._model.capabilities()
        for name in outputs.keys():
            if name not in capabilities.outputs:
                raise ValueError(
                    f"you asked for the calculation of {name}, but this model does not "
                    "support it"
                )

        types, positions, cell = _ase_to_torch_data(
            atoms=atoms, dtype=self._dtype, device=self._device
        )

        do_backward = False
        if "forces" in properties:
            do_backward = True
            positions.requires_grad_(True)

        if "stress" in properties:
            do_backward = True

            scaling = torch.eye(3, requires_grad=True, dtype=self._dtype)

            positions = positions @ scaling
            positions.retain_grad()

            cell = cell @ scaling

        if "stresses" in properties:
            raise NotImplementedError("'stresses' are not implemented yet")

        # convert from ase.Atoms to metatensor.torch.atomistic.System
        system = System(types, positions, cell)
        for options in self._model.requested_neighbors_lists():
            neighbors = _compute_ase_neighbors(
                atoms, options, dtype=self._dtype, device=self._device
            )
            register_autograd_neighbors(
                system,
                neighbors,
                check_consistency=self.parameters["check_consistency"],
            )
            system.add_neighbors_list(options, neighbors)

        run_options = ModelEvaluationOptions(
            length_unit="angstrom",
            outputs=outputs,
            selected_atoms=None,
        )

        outputs = self._model(
            [system],
            run_options,
            check_consistency=self.parameters["check_consistency"],
        )
        energy = outputs["energy"]

        if run_options.outputs["energy"].per_atom:
            assert len(energy) == 1
            assert energy.sample_names == ["system", "atom"]
            assert torch.all(energy.block().samples["system"] == 0)
            assert torch.all(
                energy.block().samples["atom"] == torch.arange(positions.shape[0])
            )
            energies = energy.block().values
            assert energies.shape == (len(atoms), 1)

            energy = sum_over_samples(energy, sample_names=["atom"])

        assert len(energy.block().gradients_list()) == 0
        energy = energy.block().values
        assert energy.shape == (1, 1)

        self.results = {}

        if "energies" in properties:
            energies_values = energies.detach().reshape(-1)
            energies_values = energies_values.to(device="cpu").to(dtype=torch.float64)
            self.results["energies"] = energies_values.numpy()

        if "energy" in properties:
            energy_values = energy.detach()
            energy_values = energy_values.to(device="cpu").to(dtype=torch.float64)
            self.results["energy"] = energy_values.numpy()[0, 0]

        if do_backward:
            energy.backward(-torch.ones_like(energy))

        if "forces" in properties:
            forces_values = system.positions.grad.reshape(-1, 3)
            forces_values = forces_values.to(device="cpu").to(dtype=torch.float64)
            self.results["forces"] = forces_values.numpy()

        if "stress" in properties:
            stress_values = -scaling.grad.reshape(3, 3) / atoms.cell.volume
            stress_values = stress_values.to(device="cpu").to(dtype=torch.float64)
            self.results["stress"] = _full_3x3_to_voigt_6_stress(stress_values.numpy())


def _find_best_device(devices: List[str]) -> torch.device:
    """
    Find the best device from the list of ``devices`` that is available to the current
    PyTorch installation.
    """

    for device in devices:
        if device == "cpu":
            return torch.device("cpu")
        elif device == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                LOGGER.warning(
                    "the model suggested to use CUDA devices before CPU, "
                    "but we are unable to find it"
                )
        elif device == "mps":
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_built()
                and torch.backends.mps.is_available()
            ):
                return torch.device("mps")
            else:
                LOGGER.warning(
                    "the model suggested to use MPS devices before CPU, "
                    "but we are unable to find it"
                )
        else:
            warnings.warn(
                f"unknown device in the model's `supported_devices`: '{device}'",
                stacklevel=2,
            )

    warnings.warn(
        "could not find a valid device in the model's `supported_devices`, "
        "falling back to CPU",
        stacklevel=2,
    )
    return torch.device("cpu")


def _ase_properties_to_metatensor_outputs(properties):
    energy_properties = []
    for p in properties:
        if p in ["energy", "energies", "forces", "stress", "stresses"]:
            energy_properties.append(p)
        else:
            raise PropertyNotImplementedError(
                f"property '{p}' it not yet supported by this calculator, "
                "even if it might be supported by the model"
            )

    output = ModelOutput()
    output.quantity = "energy"
    output.unit = "ev"
    output.explicit_gradients = []

    if "energies" in properties or "stresses" in properties:
        output.per_atom = True
    else:
        output.per_atom = False

    if "stresses" in properties:
        output.explicit_gradients = ["cell"]

    return {"energy": output}


def _compute_ase_neighbors(atoms, options, dtype, device):
    engine_cutoff = options.engine_cutoff(engine_length_unit="angstrom")
    nl = ase.neighborlist.NeighborList(
        cutoffs=[engine_cutoff] * len(atoms),
        skin=0.0,
        sorted=False,
        self_interaction=False,
        bothways=options.full_list,
        primitive=ase.neighborlist.NewPrimitiveNeighborList,
    )
    nl.update(atoms)

    cell = torch.from_numpy(atoms.cell[:])
    positions = torch.from_numpy(atoms.positions)

    samples = []
    distances = []
    cutoff2 = engine_cutoff * engine_cutoff
    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            distance = positions[j] - positions[i] + offset.dot(cell)

            distance2 = torch.dot(distance, distance).item()

            if distance2 > cutoff2:
                continue

            samples.append((i, j, offset[0], offset[1], offset[2]))
            distances.append(distance.to(dtype=dtype, device=device))

    if len(distances) == 0:
        distances = torch.zeros((0, 3), dtype=dtype, device=device)
        samples = torch.zeros((0, 5), dtype=torch.int32, device=device)
    else:
        samples = torch.tensor(samples, device=device)
        distances = torch.vstack(distances)

    return TensorBlock(
        values=distances.reshape(-1, 3, 1),
        samples=Labels(
            names=[
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            values=samples,
        ),
        components=[Labels.range("xyz", 3).to(device)],
        properties=Labels.range("distance", 1).to(device),
    )


def _ase_to_torch_data(atoms, dtype, device):
    """Get the positions and cell from ASE atoms as torch tensors"""

    types = torch.from_numpy(atoms.numbers).to(dtype=torch.int32, device=device)
    positions = torch.from_numpy(atoms.positions).to(dtype=dtype, device=device)

    if np.all(atoms.pbc):
        cell = torch.from_numpy(atoms.cell[:]).to(dtype=dtype, device=device)
    elif np.any(atoms.pbc):
        raise ValueError(
            f"partial PBC ({atoms.pbc}) are not currently supported in "
            "metatensor atomistic models"
        )
    else:
        cell = torch.zeros((3, 3), dtype=dtype, device=device)

    return types, positions, cell


def _full_3x3_to_voigt_6_stress(stress):
    ase.stress.full_3x3_to_voigt_6_stress
    """
    Re-implementation of ``ase.stress.full_3x3_to_voigt_6_stress`` which does not do the
    stress symmetrization correctly (they do ``(stress[1, 2] + stress[1, 2]) / 2.0``)
    """
    return np.transpose(
        [
            stress[0, 0],
            stress[1, 1],
            stress[2, 2],
            (stress[1, 2] + stress[2, 1]) / 2.0,
            (stress[0, 2] + stress[2, 0]) / 2.0,
            (stress[0, 1] + stress[1, 0]) / 2.0,
        ]
    )
