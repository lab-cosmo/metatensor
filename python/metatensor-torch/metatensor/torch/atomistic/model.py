import datetime
import json
import math
import os
import platform
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.profiler import record_function

from .. import Labels, TensorBlock, TensorMap
from .. import __version__ as metatensor_version
from .. import dtype_name
from . import (
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
    check_atomistic_model,
    load_model_extensions,
    unit_conversion_factor,
)
from ._extensions import _collect_extensions
from .outputs import _check_outputs


def load_atomistic_model(path, extensions_directory=None) -> "MetatensorAtomisticModel":
    """
    Check and then load the metatensor atomistic model at the given `path`.

    This function calls :py:func:`metatensor.torch.atomistic.check_atomistic_model()`
    and :py:func:`metatensor.torch.atomistic.load_model_extensions()` before attempting
    to load the model.

    :param path: path to an exported metatensor model
    :param extensions_directory: path to a directory containing all extensions required
        by the exported model
    """
    path = str(path)
    load_model_extensions(path, str(extensions_directory))
    check_atomistic_model(path)
    return torch.jit.load(path)


class ModelInterface(torch.nn.Module):
    """
    Interface for models that can be used with :py:class:`MetatensorAtomisticModel`.

    There are several requirements that models must satisfy to be usable with
    :py:class:`MetatensorAtomisticModel`. The main one is concerns the
    :py:meth:`forward` function, which must have the signature defined in this
    interface.

    Additionally, the model can request neighbor lists to be computed by the simulation
    engine, and stored inside the input :py:class:`System`. This is done by defining the
    optional :py:meth:`requested_neighbor_lists` method for the model or any of it's
    sub-module.

    :py:class:`MetatensorAtomisticModel` will check if ``requested_neighbor_lists`` is
    defined for all the sub-modules of the model, then collect and unify identical
    requests for the simulation engine.
    """

    def __init__():
        """"""
        pass

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        """
        This method should run the model for the given ``systems``, returning the
        requested ``outputs``. If ``selected_atoms`` is a set of :py:class:`Labels`,
        only the corresponding atoms should be included as "main" atoms in the
        calculation and the output.

        ``outputs`` will be a subset of the capabilities that where declared when
        exporting the model. For example if a model can compute both an ``"energy"`` and
        a ``"charge"`` output, the simulation engine might only request one them.

        The returned dictionary should have the same keys as ``outputs``, and the values
        should contains the corresponding properties of the ``systems``, as computed for
        the subset of atoms defined in ``selected_atoms``. Some outputs are
        standardized, and have additional constrains on how the associated metadata
        should look like, documented in the :ref:`atomistic-models-outputs` section. If
        you want to define a new output for your own usage, it name should looks like
        ``"<domain>::<output>"``, where ``<domain>`` indicates who defines this new
        output and ``<output>`` describes the output itself. For example,
        ``"my-package::foobar"`` for a ``foobar`` output defined in ``my-package``.

        The main use case for ``selected_atoms`` is domain decomposition, where the
        :py:class:`System` given to a model might contain both atoms in the current
        domain and some atoms from other domains; and the calculation should produce
        per-atom output only for the atoms in the domain (but still accounting for atoms
        from the other domains as potential neighbors).

        :param systems: atomistic systems on which to run the calculation
        :param outputs: set of outputs requested by the simulation engine
        :param selected_atoms: subset of atoms that should be included in the output,
            defaults to None
        :return: properties of the systems, as predicted by the machine learning model
        """

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        """
        Optional method declaring which neighbors list this model requires.

        This method can be defined on either the root model or any of it's sub-modules.
        A single module can request multiple neighbors list simultaneously if it needs
        them.

        It is then the responsibility of the code calling the model to:

        1. call this method (or more generally
           :py:meth:`MetatensorAtomisticModel.requested_neighbor_lists`) to get the list
           of the requested neighbor lists.;
        2. compute all neighbor lists corresponding to these requests and add them to
           the systems before calling the model.
        """


# This class name is prefixed with `Metatensor` because we are checking the class name
# before loading a saved model `check_atomistic_model`, to try to prevent people from
# loading arbitrary pytorch models inside the metatensor interface.
class MetatensorAtomisticModel(torch.nn.Module):
    """
    :py:class:`MetatensorAtomisticModel` is the main entry point for atomistic machine
    learning based on metatensor. It is the interface between custom, user-defined
    models and simulation engines. Users should wrap their models with this class, and
    use :py:meth:`save()` to save the wrapped model to a file. The exported models
    can then be loaded by a simulation engine to compute properties of atomistic
    systems.

    When wrapping a ``module``, you should declare what the model is capable of (using
    :py:class:`ModelCapabilities`). This includes what units the model expects as input
    and what properties the model can compute (using :py:class:`ModelOutput`). The
    simulation engine will then ask the model to compute some subset of these properties
    (through a :py:class:`ModelEvaluationOptions`), on all or a subset of atoms of an
    atomistic system.

    The wrapped module must follow the interface defined by :py:class:`ModelInterface`,
    should not already be compiled by TorchScript, and should be in "eval" mode (i.e.
    ``module.training`` should be ``False``).

    For example, a custom module predicting the energy as a constant value times the
    number of atoms could look like this

    >>> class ConstantEnergy(torch.nn.Module):
    ...     def __init__(self, constant: float):
    ...         super().__init__()
    ...         self.constant = torch.tensor(constant).reshape(1, 1)
    ...
    ...     def forward(
    ...         self,
    ...         systems: List[System],
    ...         outputs: Dict[str, ModelOutput],
    ...         selected_atoms: Optional[Labels] = None,
    ...     ) -> Dict[str, TensorMap]:
    ...         results: Dict[str, TensorMap] = {}
    ...         if "energy" in outputs:
    ...             if outputs["energy"].per_atom:
    ...                 raise NotImplementedError("per atom energy is not implemented")
    ...
    ...             dtype = systems[0].positions.dtype
    ...             energies = torch.zeros(len(systems), 1, dtype=dtype)
    ...             for i, system in enumerate(systems):
    ...                 if selected_atoms is None:
    ...                     n_atoms = len(system)
    ...                 else:
    ...                     n_atoms = len(selected_atoms)
    ...
    ...                 energies[i] = self.constant * n_atoms
    ...
    ...             systems_idx = torch.tensor([[i] for i in range(len(systems))])
    ...             energy_block = TensorBlock(
    ...                 values=energies,
    ...                 samples=Labels(["system"], systems_idx.to(torch.int32)),
    ...                 components=torch.jit.annotate(List[Labels], []),
    ...                 properties=Labels(["energy"], torch.tensor([[0]])),
    ...             )
    ...
    ...             results["energy"] = TensorMap(
    ...                 keys=Labels(["_"], torch.tensor([[0]])),
    ...                 blocks=[energy_block],
    ...             )
    ...
    ...         return results
    ...

    Wrapping and exporting this model would then look like this:

    >>> import os
    >>> import tempfile
    >>> from metatensor.torch.atomistic import MetatensorAtomisticModel
    >>> from metatensor.torch.atomistic import (
    ...     ModelCapabilities,
    ...     ModelOutput,
    ...     ModelMetadata,
    ... )
    >>> model = ConstantEnergy(constant=3.141592)
    >>> # put the model in inference mode
    >>> model = model.eval()
    >>> # Define the model capabilities
    >>> capabilities = ModelCapabilities(
    ...     outputs={
    ...         "energy": ModelOutput(
    ...             quantity="energy",
    ...             unit="eV",
    ...             per_atom=False,
    ...             explicit_gradients=[],
    ...         ),
    ...     },
    ...     atomic_types=[1, 2, 6, 8, 12],
    ...     interaction_range=0.0,
    ...     length_unit="angstrom",
    ...     supported_devices=["cpu"],
    ...     dtype="float64",
    ... )
    >>> # define metadata about this model
    >>> metadata = ModelMetadata(
    ...     name="model-name",
    ...     authors=["Some Author", "Another One"],
    ...     # references and long description can also be added
    ... )
    >>> # wrap the model
    >>> wrapped = MetatensorAtomisticModel(model, metadata, capabilities)
    >>> # save the wrapped model to disk
    >>> with tempfile.TemporaryDirectory() as directory:
    ...     wrapped.save(os.path.join(directory, "constant-energy-model.pt"))
    ...

    .. py:attribute:: module
        :type: ModelInterface

    The torch module wrapped by this :py:class:`MetatensorAtomisticModel`.

    Reading from this attribute is safe, but modifying it is not recommended,
    unless you are familiar with the implementation of the model.
    """

    # Some annotation to make the TorchScript compiler happy
    _requested_neighbor_lists: List[NeighborListOptions]

    def __init__(
        self,
        module: ModelInterface,
        metadata: ModelMetadata,
        capabilities: ModelCapabilities,
    ):
        """
        :param module: The torch module to wrap and export.
        :param capabilities: Description of the model capabilities.
        """
        super().__init__()

        if not isinstance(module, torch.nn.Module):
            raise TypeError(f"`module` should be a torch.nn.Module, not {type(module)}")

        if isinstance(module, torch.jit.RecursiveScriptModule):
            raise TypeError("module should not already be a ScriptModule")

        if module.training:
            raise ValueError("module should not be in training mode")

        _check_annotation(module)
        self.module = module

        # ============================================================================ #

        # recursively explore `module` to get all the requested_neighbor_lists
        self._requested_neighbor_lists = []
        _get_requested_neighbor_lists(
            self.module,
            self.module.__class__.__name__,
            self._requested_neighbor_lists,
            capabilities.length_unit,
        )
        # ============================================================================ #

        self._metadata = metadata
        self._capabilities = capabilities

        # check that some required capabilities are set
        if capabilities.interaction_range < 0:
            raise ValueError(
                "`capabilities.interaction_range` was not set, "
                "but it is required to run simulations"
            )

        if math.isnan(capabilities.interaction_range):
            raise ValueError(
                "`capabilities.interaction_range` should be a "
                "float between 0 and infinity"
            )

        if len(capabilities.supported_devices) == 0:
            raise ValueError(
                "`capabilities.supported_devices` was not set, "
                "but it is required to run simulations."
            )

        if capabilities.dtype == "":
            raise ValueError(
                "`capabilities.dtype` was not set, "
                "but it is required to run simulations."
            )

        if capabilities.dtype == "float32":
            self._model_dtype = torch.float32
        elif capabilities.dtype == "float64":
            self._model_dtype = torch.float64
        else:
            raise ValueError(f"unknown dtype in capabilities: {capabilities.dtype}")

    @torch.jit.export
    def capabilities(self) -> ModelCapabilities:
        """Get the capabilities of the wrapped model"""
        return self._capabilities

    @torch.jit.export
    def metadata(self) -> ModelMetadata:
        """Get the metadata of the wrapped model"""
        return self._metadata

    @torch.jit.export
    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        """
        Get the neighbors lists required by the wrapped model or any of the child
        module.
        """
        return self._requested_neighbor_lists

    def forward(
        self,
        systems: List[System],
        options: ModelEvaluationOptions,
        check_consistency: bool,
    ) -> Dict[str, TensorMap]:
        """Run the wrapped model and return the corresponding outputs.

        Before running the model, this will convert the ``systems`` data from the engine
        unit to the model unit, including all neighbors lists distances.

        After running the model, this will convert all the outputs from the model units
        to the engine units.

        :param systems: input systems on which we should run the model. The systems
            should already contain all neighbors lists corresponding to the options in
            :py:meth:`requested_neighbor_lists()`.
        :param options: options for this run of the model
        :param check_consistency: Should we run additional check that everything is
            consistent? This should be set to ``True`` when verifying a model, and to
            ``False`` once you are sure everything is running fine.

        :return: A dictionary containing all the model outputs
        """

        if check_consistency:
            with record_function("MetatensorAtomisticModel::check_inputs"):
                _check_inputs(
                    capabilities=self._capabilities,
                    requested_neighbor_lists=self._requested_neighbor_lists,
                    systems=systems,
                    options=options,
                    expected_dtype=self._model_dtype,
                )

        # convert systems from engine to model units
        with record_function("MetatensorAtomisticModel::convert_units_input"):
            if self._capabilities.length_unit != options.length_unit:
                conversion = unit_conversion_factor(
                    quantity="length",
                    from_unit=options.length_unit,
                    to_unit=self._capabilities.length_unit,
                )

                systems = _convert_systems_units(
                    systems,
                    conversion,
                    model_length_unit=self._capabilities.length_unit,
                    system_length_unit=options.length_unit,
                )

        # run the actual calculations
        with record_function("Model::forward"):
            outputs = self.module(
                systems=systems,
                outputs=options.outputs,
                selected_atoms=options.selected_atoms,
            )

        if check_consistency:
            with record_function("MetatensorAtomisticModel::check_outputs"):
                _check_outputs(
                    systems=systems,
                    requested=options.outputs,
                    selected_atoms=options.selected_atoms,
                    outputs=outputs,
                    expected_dtype=self._model_dtype,
                )

        # convert outputs from model to engine units
        with record_function("MetatensorAtomisticModel::convert_units_output"):
            for name, output in outputs.items():
                declared = self._capabilities.outputs[name]
                requested = options.outputs[name]
                if declared.quantity == "" or requested.quantity == "":
                    continue

                if declared.quantity != requested.quantity:
                    raise ValueError(
                        f"model produces values as '{declared.quantity}' for the "
                        f"'{name}' output, but the engine requested "
                        f"'{requested.quantity}'"
                    )

                conversion = unit_conversion_factor(
                    quantity=declared.quantity,
                    from_unit=declared.unit,
                    to_unit=requested.unit,
                )

                if conversion != 1.0:
                    for block in output.blocks():
                        block.values[:] *= conversion
                        for _, gradient in block.gradients():
                            gradient.values[:] *= conversion

        return outputs

    def export(self, file: str, collect_extensions: Optional[str] = None):
        """Export this model to a file that can then be loaded by simulation engine.

        .. warning::
            :py:meth:`export` is deprecated. Use :py:meth:`save` instead.

        :param file: where to save the model. This can be a path or a file-like object.
        :param collect_extensions: if not None, all currently loaded PyTorch extension
            will be collected in this directory. If this directory already exists, it
            is removed and re-created.
        """
        warnings.warn(
            message="`export()` is deprecated, use `save()` instead",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return self.save(file, collect_extensions)

    def save(self, file: Union[str, Path], collect_extensions: Optional[str] = None):
        """Save this model to a file that can then be loaded by simulation engine.

        :param file: where to save the model. This can be a path or a file-like object.
        :param collect_extensions: if not None, all currently loaded PyTorch extension
            will be collected in this directory. If this directory already exists, it
            is removed and re-created.
        """
        module = self.eval()
        if os.environ.get("PYTORCH_JIT") == "0":
            raise RuntimeError(
                "found PYTORCH_JIT=0 in the environment, "
                "we can not save models without TorchScript"
            )

        try:
            module = torch.jit.script(module)
        except RuntimeError as e:
            raise RuntimeError("could not convert the module to TorchScript") from e

        # TODO: can we freeze these?
        # module = torch.jit.freeze(module)

        # Metadata about where and when the model was exported
        export_metadata = {
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "platform": platform.machine() + "-" + platform.system(),
            # TODO: user/hostname?
        }

        if collect_extensions is not None:
            export_metadata["extensions_directory"] = str(collect_extensions)

        extensions, deps = _collect_extensions(extensions_path=collect_extensions)

        torch.jit.save(
            module.to("cpu"),  # this allows to torch.jit.load without devices
            str(file),
            _extra_files={
                "torch-version": torch.__version__,
                "metatensor-version": metatensor_version,
                "extensions": json.dumps(extensions),
                "extensions-deps": json.dumps(deps),
                "export-metadata": json.dumps(export_metadata),
                "model-metadata": self._metadata.__getstate__()[0],
            },
        )


def _get_requested_neighbor_lists(
    module: torch.nn.Module,
    module_name: str,
    requested: List[NeighborListOptions],
    length_unit: str,
):
    if hasattr(module, "requested_neighbor_lists"):
        for new_options in module.requested_neighbor_lists():
            new_options.add_requestor(module_name)

            already_requested = False
            for existing in requested:
                if existing == new_options:
                    already_requested = True
                    for requestor in new_options.requestors():
                        existing.add_requestor(requestor)

            if not already_requested:
                if new_options.length_unit not in ["", length_unit]:
                    raise ValueError(
                        f"NeighborsListOptions from {module_name} already have a "
                        f"length unit ('{new_options.length_unit}') which does not "
                        f"match the model length units ('{length_unit}')"
                    )

                new_options.length_unit = length_unit
                requested.append(new_options)

    for child_name, child in module.named_children():
        _get_requested_neighbor_lists(
            module=child,
            module_name=module_name + "." + child_name,
            requested=requested,
            length_unit=length_unit,
        )


def _check_annotation(module: torch.nn.Module):
    # check annotations on forward
    annotations = module.forward.__annotations__
    expected_arguments = [
        "systems",
        "outputs",
        "selected_atoms",
        "return",
    ]

    expected_signature = (
        "`forward(self, "
        "systems: List[System], "
        "outputs: Dict[str, ModelOutput], "
        "selected_atoms: Optional[Labels]"
        ") -> Dict[str, TensorMap]`"
    )

    if list(annotations.keys()) != expected_arguments:
        raise TypeError(
            "`module.forward()` takes unexpected arguments, expected signature is "
            + expected_signature
        )

    if annotations["systems"] != List[System]:
        raise TypeError(
            "`systems` argument must be a list of metatensor atomistic `System`, "
            f"not {annotations['system']}"
        )

    if annotations["outputs"] != Dict[str, ModelOutput]:
        raise TypeError(
            "`outputs` argument must be `Dict[str, ModelOutput]`, "
            f"not {annotations['outputs']}"
        )

    if annotations["selected_atoms"] != Optional[Labels]:
        raise TypeError(
            "`selected_atoms` argument must be `Optional[Labels]`, "
            f"not {annotations['selected_atoms']}"
        )

    if annotations["return"] != Dict[str, TensorMap]:
        raise TypeError(
            "`forward()` must return a `Dict[str, TensorMap]`, "
            f"not {annotations['return']}"
        )


def _check_inputs(
    capabilities: ModelCapabilities,
    requested_neighbor_lists: List[NeighborListOptions],
    systems: List[System],
    options: ModelEvaluationOptions,
    expected_dtype: torch.dtype,
):
    if len(systems) == 0:
        return

    global_device = systems[0].device
    global_dtype = systems[0].positions.dtype

    if global_dtype != expected_dtype:
        raise ValueError(
            f"wrong dtype for the data: the model wants {dtype_name(expected_dtype)}, "
            f"we got {dtype_name(global_dtype)}"
        )

    # check that the requested outputs match what the model can do
    for name, requested in options.outputs.items():
        if name not in capabilities.outputs:
            raise ValueError(
                f"this model can not compute '{name}', the implemented "
                f"outputs are {capabilities.outputs.keys()}"
            )

        possible = capabilities.outputs[name]

        for parameter in requested.explicit_gradients:
            if parameter not in possible.explicit_gradients:
                raise ValueError(
                    f"this model can not compute explicit gradients of '{name}' "
                    f"with respect to '{parameter}'"
                )

        if requested.per_atom and not possible.per_atom:
            raise ValueError(
                f"this model can not compute '{name}' per atom, only globally"
            )

    selected_atoms = options.selected_atoms
    if selected_atoms is not None:
        if selected_atoms.device != global_device:
            raise ValueError(
                "expected all selected_atoms to be on the same device as the systems, "
                f"got {selected_atoms.device} and {global_device}"
            )

        if selected_atoms.names != ["system", "atom"]:
            raise ValueError(
                "invalid names for selected_atoms: expected "
                f"['system', 'atom'], got {selected_atoms.names}"
            )

        possible_atoms_values: List[List[int]] = []
        for s, system in enumerate(systems):
            for a in range(len(system)):
                possible_atoms_values.append([s, a])

        possible_atoms = Labels(
            ["system", "atom"],
            torch.tensor(possible_atoms_values, device=global_device),
        )

        intersection = selected_atoms.intersection(possible_atoms)
        if len(intersection) != len(selected_atoms):
            raise ValueError(
                "invalid selected_atoms: there are entries that are not "
                "possible for the current systems"
            )

    for system in systems:
        if system.device != global_device:
            raise ValueError(
                "expected all systems to be on the same device, "
                f"got {global_device} and {system.device}"
            )

        if not system.positions.dtype == global_dtype:
            raise ValueError(
                "expected all systems to have the same dtype, "
                f"got {global_dtype} and {system.positions.dtype}"
            )

        # check that the atomic types of the system match the one the model supports
        all_types = torch.unique(system.types)
        for atom_type in all_types:
            if atom_type not in capabilities.atomic_types:
                raise ValueError(
                    f"this model can not run for the atomic type '{atom_type.item()}'"
                )

        # Check neighbors lists
        known_neighbor_lists = system.known_neighbor_lists()
        for request in requested_neighbor_lists:
            found = False
            for known in known_neighbor_lists:
                if request == known:
                    found = True

            if not found:
                raise ValueError(
                    "missing neighbors list in the system: the model requested "
                    f"a list for {request}, but it was not computed and stored "
                    "in the system"
                )


def _convert_systems_units(
    systems: List[System],
    conversion: float,
    model_length_unit: str,
    system_length_unit: str,
) -> List[System]:
    if conversion == 1.0:
        return systems

    new_systems: List[System] = []
    for system in systems:
        new_system = System(
            types=system.types,
            positions=conversion * system.positions,
            cell=conversion * system.cell,
        )

        # also update the neighbors list distances
        for request in system.known_neighbor_lists():
            neighbors = system.get_neighbor_list(request)
            new_system.add_neighbor_list(
                request,
                TensorBlock(
                    values=conversion * neighbors.values,
                    samples=neighbors.samples,
                    components=neighbors.components,
                    properties=neighbors.properties,
                ),
            )

        known_data = system.known_data()
        if len(known_data) != 0:
            warnings.warn(
                "the model requires a different length unit "
                f"({model_length_unit}) than the system ({system_length_unit}), "
                f"but we don't know how to convert custom data ({known_data}) "
                "accordingly",
                stacklevel=2,
            )

        for data in known_data:
            new_system.add_data(data, system.get_data(data))

        new_systems.append(new_system)

    return new_systems
