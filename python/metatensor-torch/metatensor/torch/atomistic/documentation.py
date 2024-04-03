from typing import Dict, List, Optional

import torch

from ..documentation import Labels, TensorBlock


class System:
    """
    A System contains all the information about an atomistic system; and should be used
    as the input of metatensor atomistic models.
    """

    def __init__(
        self,
        types: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
    ):
        """
        You can create a :py:class:`System` with ``types``, ``positions`` and ``cell``
        tensors, or convert data from other libraries.

        .. admonition:: Converting data to metatensor `System`

            We provide a way to convert :py:class:`ase.Atoms` instances to
            :py:class:`System` using the :py:func:`systems_to_torch()` function.

            In addition, some external packages provide ways to create
            :py:class:`System` using data from other libraries:

            - `rascaline`_ has the :py:func:`rascaline.torch.systems_to_torch()`
              function that can convert from ASE, chemfiles and PySCF.

            .. _rascaline: https://luthaf.fr/rascaline/latest/index.html

        :param types: 1D tensor of integer representing the particles identity. For
            atoms, this is typically their atomic numbers.

        :param positions: 2D tensor of shape (len(types), 3) containing the Cartesian
            positions of all particles in the system.

        :param cell: 2D tensor of shape (3, 3), describing the bounding box/unit cell of
            the system. Each row should be one of the bounding box vector; and columns
            should contain the x, y, and z components of these vectors (i.e. the cell
            should be given in row-major order). Systems are assumed to obey periodic
            boundary conditions, non-periodic systems should set the cell to 0.
        """

    def __len__(self) -> int:
        pass

    @property
    def types(self) -> torch.Tensor:
        """Tensor of 32-bit integers representing the particles identity"""

    @property
    def positions(self) -> torch.Tensor:
        """
        Tensor of floating point values containing the particles cartesian coordinates
        """

    @property
    def cell(self) -> torch.Tensor:
        """Tensor of floating point values containing bounding box/cell of the system"""

    @property
    def device(self) -> torch.device:
        """get the device of all the arrays stored inside this :py:class:`System`"""

    @property
    def dtype(self) -> torch.dtype:
        """
        get the dtype of all the arrays stored inside this :py:class:`System`

        .. warning::

            Due to limitations in TorchScript C++ extensions, the dtype is returned as
            an integer, which can not be compared with :py:class:`torch.dtype`
            instances. See :py:meth:`TensorBlock.dtype` for more information.
        """

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> "System":
        """
        Move all the arrays in this system to the given ``dtype`` and ``device``.

        :param dtype: new dtype to use for all arrays. The dtype stays the same if this
            is set to ``None``.
        :param device: new device to use for all arrays. The device stays the same if
            this is set to ``None``.
        """

    def add_neighbors_list(
        self,
        options: "NeighborsListOptions",
        neighbors: TensorBlock,
    ):
        """
        Add a new neighbors list in this system corresponding to the given ``options``.

        The neighbors list should have the following samples: ``"first_atom"``,
        ``"second_atom"``, ``"cell_shift_a"``, ``"cell_shift_b"``, ``"cell_shift_c"``,
        containing the index of the first and second atoms (matching the "atom" sample
        in the positions); and the number of cell vector a/b/c to add to the positions
        difference to get the pair vector.

        The neighbors should also have a single component ``"xyz"`` with values ``[0, 1,
        2]``; and a single property ``"distance"`` with value 0.

        The neighbors values must contain the distance vector from the first to the
        second atom, i.e. ``positions[second_atom] - positions[first_atom] +
        cell_shift_a * cell_a + cell_shift_b * cell_b + cell_shift_c * cell_c``.

        :param options: options of the neighbors list
        :param neighbors: list of neighbors stored in a :py:class:`TensorBlock`
        """

    def get_neighbors_list(
        self,
        options: "NeighborsListOptions",
    ) -> TensorBlock:
        """
        Retrieve a previously stored neighbors list with the given ``options``, or throw
        an error if no such neighbors list exists.

        :param options: options of the neighbors list to retrieve
        """

    def known_neighbors_lists(self) -> List["NeighborsListOptions"]:
        """
        Get all the neighbors lists options registered with this :py:class:`System`
        """

    def add_data(self, name: str, data: TensorBlock):
        """
        Add custom data to this system, stored as :py:class:`TensorBlock`.

        This is intended for experimentation with models that need more data as input,
        and moved into a field of ``System`` later.

        :param name: name of the custom data
        :param data: values of the custom data
        """

    def get_data(self, name: str) -> TensorBlock:
        """
        Retrieve custom data stored in this System with the given ``name``, or throw
        an error if no data can be found.

        :param name: name of the custom data to retrieve
        """

    def known_data(self) -> List[str]:
        """
        Get the name of all the custom data registered with this :py:class:`System`
        """


class NeighborsListOptions:
    """Options for the calculation of a neighbors list"""

    def __init__(self, cutoff: float, full_list: bool, requestor: str = ""):
        """
        :param cutoff: spherical cutoff radius for the neighbors list, in the
            model units
        :param full_list: should the list be a full or half neighbors list
        :param requestor: who requested this neighbors list, you can add additional
            requestors later using :py:meth:`add_requestor`
        """

    @property
    def cutoff(self) -> float:
        """Spherical cutoff radius for this neighbors list in model units"""

    @property
    def length_unit(self) -> str:
        """
        The unit of length used for the cutoff.

        This is typically set by :py:class:`MetatensorAtomisticModel` when collecting
        all neighbors list requests.

        The list of possible units is available :ref:`here <known-quantities-units>`.
        """

    def engine_cutoff(self, engine_length_unit: str) -> float:
        """
        Spherical cutoff radius for this neighbors list in engine units.

        The engine must provide the unit it uses for lengths, and the cutoff will
        automatically be converted.
        """

    @property
    def full_list(self) -> bool:
        """
        Should the list be a full neighbors list (contains both the pair ``i->j`` and
        ``j->i``) or a half neighbors list (contains only the pair ``i->j``)
        """

    def requestors(self) -> List[str]:
        """Get the list of modules requesting this neighbors list"""

    def add_requestor(self, requestor: str):
        """
        Add another ``requestor`` to the list of modules requesting this neighbors list
        """

    def __repr__(self) -> str:
        pass

    def __str__(self) -> str:
        pass

    def __eq__(self, other: "NeighborsListOptions") -> bool:
        pass

    def __ne__(self, other: "NeighborsListOptions") -> bool:
        pass


class ModelOutput:
    """Description of one of the quantity a model can compute."""

    def __init__(
        self,
        quantity: str = "",
        unit: str = "",
        per_atom: bool = False,
        explicit_gradients: List[str] = [],  # noqa B006
    ):
        pass

    @property
    def quantity(self) -> str:
        """
        Quantity of the output (e.g. energy, dipole, …).  If this is an empty string, no
        unit conversion will be performed.

        The list of possible quantities is available :ref:`here
        <known-quantities-units>`.
        """

    @property
    def unit(self) -> str:
        """
        Unit of the output. If this is an empty string, no unit conversion will be
        performed.

        The list of possible units is available :ref:`here <known-quantities-units>`.
        """

    per_atom: bool
    """Is the output defined per-atom or for the overall structure"""

    explicit_gradients: List[str]
    """
    Which gradients should be computed eagerly and stored inside the output
    :py:class:`TensorMap`.
    """


class ModelCapabilities:
    """Description of a model capabilities, i.e. everything a model can do."""

    def __init__(
        self,
        outputs: Dict[str, ModelOutput] = {},  # noqa B006
        atomic_types: List[int] = [],  # noqa B006
        interaction_range: float = -1,
        length_unit: str = "",
        supported_devices: List[str] = [],  # noqa B006
        dtype: str = "",
    ):
        pass

    outputs: Dict[str, ModelOutput]
    """
    All possible outputs from this model and corresponding settings.

    During a specific run, a model might be asked to only compute a subset of these
    outputs.
    """

    atomic_types: List[int]
    """which atomic types the model can handle"""

    interaction_range: float
    """
    How far a given atom needs to know about other atoms, in the length unit of the
    model.

    For a short range model, this is the same as the largest neighbors list cutoff. For
    a message passing model, this is the cutoff of one environment times the number of
    message passing steps. For an explicit long range model, this should be set to
    infinity (``float("inf")``/``math.inf``/``torch.inf`` in Python).
    """

    @property
    def length_unit() -> str:
        """
        Unit used by the model for its inputs.

        This applies to the ``interaction_range``, any cutoff in neighbors lists, the
        atoms positions and the system cell.

        The list of possible units is available :ref:`here <known-quantities-units>`.
        """

    @property
    def dtype() -> str:
        """
        The dtype of this model

        This can be ``"float32"`` or ``"float64"``, and must be used by the engine as
        the dtype of all inputs and outputs for this model.
        """

    def engine_interaction_range(self, engine_length_unit: str) -> float:
        """
        Same as :py:attr:`interaction_range`, but in the unit of length used by the
        engine.
        """

    supported_devices: List[str]
    """
    What devices can this model run on? This should only contain the ``device_type``
    part of the device, and not the device number (i.e. this should be ``"cuda"``, not
    ``"cuda:0"``).

    Devices should be ordered in order of preference: the first entry in this list
    should be the best device for this model, and so on.
    """


class ModelEvaluationOptions:
    """
    Options requested by the simulation engine/evaluation code when doing a single model
    evaluation.
    """

    def __init__(
        self,
        length_unit: str = "",
        outputs: Dict[str, ModelOutput] = {},  # noqa B006
        selected_atoms: Optional[Labels] = None,
    ):
        pass

    @property
    def length_unit(self) -> str:
        """
        Unit of lengths the engine uses for the model input.

        The list of possible units is available :ref:`here <known-quantities-units>`.
        """

    outputs: Dict[str, ModelOutput]
    """requested outputs for this run and corresponding settings"""

    @property
    def selected_atoms() -> Optional[Labels]:
        """
        Only run the calculation for a selected subset of atoms.

        If this is set to ``None``, run the calculation on all atoms. If this is a set
        of :py:class:`metatensor.torch.Labels`, it will have two dimensions named
        ``"system"`` and ``"atom"``, containing the 0-based indices of all the atoms in
        the selected subset.
        """


class ModelMetadata:
    """Metadata about a specific exported model"""

    def __init__(
        self,
        name: str = "",
        description: str = "",
        authors: List[str] = [],  # noqa: B006
        references: Dict[str, List[str]] = {},  # noqa: B006
    ):
        pass

    name: str
    """Name of this model"""

    description: str
    """Description of this model"""

    authors: List[str]
    """List of authors for this model"""

    references: Dict[str, List[str]]
    """
    Academic references for this model. The top level dict can have three keys:

    - "implementation": for reference to software used in the implementation
      of the model
    - "architecture": for reference that introduced the general architecture
      used by this model
    - "model": for reference specific to this exact model
    """

    def print(self) -> str:
        """
        Format the model metadata into a string. This is the same format used for
        ``__str__`` and ``__repr__``.
        """


def check_atomistic_model(path: str):
    """
    Check that the file at ``path`` contains an exported metatensor atomistic model, and
    that this model can be loaded in the current process.

    This function should be called before :py:func:`torch.jit.load()` when loading an
    existing model.

    :param path: path to the exported model file
    """


def register_autograd_neighbors(
    system: System, neighbors: TensorBlock, check_consistency: bool
):
    """
    Register a new autograd node going from (``system.positions``, ``system.cell``) to
    the ``neighbors`` distance vectors.

    This does not recompute the distance vectors, but work as-if all the data in
    ``neighbors.values`` was computed directly from ``system.positions`` and
    ``system.cell``, allowing downstream models to use it directly with full autograd
    integration.

    :param system: system containing the positions and cell used to compute the
        neighbors list
    :param system: neighbors list, following the same format as
        :py:meth:`System.add_neighbors_list`
    :param check_consistency: can be set to ``True`` to run a handful of additional
        checks in case the data in neighbors does not follow what's expected.
    """


def unit_conversion_factor(quantity: str, from_unit: str, to_unit: str):
    """
    Get the multiplicative conversion factor from ``from_unit`` to ``to_unit``. Both
    units must be valid and known for the given physical ``quantity``. The set of valid
    quantities and units is available :ref:`here <known-quantities-units>`.

    :param quantity: name of the physical quantity
    :param from_unit: current unit of the data
    :param to_unit: target unit of the data
    """
