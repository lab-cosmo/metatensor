import copy
import ctypes
import warnings
from typing import Generator, List, Sequence, Tuple

from . import data
from ._c_api import c_uintptr_t, mts_array_t, mts_block_t, mts_labels_t
from ._c_lib import _get_library
from .data import (
    Array,
    ArrayWrapper,
    Device,
    DeviceWarning,
    DType,
    mts_array_to_python_array,
)
from .labels import Labels
from .status import _check_pointer
from .utils import _to_arguments_parse


class TensorBlock:
    """
    Basic building block for a :py:class:`TensorMap`.

    A single block contains a n-dimensional :py:class:`metatensor.data.Array`,
    and n sets of :py:class:`Labels` (one for each dimension). The first
    dimension is the *samples* dimension, the last dimension is the *properties*
    dimension. Any intermediate dimension is called a *component* dimension.

    Samples should be used to describe *what* we are representing, while
    properties should contain information about *how* we are representing it.
    Finally, components should be used to describe vectorial or tensorial
    components of the data.

    A block can also contain gradients of the values with respect to a variety
    of parameters. In this case, each gradient is a :py:class:`TensorBlock` with
    a separate set of samples and possibly components, but which shares the same
    property labels as the original :py:class:`TensorBlock`.

    >>> import numpy as np
    >>> block = TensorBlock(
    ...     values=np.array(
    ...         [
    ...             [1, 2, 4],
    ...             [3, 5, 6],
    ...         ]
    ...     ),
    ...     samples=Labels("samples", np.array([[4], [2]])),
    ...     components=[],
    ...     properties=Labels("properties", np.array([[0], [1], [2]])),
    ... )
    >>> block
    TensorBlock
        samples (2): ['samples']
        components (): []
        properties (3): ['properties']
        gradients: None
    >>> block.samples
    Labels(
        samples
           4
           2
    )
    >>> block.values[block.samples.position([2])]
    array([3, 5, 6])
    """

    def __init__(
        self,
        values: Array,
        samples: Labels,
        components: Sequence[Labels],
        properties: Labels,
    ):
        """
        :param values: array containing the values for this block
        :param samples: labels describing the samples (first dimension of the array)
        :param components: list of labels describing the components (intermediate
            dimensions of the array). This should be an empty list for scalar/invariant
            data.
        :param properties: labels describing the properties (last dimension of the
            array)
        """
        self._lib = _get_library()
        self._parent = None
        self._gradient_parameters = []

        if not isinstance(samples, Labels):
            raise TypeError(f"`samples` must be metatensor Labels, not {type(samples)}")

        components = list(components)
        for component in components:
            if not isinstance(component, Labels):
                raise TypeError(
                    "`components` elements must be metatensor Labels, "
                    f"not {type(component)}"
                )

        if not isinstance(properties, Labels):
            raise TypeError(
                f"`properties` must be metatensor Labels, not {type(properties)}"
            )

        components_array = ctypes.ARRAY(mts_labels_t, len(components))()
        for i, component in enumerate(components):
            components_array[i] = component._as_mts_labels_t()

        values = ArrayWrapper(values)

        self._actual_ptr = self._lib.mts_block(
            values.into_mts_array(),
            samples._as_mts_labels_t(),
            components_array,
            len(components_array),
            properties._as_mts_labels_t(),
        )
        _check_pointer(self._actual_ptr)

        if not data.array_device_is_cpu(self.values):
            warnings.warn(
                "Values and labels for this block are on different devices: "
                f"labels are always on CPU, and values are on device '{self.device}'. "
                "If you are using PyTorch and need the labels to also be on "
                f"{self.device}, you should use `metatensor.torch.TensorBlock`.",
                category=DeviceWarning,
                stacklevel=2,
            )

    @staticmethod
    def _from_ptr(ptr, parent):
        """
        create a block from a pointer, either owning its data (new block as a
        copy of an existing one) or not (block inside a :py:class:`TensorMap`)
        """
        _check_pointer(ptr)
        obj = TensorBlock.__new__(TensorBlock)
        obj._lib = _get_library()
        obj._gradient_parameters = []
        obj._actual_ptr = ptr
        # keep a reference to the parent object (usually a TensorMap) to
        # prevent it from being garbage-collected & removing this block
        obj._parent = parent
        return obj

    @property
    def _ptr(self):
        if self._actual_ptr is None:
            raise ValueError(
                "this block has been moved inside a TensorMap/another TensorBlock "
                "and can no longer be used"
            )

        return self._actual_ptr

    def _move_ptr(self):
        assert self._parent is None
        self._actual_ptr = None

    def __del__(self):
        if (
            hasattr(self, "_lib")
            and self._lib is not None
            and hasattr(self, "_actual_ptr")
            and hasattr(self, "_parent")
        ):
            if self._parent is None:
                self._lib.mts_block_free(self._actual_ptr)

    def __copy__(self):
        raise ValueError(
            "shallow copies of TensorBlock are not possible, use a deepcopy instead"
        )

    def __deepcopy__(self, _memodict):
        new_ptr = self._lib.mts_block_copy(self._ptr)
        return TensorBlock._from_ptr(new_ptr, parent=None)

    def __reduce__(self):
        raise NotImplementedError(
            "Pickling for is not implemented for TensorBlocks, wrap the block in a "
            "TensorMap first"
        )

    def copy(self) -> "TensorBlock":
        """
        get a deep copy of this block, including all the data and metadata
        """
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        if len(self._gradient_parameters) != 0:
            s = f"Gradient TensorBlock ('{'/'.join(self._gradient_parameters)}')\n"
        else:
            s = "TensorBlock\n"
        s += f"    samples ({len(self.samples)}): {str(list(self.samples.names))}"
        s += "\n"
        s += "    components ("
        s += ", ".join([str(len(c)) for c in self.components])
        s += "): ["
        for ic in self.components:
            for name in ic.names[:]:
                s += "'" + name + "', "
        if len(self.components) > 0:
            s = s[:-2]
        s += "]\n"
        s += f"    properties ({len(self.properties)}): "
        s += f"{str(list(self.properties.names))}\n"
        s += "    gradients: "
        if len(self.gradients_list()) > 0:
            s += f"{str(list(self.gradients_list()))}"
        else:
            s += "None"

        return s

    def __eq__(self, other):
        from metatensor.operations import equal_block

        return equal_block(self, other)

    def __ne__(self, other):
        from metatensor.operations import equal_block

        return not equal_block(self, other)

    @property
    def _raw_values(self) -> mts_array_t:
        """Get the raw ``mts_array_t`` corresponding to this block's values"""
        data = mts_array_t()
        self._lib.mts_block_data(self._ptr, data)
        return data

    @property
    def values(self) -> Array:
        """
        Get the values for this block.

        The array type depends on how the block was created. Currently, numpy
        ``ndarray`` and torch ``Tensor`` are supported.
        """

        return mts_array_to_python_array(self._raw_values, parent=self)

    @property
    def samples(self) -> Labels:
        """
        Get the sample :py:class:`Labels` for this block.

        The entries in these labels describe the first dimension of the
        ``values`` array.
        """
        return self._labels(0)

    @property
    def components(self) -> List[Labels]:
        """
        Get the component :py:class:`Labels` for this block.

        The entries in these labels describe intermediate dimensions of the
        ``values`` array.
        """
        n_components = len(self.values.shape) - 2

        result = []
        for axis in range(n_components):
            result.append(self._labels(axis + 1))

        return result

    @property
    def properties(self) -> Labels:
        """
        Get the property :py:class:`Labels` for this block.

        The entries in these labels describe the last dimension of the
        ``values`` array. The properties are guaranteed to be the same for
        values and gradients in the same block.
        """
        property_axis = len(self.values.shape) - 1
        return self._labels(property_axis)

    def _labels(self, axis) -> Labels:
        result = mts_labels_t()
        self._lib.mts_block_labels(self._ptr, axis, result)
        return Labels._from_mts_labels_t(result)

    def gradient(self, parameter: str) -> "TensorBlock":
        """
        Get the gradient of the block ``values``  with respect to the given
        ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)

        >>> import numpy as np
        >>> from metatensor import Labels, TensorBlock
        >>> block = TensorBlock(
        ...     values=np.full((3, 1, 5), 1.0),
        ...     samples=Labels(["system"], np.array([[0], [2], [4]])),
        ...     components=[Labels.range("component", 1)],
        ...     properties=Labels.range("property", 5),
        ... )
        >>> positions_gradient = TensorBlock(
        ...     values=np.full((2, 3, 1, 5), 11.0),
        ...     samples=Labels(["sample", "atom"], np.array([[0, 2], [2, 3]])),
        ...     components=[
        ...         Labels.range("direction", 3),
        ...         Labels.range("component", 1),
        ...     ],
        ...     properties=Labels.range("property", 5),
        ... )

        >>> block.add_gradient("positions", positions_gradient)
        >>> cell_gradient = TensorBlock(
        ...     values=np.full((2, 3, 3, 1, 5), 15.0),
        ...     samples=Labels.range("sample", 2),
        ...     components=[
        ...         Labels.range("direction_1", 3),
        ...         Labels.range("direction_2", 3),
        ...         Labels.range("component", 1),
        ...     ],
        ...     properties=Labels.range("property", 5),
        ... )
        >>> block.add_gradient("cell", cell_gradient)

        >>> positions_gradient = block.gradient("positions")
        >>> print(positions_gradient)
        Gradient TensorBlock ('positions')
            samples (2): ['sample', 'atom']
            components (3, 1): ['direction', 'component']
            properties (5): ['property']
            gradients: None

        >>> cell_gradient = block.gradient("cell")
        >>> print(cell_gradient)
        Gradient TensorBlock ('cell')
            samples (2): ['sample']
            components (3, 3, 1): ['direction_1', 'direction_2', 'component']
            properties (5): ['property']
            gradients: None
        """
        gradient_block = ctypes.POINTER(mts_block_t)()
        self._lib.mts_block_gradient(
            self._ptr, parameter.encode("utf8"), gradient_block
        )

        gradient = TensorBlock._from_ptr(gradient_block, parent=self)

        gradient._gradient_parameters = copy.deepcopy(self._gradient_parameters)
        gradient._gradient_parameters.append(parameter)

        return gradient

    def add_gradient(self, parameter: str, gradient: "TensorBlock"):
        """
        Add gradient with respect to ``parameter`` in this block.

        :param parameter:
            add gradients with respect to this ``parameter`` (e.g. ``positions``,
            ``cell``, ...)

        :param gradient:
            a :py:class:`TensorBlock` whose values contain the gradients of this
            :py:class:`TensorBlock` values with respect to ``parameter``. The labels
            of the gradient :py:class:`TensorBlock` should be organized as follows:

            - its samples must contain ``"sample"`` as the first dimension, with values
              containing the index of the corresponding samples in this
              :py:class:`TensorBlock`, and arbitrary supplementary samples dimension;
            - its components must contain at least the same components as this
              :py:class:`TensorBlock`, with any additional components coming before
              those;
            - its properties must match exactly those of this :py:class:`TensorBlock`.

        >>> import numpy as np
        >>> from metatensor import Labels, TensorBlock
        >>> block = TensorBlock(
        ...     values=np.full((3, 1, 1), 1.0),
        ...     samples=Labels(["system"], np.array([[0], [2], [4]])),
        ...     components=[Labels.range("component", 1)],
        ...     properties=Labels.range("property", 1),
        ... )
        >>> gradient = TensorBlock(
        ...     values=np.full((2, 1, 1), 11.0),
        ...     samples=Labels(["sample", "parameter"], np.array([[0, -2], [2, 3]])),
        ...     components=[Labels.range("component", 1)],
        ...     properties=Labels.range("property", 1),
        ... )
        >>> block.add_gradient("parameter", gradient)
        >>> print(block)
        TensorBlock
            samples (3): ['system']
            components (1): ['component']
            properties (1): ['property']
            gradients: ['parameter']
        """
        if self._parent is not None:
            raise ValueError(
                "cannot add gradient on this block since it is a view inside "
                "a TensorMap or another TensorBlock"
            )

        if self.dtype != gradient.dtype:
            raise ValueError(
                "values and the new gradient must have the same dtype, "
                f"got {self.dtype} and {gradient.dtype}"
            )

        if self.device != gradient.device:
            raise ValueError(
                "values and the new gradient must be on the same device, "
                f"got {self.device} and {gradient.device}"
            )

        # mts_block_add_gradient already checks that all arrays have the same origin
        # (i.e. they are all numpy, or all torch, or ...), so we don't need to check it
        # again here.

        gradient_ptr = gradient._ptr

        # the gradient is moved inside this block, assign NULL to
        # `gradient._ptr` to prevent accessing invalid data from Python and
        # double free
        gradient._move_ptr()

        self._lib.mts_block_add_gradient(
            self._ptr, parameter.encode("utf8"), gradient_ptr
        )

    def gradients_list(self) -> List[str]:
        """get a list of all gradients defined in this block"""
        parameters = ctypes.POINTER(ctypes.c_char_p)()
        count = c_uintptr_t()
        self._lib.mts_block_gradients_list(self._ptr, parameters, count)

        result = []
        for i in range(count.value):
            result.append(parameters[i].decode("utf8"))

        return result

    def has_gradient(self, parameter: str) -> bool:
        """
        Check if this block contains gradient information with respect to the
        given ``parameter``.

        :param parameter: check for gradients with respect to this ``parameter``
            (e.g. ``positions``, ``cell``, ...)
        """
        return parameter in self.gradients_list()

    def gradients(self) -> Generator[Tuple[str, "TensorBlock"], None, None]:
        """Get an iterator over all gradients defined in this block."""
        for parameter in self.gradients_list():
            yield (parameter, self.gradient(parameter))

    @property
    def dtype(self) -> DType:
        """
        Get the dtype of all the values and gradient arrays stored inside this
        :py:class:`TensorBlock`.
        """
        return data.array_dtype(self.values)

    @property
    def device(self) -> Device:
        """
        Get the device of all the values and gradient arrays stored inside this
        :py:class:`TensorBlock`.
        """
        return data.array_device(self.values)

    def to(self, *args, **kwargs) -> "TensorBlock":
        """
        Move all the arrays in this block (values and gradients) to the given ``dtype``,
        ``device`` and ``arrays`` backend.

        :param dtype: new dtype to use for all arrays. The dtype stays the same if this
            is set to ``None``.
        :param device: new device to use for all arrays. The device stays the same if
            this is set to ``None``.
        :param arrays: new backend to use for the arrays. This can be either
            ``"numpy"``, ``"torch"`` or ``None`` (keeps the existing backend); and must
            be given as a keyword argument (``arrays="numpy"``).
        """
        arrays = kwargs.pop("arrays", None)
        dtype, device = _to_arguments_parse("`TensorBlock.to`", *args, **kwargs)

        values = self.values

        if arrays is not None:
            values = data.array_change_backend(values, arrays)

        if dtype is not None:
            values = data.array_change_dtype(values, dtype)

        if device is not None:
            values = data.array_change_device(values, device)

        block = TensorBlock(values, self.samples, self.components, self.properties)
        for parameter, gradient in self.gradients():
            block.add_gradient(
                parameter, gradient.to(dtype=dtype, device=device, arrays=arrays)
            )

        return block
