import ctypes
from typing import Any, NewType, Union

import numpy as np

from .._c_api import c_uintptr_t, mts_array_t, mts_data_origin_t
from ..status import _check_status
from ..utils import _call_with_growing_buffer, _ptr_to_ndarray
from .array import _object_from_ptr, _origin_numpy, _origin_pytorch, _register_origin


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    # This NewType is only used for typechecking and documentation purposes
    Array = NewType("Array", Union[np.ndarray, torch.Tensor])
else:
    Array = NewType("Array", np.ndarray)

Array.__doc__ = """
An ``Array`` contains the actual data stored in a
:py:class:`metatensor.TensorBlock`.

This data is manipulated by ``metatensor`` in a completely opaque way: this
library does not know what's inside the arrays appart from a small set of
constrains:

- the array contains numeric data (loading and saving
  :py:class:`metatensor.TensorMap` additionally assumes arrays of 64-bit IEEE-754
  floating points numbers);
- they are stored as row-major, n-dimensional arrays with at least 2 dimensions;
- it is possible to create new arrays and move data from one array to another.

The actual type of an ``Array`` depends on how the
:py:class:`metatensor.TensorBlock` was created. Currently,
:py:class:`numpy.ndarray` and :py:class:`torch.Tensor` are supported.
"""


_ADDITIONAL_ORIGINS = {}


def register_external_data_wrapper(origin, klass):
    """
    Register a non-Python data origin and the corresponding class wrapper.

    The wrapper class constructor must take two arguments (raw ``mts_array`` and
    python ``parent`` object) and return a subclass of either
    :py:class:`numpy.ndarray` or :py:class:`torch.Tensor`, which keeps
    ``parent`` alive. The :py:class:`metatensor.data.ExternalCpuArray` class
    should provide the right behavior for data living in CPU memory, and can
    serve as an example for more advanced custom arrays.

    :param origin: new origin to register as a string
    :param klass: wrapper class to use for this origin
    """

    if not isinstance(origin, str):
        raise ValueError(f"origin must be a string, got {type(origin)}")

    global _ADDITIONAL_ORIGINS
    _ADDITIONAL_ORIGINS[_register_origin(origin)] = klass


def mts_array_to_python_array(mts_array, parent=None):
    """Convert a raw mts_array to a Python ``Array``.

    Either the underlying array was allocated by Python, and the Python object
    is directly returned; or the underlying array was not allocated by Python,
    and additional origins are searched for a suitable Python wrapper class.
    """
    origin = data_origin(mts_array)
    if _is_python_origin(origin):
        return _object_from_ptr(mts_array.ptr).array
    elif origin in _ADDITIONAL_ORIGINS:
        return _ADDITIONAL_ORIGINS[origin](mts_array, parent=parent)
    else:
        raise ValueError(
            f"unable to handle data coming from '{data_origin_name(origin)}', "
            "you should maybe register a new array wrapper with metatensor"
        )


def mts_array_was_allocated_by_python(mts_array):
    """Check if a given mts_array was allocated by Python"""
    return _is_python_origin(data_origin(mts_array))


def _is_python_origin(origin):
    return origin in [_origin_numpy(), _origin_pytorch()]


def data_origin(mts_array):
    """Get the data origin of an mts_array"""
    origin = mts_data_origin_t()
    mts_array.origin(mts_array.ptr, origin)
    return origin.value


def data_origin_name(origin):
    """Get the name of the data origin of an mts_array"""
    from .._c_lib import _get_library

    lib = _get_library()

    return _call_with_growing_buffer(
        lambda buffer, bufflen: lib.mts_get_data_origin(origin, buffer, bufflen)
    )


# ============================================================================ #


class ExternalCpuArray(np.ndarray):
    """
    Small wrapper class around ``np.ndarray``, adding a reference to a parent
    Python object that actually owns the memory used inside the array. This
    prevents the parent from being garbage collected while the ndarray is still
    alive, thus preventing use-after-free.

    This is intended to be used to wrap Rust-owned memory inside a numpy's
    ndarray, while making sure the memory owner is kept around for long enough.
    """

    def __new__(cls, mts_array: mts_array_t, parent: Any):
        """
        :param mts_array: raw array to wrap in a Python-compatible class
        :param parent: owner of the raw array, we will keep a reference to this
            python object
        """

        shape_ptr = ctypes.POINTER(c_uintptr_t)()
        shape_count = c_uintptr_t()
        status = mts_array.shape(mts_array.ptr, shape_ptr, shape_count)
        _check_status(status)

        shape = []
        for i in range(shape_count.value):
            shape.append(shape_ptr[i])

        data = ctypes.POINTER(ctypes.c_double)()
        status = mts_array.data(mts_array.ptr, data)
        _check_status(status)

        array = _ptr_to_ndarray(data, shape, np.float64)
        obj = array.view(cls)

        # keep a reference to the parent object (if any) to prevent it from
        # being garbage-collected too early.
        obj._parent = parent

        return obj

    def __array_finalize__(self, obj):
        # keep the parent around when creating sub-views of this array
        self._parent = getattr(obj, "_parent", None)

    def __array_wrap__(self, new, context=None, return_scalar=False):
        self_ptr = self.ctypes.data
        self_size = self.nbytes

        new_ptr = new.ctypes.data

        if self_ptr <= new_ptr <= self_ptr + self_size:
            # if the new array is a view inside memory owned by self, wrap it in
            # a ExternalCpuArray
            return super().__array_wrap__(new)
        else:
            # return the ndarray straight away
            return np.asarray(new)
