import ctypes
from typing import Union

import numpy as np

from .._c_api import c_uintptr_t, mts_array_t, mts_data_origin_t
from ..utils import catch_exceptions


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _register_origin(name):
    from .._c_lib import _get_library

    lib = _get_library()
    origin = mts_data_origin_t(0)
    lib.mts_register_data_origin(name.encode("utf8"), origin)
    return origin.value


def _is_numpy_array(array):
    return isinstance(array, np.ndarray)


def _is_torch_array(array):
    if not HAS_TORCH:
        return False

    return isinstance(array, torch.Tensor)


_NUMPY_STORAGE_ORIGIN = None
_TORCH_STORAGE_ORIGIN = None


def _origin_numpy():
    global _NUMPY_STORAGE_ORIGIN
    if _NUMPY_STORAGE_ORIGIN is None:
        _NUMPY_STORAGE_ORIGIN = _register_origin(__name__ + ".numpy")

    return _NUMPY_STORAGE_ORIGIN


def _origin_pytorch():
    global _TORCH_STORAGE_ORIGIN
    if _TORCH_STORAGE_ORIGIN is None:
        _TORCH_STORAGE_ORIGIN = _register_origin(__name__ + ".torch")

    return _TORCH_STORAGE_ORIGIN


if HAS_TORCH:
    torch_dtype = torch.dtype
    torch_device = torch.device
else:

    class torch_dtype:
        pass

    class torch_device:
        pass


DType = Union[np.dtype, torch_dtype]
"""Type representing a dtype in either numpy or torch"""

Device = Union[str, torch_device]
"""Type representing a device in either numpy or torch"""


def array_dtype(array) -> DType:
    """Get the dtype of an array"""
    if _is_numpy_array(array) or _is_torch_array(array):
        return array.dtype
    else:
        raise TypeError(f"unknown array type: {type(array)}")


def array_change_dtype(array, dtype: DType):
    """Change the dtype of an array"""
    if _is_numpy_array(array):
        return array.astype(dtype)
    elif _is_torch_array(array):
        return array.to(dtype=dtype)
    else:
        raise TypeError(f"unknown array type: {type(array)}")


def array_device(array) -> Device:
    """Get the device of an array"""
    if _is_numpy_array(array):
        return "cpu"
    elif _is_torch_array(array):
        return array.device
    else:
        raise TypeError(f"unknown array type: {type(array)}")


def array_device_is_cpu(array) -> bool:
    """Check if the device of an array is CPU"""
    if _is_numpy_array(array):
        return True
    elif _is_torch_array(array):
        return array.device.type == torch.device("cpu").type
    else:
        raise TypeError(f"unknown array type: {type(array)}")


def array_change_device(array, device: Device):
    """Change the device of an array"""
    if _is_numpy_array(array):
        if device != "cpu":
            raise ValueError(f"can not move numpy array to non-cpu device: {device}")
        return array
    elif _is_torch_array(array):
        return array.to(device=device)
    else:
        raise TypeError(f"unknown array type: {type(array)}")


def array_change_backend(array, backend: str):
    if _is_numpy_array(array):
        if backend == "numpy":
            return array
        elif backend == "torch":
            if not HAS_TORCH:
                raise ModuleNotFoundError(
                    "can not convert to `torch` arrays since PyTorch is not installed"
                )
            else:
                return torch.from_numpy(array)
        else:
            raise ValueError(f"unknown array backend: '{backend}'")

    elif _is_torch_array(array):
        if backend == "numpy":
            return array.numpy()
        elif backend == "torch":
            return array
        else:
            raise ValueError(f"unknown array backend: '{backend}'")

    else:
        raise TypeError(f"unknown array type: {type(array)}")


class DeviceWarning(RuntimeWarning):
    """
    Custom warning class for device mismatch in :py:class:`TensorBlock` and
    :py:class:`TensorMap`.
    """


class ArrayWrapper:
    """Small wrapper making Python arrays compatible with ``mts_array_t``."""

    def __init__(self, array):
        self.array = array
        self._shape = ctypes.ARRAY(c_uintptr_t, len(array.shape))(*array.shape)

        if _is_numpy_array(array):
            array_origin = _origin_numpy()
        elif _is_torch_array(array):
            array_origin = _origin_pytorch()
        else:
            raise ValueError(f"unknown array type: {type(array)}")

        mts_array = mts_array_t()
        # `mts_array_t::ptr` is a pointer to the PyObject `self`
        mts_array.ptr = ctypes.cast(
            ctypes.pointer(ctypes.py_object(self)), ctypes.c_void_p
        )

        @catch_exceptions
        def mts_array_origin(this, origin):
            origin[0] = array_origin

        # use storage.XXX.__class__ to get the right type for all functions
        mts_array.origin = mts_array.origin.__class__(mts_array_origin)

        mts_array.data = mts_array.data.__class__(_mts_array_data)

        mts_array.shape = mts_array.shape.__class__(_mts_array_shape)
        mts_array.reshape = mts_array.reshape.__class__(_mts_array_reshape)
        mts_array.swap_axes = mts_array.swap_axes.__class__(_mts_array_swap_axes)

        mts_array.create = mts_array.create.__class__(_mts_array_create)
        mts_array.copy = mts_array.copy.__class__(_mts_array_copy)
        mts_array.destroy = mts_array.destroy.__class__(_mts_array_destroy)

        mts_array.move_samples_from = mts_array.move_samples_from.__class__(
            _mts_array_move_samples_from
        )

        self._mts_array = mts_array

    def into_mts_array(self):
        """
        Get an mts_array_t instance for the wrapper array.

        This function increase the Python-side reference count to the wrapper to
        ensure the wrapper and arrays are kept alive. The reference count is
        reduced again when calling `mts_array_t::destroy` (which will typically
        be done by the Rust side of the code).
        """
        # The returned array is keeping a reference to this python object, we
        # need to tell Python so that it does not garbage-collect the wrapper
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(self))

        return self._mts_array


def _object_from_ptr(ptr):
    """Extract the Python object from a pointer to the PyObject"""
    return ctypes.cast(ptr, ctypes.POINTER(ctypes.py_object)).contents.value


@catch_exceptions
def _mts_array_data(this, data):
    storage = _object_from_ptr(this)

    if _is_numpy_array(storage.array):
        array = storage.array

    elif _is_torch_array(storage.array):
        array = storage.array

        if array.device.type != "cpu":
            raise ValueError("can only get data pointer for tensors on CPU")

        # `.numpy()` will fail if the data is on GPU or requires gradient
        # tracking, and the resulting array is sharing data storage with the
        # tensor, meaning we can take a pointer to it without the array being
        # freed immediately.
        array = array.numpy()

    if not array.data.c_contiguous:
        raise ValueError("can not get data pointer for non contiguous array")

    if not array.dtype == np.float64:
        raise ValueError(
            f"can not get data pointer for array type {array.dtype}, "
            "only float64 is supported. If you are trying to save a TensorMap "
            "to a file, you can set `use_numpy=True`."
        )

    data[0] = array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


@catch_exceptions
def _mts_array_shape(this, shape_ptr, shape_count):
    wrapper = _object_from_ptr(this)

    shape_ptr[0] = wrapper._shape
    shape_count[0] = len(wrapper._shape)


@catch_exceptions
def _mts_array_reshape(this, shape_ptr, shape_count):
    wrapper = _object_from_ptr(this)

    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    wrapper.array = wrapper.array.reshape(shape)
    wrapper._shape = ctypes.ARRAY(c_uintptr_t, len(shape))(*shape)


@catch_exceptions
def _mts_array_swap_axes(this, axis_1, axis_2):
    wrapper = _object_from_ptr(this)
    wrapper.array = wrapper.array.swapaxes(axis_1, axis_2)

    shape = wrapper.array.shape
    wrapper._shape = ctypes.ARRAY(c_uintptr_t, len(shape))(*shape)


@catch_exceptions
def _mts_array_create(this, shape_ptr, shape_count, new_array):
    wrapper = _object_from_ptr(this)

    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])
    dtype = wrapper.array.dtype

    if _is_numpy_array(wrapper.array):
        array = np.zeros(shape, dtype=dtype)
    elif _is_torch_array(wrapper.array):
        array = torch.zeros(shape, dtype=dtype, device=wrapper.array.device)

    new_wrapper = ArrayWrapper(array)
    new_array[0] = new_wrapper.into_mts_array()


@catch_exceptions
def _mts_array_copy(this, new_array):
    wrapper = _object_from_ptr(this)

    if _is_numpy_array(wrapper.array):
        array = wrapper.array.copy()
    elif _is_torch_array(wrapper.array):
        array = wrapper.array.clone()

    new_wrapper = ArrayWrapper(array)
    new_array[0] = new_wrapper.into_mts_array()


@catch_exceptions
def _mts_array_destroy(this):
    wrapper = _object_from_ptr(this)
    # remove the additional reference to the wrapper, added in `into_mts_array``
    ctypes.pythonapi.Py_DecRef(ctypes.py_object(wrapper))


@catch_exceptions
def _mts_array_move_samples_from(
    this,
    input,
    samples_ptr,
    samples_count,
    property_start,
    property_end,
):
    output = _object_from_ptr(this).array
    input = _object_from_ptr(input).array

    input_samples = []
    output_samples = []
    for i in range(samples_count):
        input_samples.append(samples_ptr[i].input)
        output_samples.append(samples_ptr[i].output)

    properties = slice(property_start, property_end)
    output[output_samples, ..., properties] = input[input_samples, ..., :]
