import ctypes
import functools
import operator
import os

import numpy as np


try:
    import torch

    TorchDevice = torch.device
    TorchDtype = torch.dtype
except ImportError:

    class TorchDevice:
        pass

    class TorchDtype:
        pass


from ._c_api import MTS_BUFFER_SIZE_ERROR
from .status import MetatensorError, _save_exception


def _call_with_growing_buffer(callback, initial=1024):
    bufflen = initial

    while True:
        buffer = ctypes.create_string_buffer(bufflen)
        try:
            callback(buffer, bufflen)
            break
        except MetatensorError as e:
            if e.status == MTS_BUFFER_SIZE_ERROR:
                # grow the buffer and retry
                bufflen *= 2
            else:
                raise
    return buffer.value.decode("utf8")


def catch_exceptions(function):
    @functools.wraps(function)
    def inner(*args, **kwargs):
        try:
            function(*args, **kwargs)
        except Exception as e:
            _save_exception(e)
            return -1
        return 0

    return inner


def _ptr_to_ndarray(ptr, shape, dtype):
    if functools.reduce(operator.mul, shape) == 0:
        return np.empty(shape=shape, dtype=dtype)

    assert ptr is not None
    array = np.ctypeslib.as_array(ptr, shape=shape)
    assert array.dtype == dtype
    assert not array.flags["OWNDATA"]
    array.flags["WRITEABLE"] = True
    return array


def _ptr_to_const_ndarray(ptr, shape, dtype):
    array = _ptr_to_ndarray(ptr, shape, dtype)
    array.flags["WRITEABLE"] = False
    return array


def _to_arguments_parse(context, *args, **kwargs):
    """Parse arguments to the various `to()` functions"""
    dtype = kwargs.get("dtype")
    device = kwargs.get("device")

    for positional in args:
        if isinstance(positional, (TorchDevice, str)):
            if device is None:
                device = positional
                continue
            else:
                raise ValueError(f"can not give a device twice in {context}")
        elif isinstance(positional, TorchDtype):
            if dtype is None:
                dtype = positional
                continue
            else:
                raise ValueError(f"can not give a dtype twice in {context}")
        else:
            # checking for numpy dtype is a bit more complex,
            # since a lof of things can be dtypes in numpy
            try:
                positional_as_dtype = np.dtype(positional)
            except TypeError:
                # failed to parse as a dtype, this should end up in the TypeError below
                positional_as_dtype = np.object_

            if np.issubdtype(positional_as_dtype, np.number):
                if dtype is None:
                    dtype = positional
                    continue
                else:
                    raise ValueError(f"can not give a dtype twice in {context}")

        raise TypeError(f"unexpected type in {context}: {type(positional)}")

    return dtype, device


cmake_prefix_path = os.path.join(os.path.dirname(__file__), "lib", "cmake")
"""
Path containing the CMake configuration files for the underlying C library
"""
