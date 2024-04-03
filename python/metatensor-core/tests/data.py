import gc
import os
import weakref

import numpy as np
from numpy.testing import assert_equal


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import ctypes

import metatensor
from metatensor._c_api import (
    MTS_SUCCESS,
    c_uintptr_t,
    mts_array_t,
    mts_sample_mapping_t,
)


def free_mts_array(array):
    array.destroy(array.ptr)


class ArrayWrapperMixin:
    def test_origin(self):
        array = self.create_array((2, 3, 4))
        wrapper = metatensor.data.ArrayWrapper(array)
        mts_array = wrapper.into_mts_array()

        assert id(metatensor.data.mts_array_to_python_array(mts_array)) == id(array)

        origin = metatensor.data.data_origin(mts_array)
        assert metatensor.data.data_origin_name(origin) == self.expected_origin()

        free_mts_array(mts_array)

    def test_shape(self):
        array = self.create_array((2, 3, 4))
        wrapper = metatensor.data.ArrayWrapper(array)
        mts_array = wrapper.into_mts_array()

        assert _get_shape(mts_array, self) == [2, 3, 4]

        new_shape = ctypes.ARRAY(c_uintptr_t, 4)(2, 3, 2, 2)
        status = mts_array.reshape(mts_array.ptr, new_shape, len(new_shape))
        assert status == MTS_SUCCESS

        assert _get_shape(mts_array, self) == [2, 3, 2, 2]

        free_mts_array(mts_array)

    def test_swap_axes(self):
        array = self.create_array((2, 3, 18, 23))
        wrapper = metatensor.data.ArrayWrapper(array)
        mts_array = wrapper.into_mts_array()

        mts_array.swap_axes(mts_array.ptr, 1, 3)
        assert _get_shape(mts_array, self) == [2, 23, 18, 3]

        free_mts_array(mts_array)

    def test_create(self):
        array = self.create_array((2, 3))
        array_ref = weakref.ref(array)

        wrapper = metatensor.data.ArrayWrapper(array)
        mts_array = wrapper.into_mts_array()

        new_mts_array = mts_array_t()
        new_shape = ctypes.ARRAY(c_uintptr_t, 2)(18, 4)
        status = mts_array.create(
            mts_array.ptr, new_shape, len(new_shape), new_mts_array
        )
        assert status == MTS_SUCCESS

        new_array = metatensor.data.mts_array_to_python_array(new_mts_array)
        assert id(new_array) != id(array)

        assert_equal(new_array, np.zeros((18, 4)))

        del array
        del wrapper
        gc.collect()

        # there is still one reference to the array through mts_array
        assert array_ref() is not None

        free_mts_array(mts_array)
        del mts_array
        gc.collect()

        assert array_ref() is None

        free_mts_array(new_mts_array)

    def test_copy(self):
        array = self.create_array((2, 3, 4))
        array[1, :, :] = 3
        array[1, 2, :] = 5
        wrapper = metatensor.data.ArrayWrapper(array)
        mts_array = wrapper.into_mts_array()

        copy = mts_array_t()
        status = mts_array.copy(mts_array.ptr, copy)
        assert status == MTS_SUCCESS

        array_copy = metatensor.data.mts_array_to_python_array(copy)
        assert id(array_copy) != id(array)

        assert_equal(np.array(array_copy), np.array(array))

        free_mts_array(mts_array)
        free_mts_array(copy)

    def test_move_samples_from(self):
        array = self.create_array((2, 3, 8))
        array[:] = 4.0
        wrapper = metatensor.data.ArrayWrapper(array)
        mts_array = wrapper.into_mts_array()

        other = self.create_array((1, 3, 4))
        other[:] = 2.0
        wrapper_other = metatensor.data.ArrayWrapper(other)
        mts_array_other = wrapper_other.into_mts_array()

        move = mts_sample_mapping_t(input=0, output=1)
        move_array = ctypes.ARRAY(mts_sample_mapping_t, 1)(move)

        mts_array.move_samples_from(
            mts_array.ptr,
            mts_array_other.ptr,
            move_array,
            len(move_array),
            3,
            7,
        )
        expected = np.array(
            [
                # unmodified first sample
                [
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                ],
                # second sample changed for properties 3:7
                [
                    [4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 4.0],
                    [4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 4.0],
                    [4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0, 4.0],
                ],
            ]
        )
        assert_equal(array, expected)

        free_mts_array(mts_array)
        free_mts_array(mts_array_other)


class TestNumpyData(ArrayWrapperMixin):
    def expected_origin(self):
        return "metatensor.data.array.numpy"

    def create_array(self, shape):
        return np.zeros(shape)


if HAS_TORCH:

    class TestTorchData(ArrayWrapperMixin):
        def expected_origin(self):
            return "metatensor.data.array.torch"

        def create_array(self, shape):
            return torch.zeros(shape, device="cpu")


def _get_shape(mts_array, test):
    shape_ptr = ctypes.POINTER(c_uintptr_t)()
    shape_count = c_uintptr_t()
    status = mts_array.shape(mts_array.ptr, shape_ptr, shape_count)

    assert status == MTS_SUCCESS

    shape = []
    for i in range(shape_count.value):
        shape.append(shape_ptr[i])

    return shape


TEST_ORIGIN = metatensor.data.array._register_origin("python.test-origin")
metatensor.data.register_external_data_wrapper(
    "python.test-origin",
    metatensor.data.extract.ExternalCpuArray,
)


@metatensor.utils.catch_exceptions
def create_test_array(shape_ptr, shape_count, array):
    shape = []
    for i in range(shape_count):
        shape.append(shape_ptr[i])

    data = np.zeros(shape)
    wrapper = metatensor.data.array.ArrayWrapper(data)
    mts_array = wrapper.into_mts_array()

    @metatensor.utils.catch_exceptions
    def test_origin(this, origin):
        origin[0] = TEST_ORIGIN

    mts_array.origin = mts_array.origin.__class__(test_origin)

    array[0] = mts_array


def test_parent_keepalive():
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "data.npz",
    )
    tensor = metatensor.io.load_custom_array(path.encode("utf8"), create_test_array)

    values = tensor.block(0).values
    assert isinstance(values, metatensor.data.extract.ExternalCpuArray)
    assert values._parent is not None

    tensor_ref = weakref.ref(tensor)
    del tensor
    gc.collect()

    # values should keep the tensor alive
    assert tensor_ref() is not None

    view = values[::2]
    del values
    gc.collect()

    # view should keep the tensor alive
    assert tensor_ref() is not None

    transformed = np.sum(view)
    del view
    gc.collect()

    # transformed should NOT keep the tensor alive
    assert tensor_ref() is None

    assert np.isclose(transformed, 1.1596965632269784)
