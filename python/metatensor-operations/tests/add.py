import numpy as np
import pytest

import metatensor
from metatensor import Labels, TensorBlock, TensorMap

from . import _gradcheck


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def keys():
    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    return keys


@pytest.fixture
def tensor_A(keys):
    block_1 = TensorBlock(
        values=np.array([[1.0, 2], [3, 5]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[[6.0, 1], [7, 2]], [[8, 3], [9, 4]]]),
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1]])),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=np.array([[1.0, 2], [3, 4], [5, 6]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [[[10.0, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )

    return TensorMap(keys, [block_1, block_2])


@pytest.fixture
def tensor_B(keys):
    block_1 = TensorBlock(
        values=np.array([[1.5, 2.1], [6.7, 10.2]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[[1, 0.1], [2, 0.2]], [[3, 0.3], [4.5, 0.4]]]),
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1]])),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=np.array([[10, 200.8], [3.76, 4.432], [545, 26]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[1.0, 1.1], [1.2, 1.3]],
                    [[1.4, 1.5], [1.0, 1.1]],
                    [[1.2, 1.3], [1.4, 1.5]],
                ]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )

    return TensorMap(keys, [block_1, block_2])


@pytest.fixture
def tensor_res_1(keys):
    block_1 = TensorBlock(
        values=np.array([[2.5, 4.1], [9.7, 15.2]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(np.array([[[7, 1.1], [9, 2.2]], [[11, 3.3], [13.5, 4.4]]])),
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1]])),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=np.array([[11, 202.8], [6.76, 8.432], [550, 32]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )

    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [
                    [[11.0, 12.1], [13.2, 14.3]],
                    [[15.4, 16.5], [11.0, 12.1]],
                    [[13.2, 14.3], [15.4, 16.5]],
                ]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )
    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))

    return TensorMap(keys, [block_1, block_2])


@pytest.fixture
def tensor_res_2(keys):
    block_1 = TensorBlock(
        values=np.array([[6.1, 7.1], [8.1, 10.1]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[[6.0, 1], [7, 2]], [[8, 3], [9, 4]]]),
            samples=Labels(["sample", "g"], np.array([[0, 1], [1, 1]])),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=np.array([[6.1, 7.1], [8.1, 9.1], [10.1, 11.1]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array(
                [[[10.0, 11], [12, 13]], [[14, 15], [10, 11]], [[12, 13], [14, 15]]]
            ),
            samples=Labels(
                ["sample", "g"],
                np.array([[0, 1], [1, 1], [2, 1]]),
            ),
            components=[
                Labels(["c"], np.array([[0], [1]])),
            ],
            properties=block_2.properties,
        ),
    )

    return TensorMap(keys, [block_1, block_2])


def test_self_add_tensors_no_gradient(tensor_A, tensor_B, tensor_res_1):
    tensor_A = metatensor.remove_gradients(tensor_A)
    tensor_B = metatensor.remove_gradients(tensor_B)
    tensor_res_1 = metatensor.remove_gradients(tensor_res_1)

    tensor_A_copy = tensor_A.copy()
    tensor_B_copy = tensor_B.copy()

    metatensor.allclose_raise(metatensor.add(tensor_A, tensor_B), tensor_res_1)

    # Check the tensors haven't be modified in place
    metatensor.equal_raise(tensor_A, tensor_A_copy)
    metatensor.equal_raise(tensor_B, tensor_B_copy)


def test_self_add_tensors_gradient(tensor_A, tensor_B, tensor_res_1):
    metatensor.allclose_raise(metatensor.add(tensor_A, tensor_B), tensor_res_1)


def test_self_add_scalar_gradient(tensor_A, tensor_res_2):
    tensor_A_copy = tensor_A.copy()

    metatensor.allclose_raise(metatensor.add(tensor_A, 5.1), tensor_res_2)

    # Check the tensors haven't be modified in place
    metatensor.equal_raise(tensor_A, tensor_A_copy)


def test_self_add_error():
    block = TensorBlock(
        values=np.array([[1, 2], [3, 5]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
    tensor = TensorMap(keys, [block])

    message = "`A` must be a metatensor TensorMap, not <class 'numpy.ndarray'>"
    with pytest.raises(TypeError, match=message):
        metatensor.add(np.ones((3, 4)), tensor)

    message = (
        "`B` must be a metatensor TensorMap or a scalar value, "
        "not <class 'numpy.ndarray'>"
    )
    with pytest.raises(TypeError, match=message):
        metatensor.add(tensor, np.ones((3, 4)))


def test_add_finite_difference():
    def function(array):
        tensor_1 = _gradcheck.cartesian_linear(array)
        tensor_2 = _gradcheck.cartesian_cubic(array)
        return metatensor.add(tensor_1, tensor_2)

    array = np.random.rand(5, 3)
    _gradcheck.check_finite_differences(function, array, parameter="g")


@pytest.mark.skipif(not HAS_TORCH, reason="requires torch")
def test_torch_add_finite_difference():
    def function(array):
        tensor_1 = _gradcheck.cartesian_linear(array)
        tensor_2 = _gradcheck.cartesian_cubic(array)
        return metatensor.add(tensor_1, tensor_2)

    array = torch.rand(5, 3, dtype=torch.float64)
    _gradcheck.check_finite_differences(function, array, parameter="g")
