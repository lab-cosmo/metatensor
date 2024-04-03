import os

import numpy as np
import pytest

import metatensor
from metatensor import Labels, NotEqualError, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_allclose_nograd():
    block_1 = TensorBlock(
        values=np.array([[1.0, 2], [3, 5]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels.range("p", 2),
    )
    block_2 = TensorBlock(
        values=np.array([[1.0, 2], [3, 4], [5, 6], [1, 2], [3, 4], [5, 6]]),
        samples=Labels.range("s", 6),
        components=[],
        properties=Labels.range("p", 2),
    )

    block_3 = TensorBlock(
        values=np.array([[1.0], [2]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_4 = TensorBlock(
        values=np.array([[23.0], [53], [83]]),
        samples=Labels(["s"], np.array([[0], [2], [7]])),
        components=[],
        properties=Labels(["p"], np.array([[6]])),
    )
    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0], [1, 0]]))
    X = TensorMap(keys, [block_1, block_2])
    assert metatensor.allclose(X, X)
    Y = TensorMap(keys, [block_3, block_4])
    assert not metatensor.allclose(X, Y)

    message = (
        r"blocks for key \(key_1=0, key_2=0\) are different: values shapes are "
        r"different"
    )
    with pytest.raises(NotEqualError, match=message):
        metatensor.allclose_raise(X, Y)

    block_1_c = TensorBlock(
        values=np.array(
            [
                [
                    [[1, 0.5], [4, 2], [1.5, 6.5]],
                    [[2, 1], [6, 3], [6.1, 3.5]],
                    [[9, 9], [9, 9.8], [10, 10.5]],
                ],
                [
                    [[3, 1.5], [7, 3.5], [3.7, 1.5]],
                    [[5, 2.5], [8, 4], [6.3, 1.5]],
                    [[5, 7.1], [8, 4.8], [6.3, 14.466]],
                ],
            ],
            dtype=np.float64,
        ),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[
            Labels.range("c_1", 3),
            Labels.range("c_2", 3),
        ],
        properties=Labels.range("p", 2),
    )
    block_1_c_copy = TensorBlock(
        values=block_1_c.values + 0.1e-6,
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[
            Labels.range("c_1", 3),
            Labels.range("c_2", 3),
        ],
        properties=Labels.range("p", 2),
    )

    block_2_c = TensorBlock(
        values=np.array(
            [
                [[[1, 2], [6.8, 2.8]], [[4.1, 6.2], [66.8, 62.8]]],
                [[[3, 4], [36, 6.4]], [[83, 84], [73.76, 76.74]]],
                [[[5, 6], [58, 68]], [[23.4, 5643.3], [234.5, 3247.6]]],
                [[[5.6, 6.6], [5.68, 668]], [[55.6, 676.76], [775.68, 0.668]]],
                [[[1, 2], [17.7, 27.7]], [[77.1, 22.2], [1.11, 3.42]]],
            ]
        ),
        samples=Labels.range("s", 5),
        components=[
            Labels(["c_1"], np.array([[3], [5]])),
            Labels(["c_2"], np.array([[6], [8]])),
        ],
        properties=Labels.range("p", 2),
    )
    block_2_c_copy = TensorBlock(
        values=block_2_c.values + 0.1e-6,
        samples=Labels.range("s", 5),
        components=[
            Labels(["c_1"], np.array([[3], [5]])),
            Labels(["c_2"], np.array([[6], [8]])),
        ],
        properties=Labels.range("p", 2),
    )
    X_c = TensorMap(keys, [block_1_c, block_2_c])
    X_c_copy = TensorMap(keys, [block_1_c_copy, block_2_c_copy])
    assert not metatensor.allclose(X, X_c)

    assert metatensor.allclose(X_c, X_c)
    assert not metatensor.allclose(X_c, X_c_copy)
    assert metatensor.allclose(X_c, X_c_copy, rtol=1e-5)
    assert metatensor.allclose(X_c, X_c_copy, atol=1e-1)


def test_self_allclose_grad():
    tensor1 = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        use_numpy=True,
    )
    blocks = []
    blocks_e6 = []
    for block in tensor1:
        blocks.append(block.copy())
        be6 = block.copy()
        be6.values[:] += 1e-6
        blocks_e6.append(be6)

    tensor1_copy = TensorMap(tensor1.keys, blocks)
    tensor1_e6 = TensorMap(tensor1.keys, blocks_e6)
    assert metatensor.allclose(tensor1, tensor1_copy)
    assert not metatensor.allclose(tensor1, tensor1_e6)
    assert metatensor.allclose(tensor1, tensor1_e6, rtol=1e-5, atol=1e-5)

    message = (
        r"blocks for key \(o3_lambda=0, o3_sigma=1, center_type=1, "
        r"neighbor_type=1\) are different: values are not allclose"
    )
    with pytest.raises(NotEqualError, match=message):
        metatensor.allclose_raise(tensor1, tensor1_e6)

    message = "values are not allclose"
    with pytest.raises(NotEqualError, match=message):
        metatensor.allclose_block_raise(tensor1.block(0), tensor1_e6.block(0))


def test_self_allclose_exceptions():
    block_1 = TensorBlock(
        values=np.array([[1], [2]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_2 = TensorBlock(
        values=np.array([[1], [2]]),
        samples=Labels(["other_s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_3 = TensorBlock(
        values=np.array([[1], [2]]),
        samples=Labels(["s"], np.array([[0], [6]])),
        components=[],
        properties=Labels(["p"], np.array([[0]])),
    )

    block_4 = TensorBlock(
        values=np.array([[[1], [4]], [[44], [2]]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[
            Labels(["c"], np.array([[0], [6]])),
        ],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_5 = TensorBlock(
        values=np.array([[[1], [4]], [[44], [2]]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[
            Labels(["other_c"], np.array([[0], [6]])),
        ],
        properties=Labels(["p"], np.array([[0]])),
    )

    block_6 = TensorBlock(
        values=np.array([[[1], [4]], [[44], [2]]]),
        samples=Labels(["s"], np.array([[2], [0]])),
        components=[
            Labels(["c"], np.array([[0], [6]])),
        ],
        properties=Labels(["p"], np.array([[0]])),
    )

    assert not metatensor.allclose_block(block_1, block_2)

    message = (
        "inputs to 'allclose' should have the same samples, "
        "but they are not the same or not in the same order"
    )
    with pytest.raises(NotEqualError, match=message):
        metatensor.allclose_block_raise(block_1, block_2)

    message = (
        "inputs to 'allclose' should have the same samples, "
        "but they are not the same or not in the same order"
    )
    with pytest.raises(NotEqualError, match=message):
        metatensor.allclose_block_raise(block_1, block_3)

    message = "values shapes are different"
    with pytest.raises(NotEqualError, match=message):
        metatensor.allclose_block_raise(block_1, block_4)

    message = (
        "inputs to 'allclose' should have the same components, "
        "but they are not the same or not in the same order"
    )
    with pytest.raises(NotEqualError, match=message):
        metatensor.allclose_block_raise(block_5, block_4)

    message = (
        "inputs to 'allclose' should have the same samples, "
        "but they are not the same or not in the same order"
    )
    with pytest.raises(NotEqualError, match=message):
        metatensor.allclose_block_raise(block_6, block_4)


def test_self_allclose_exceptions_gradient():
    block_1 = TensorBlock(
        values=np.array([[1.0], [2.0]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((2, 1), 11.0),
            samples=Labels(["sample", "g"], np.array([[0, -2], [1, 3]])),
            components=[],
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=np.array([[1.0], [2.0]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0]])),
    )

    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((2, 1), 11.0),
            samples=Labels(["sample", "other_g"], np.array([[0, -2], [1, 3]])),
            components=[],
            properties=block_2.properties,
        ),
    )

    message = (
        "inputs to 'allclose' should have the same gradients: "
        "gradient 'g' samples are not the same or not in the same order"
    )
    with pytest.raises(NotEqualError, match=message):
        metatensor.allclose_block_raise(block_1, block_2)

    block_3 = TensorBlock(
        values=np.array([[1.0], [2.0]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0]])),
    )

    block_3.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((2, 1), 1.0),
            samples=Labels(["sample", "g"], np.array([[0, -2], [1, 3]])),
            components=[],
            properties=block_3.properties,
        ),
    )

    message = "gradient 'g' values are not allclose"
    with pytest.raises(NotEqualError, match=message):
        metatensor.allclose_block_raise(block_1, block_3)

    block_4 = TensorBlock(
        values=np.array([[1.0], [2.0]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0]])),
    )

    block_4.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((2, 3, 1), 1.0),
            samples=Labels(["sample", "g"], np.array([[0, -2], [1, 3]])),
            components=[Labels(["c_1"], np.array([[-1], [0], [1]]))],
            properties=block_4.properties,
        ),
    )

    block_5 = TensorBlock(
        values=np.array([[1.0], [2.0]]),
        samples=Labels(["s"], np.array([[0], [2]])),
        components=[],
        properties=Labels(["p"], np.array([[0]])),
    )

    block_5.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((2, 3, 1), 1.0),
            samples=Labels(["sample", "g"], np.array([[0, -2], [1, 3]])),
            components=[Labels(["c_1"], np.array([[-1], [6], [1]]))],
            properties=block_5.properties,
        ),
    )

    message = (
        "inputs to 'allclose' should have the same gradients: "
        "gradient 'g' components are not the same or not in the same order"
    )
    with pytest.raises(NotEqualError, match=message):
        metatensor.allclose_block_raise(block_5, block_4)
