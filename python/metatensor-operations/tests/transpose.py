import os

import numpy as np
import pytest

import equistore
from equistore import Labels, TensorBlock


DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
TEST_FILE = "qm7-spherical-expansion.npz"


@pytest.fixture
def tensorblock():
    block = TensorBlock(
        samples=Labels(["s"], np.array([[0], [1], [2], [3]])),
        components=[
            Labels(["c3"], np.array([[0], [1], [2]])),
            Labels(["c4"], np.array([[4], [5], [6]])),
            Labels(["c5"], np.array([[7], [8], [9]])),
        ],
        properties=Labels(["p"], np.array([[-1]])),
        values=np.full((4, 3, 3, 3, 1), 1),
    )
    return block


@pytest.fixture
def tensorblock_with_gradient():
    block = TensorBlock(
        samples=Labels(["s"], np.array([[0], [1], [2], [3]])),
        components=[
            Labels(["c3"], np.array([[0], [1], [2]])),
            Labels(["c4"], np.array([[4], [5], [6]])),
            Labels(["c5"], np.array([[7], [8], [9]])),
        ],
        properties=Labels(["p"], np.array([[-1]])),
        values=np.full((4, 3, 3, 3, 1), 1),
    )
    block.add_gradient(
        "parameter",
        samples=Labels(["sample", "ds"], np.array([[0, 0], [1, 1], [2, 2], [3, 3]])),
        components=[
            Labels(
                ["c1"],
                np.array(
                    [
                        [0],
                        [1],
                    ]
                ),
            ),
            Labels(
                ["c2"],
                np.array(
                    [
                        [2],
                        [3],
                    ]
                ),
            ),
            Labels(["c3"], np.array([[0], [1], [2]])),
            Labels(["c4"], np.array([[4], [5], [6]])),
            Labels(["c5"], np.array([[7], [8], [9]])),
        ],
        data=np.full((4, 2, 2, 3, 3, 3, 1), -1),
    )
    return block


@pytest.fixture
def tensormap():
    return equistore.load(os.path.join(DATA_ROOT, TEST_FILE), use_numpy=True)


class TestTranspose:
    def test_transpose_block(self, tensorblock):
        """
        Tests :py:func:`tranpose_block` for a block without a gradient.
        """
        block_T = equistore.transpose_block(tensorblock)
        # Check shape
        assert block_T.values.shape == (1, 3, 3, 3, 4)
        # Check values

    def transpose_block_with_gradient(self, tensorblock_with_gradient):
        """
        Tests :py:func:`tranpose_block` with a block that has a gradient.
        """
        # Check shape
        # Check values

    def test_transpose(self, tensormap):
        """
        Tests :py:func:`tranpose` for a TensorMap.
        """
        # Check shape
        # Check values
