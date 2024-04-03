import numpy as np
import pytest

import metatensor
from metatensor import Labels


def test_ordinary_usage():
    """Test one-hot encoding for the ordinary use case."""
    original_labels = Labels(
        names=["atom", "type"],
        values=np.array([[0, 6], [1, 1], [2, 1], [3, 1], [4, 6], [5, 1]]),
    )
    possible_labels = Labels(names=["type"], values=np.array([[1], [6]]))
    correct_encoding = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 0],
        ],
        dtype=np.int32,
    )
    one_hot_encoding = metatensor.one_hot(original_labels, possible_labels)
    np.testing.assert_allclose(one_hot_encoding, correct_encoding)


def test_additional_value():
    """Test one-hot encoding when additional values are provided."""
    original_labels = Labels(
        names=["atom", "type"],
        values=np.array([[0, 6], [1, 1], [2, 1], [3, 1], [4, 6], [5, 1]]),
    )
    possible_labels = Labels(names=["type"], values=np.array([[1], [6], [8]]))
    correct_encoding = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=np.int32,
    )
    one_hot_encoding = metatensor.one_hot(original_labels, possible_labels)
    np.testing.assert_allclose(one_hot_encoding, correct_encoding)


def test_multiple_names():
    """Test one-hot encoding if multiple dimension names are provided."""
    original_labels = Labels(
        names=["atom", "type"],
        values=np.array([[0, 6], [1, 1], [2, 1], [3, 1], [4, 6], [5, 1]]),
    )
    possible_labels = Labels(
        names=["atom", "type"],
        values=np.array([[0, 6], [1, 1]]),
    )
    with pytest.raises(ValueError, match="only one label dimension can be extracted"):
        metatensor.one_hot(original_labels, possible_labels)


def test_wrong_name():
    """Test one-hot encoding if a wrong name is provided."""
    original_labels = Labels(
        names=["atom", "type"],
        values=np.array([[0, 6], [1, 1], [2, 1], [3, 1], [4, 6], [5, 1]]),
    )
    possible_labels = Labels(names=["not_present"], values=np.array([[1], [6], [8]]))

    message = "'not_present' not found in the dimensions of these Labels"
    with pytest.raises(ValueError, match=message):
        metatensor.one_hot(original_labels, possible_labels)


def test_missing_value():
    """Test one-hot encoding if there is a value is missing
    from the provided dimension label."""
    original_labels = Labels(
        names=["atom", "type"],
        values=np.array([[0, 6], [1, 1], [2, 1], [3, 1], [4, 6], [5, 1]]),
    )
    possible_labels = Labels(names=["type"], values=np.array([[1], [8]]))

    message = "type=6 is present in the labels, but was not found in the dimension"
    with pytest.raises(ValueError, match=message):
        metatensor.one_hot(original_labels, possible_labels)
