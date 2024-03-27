"""
Module to test FPS and CUR selectors in
metatensor.learn.selection.feature_selection
"""

import numpy as np
import pytest
import skmatter.feature_selection
from numpy.testing import assert_equal, assert_raises

import metatensor
from metatensor import Labels
from metatensor.learn.selection.feature_selection import CUR, FPS

from .selection_utils import (
    random_single_block_no_components_tensor_map,
    random_tensor_map_with_components,
)


@pytest.fixture
def X1():
    return random_single_block_no_components_tensor_map(
        use_torch=False, use_metatensor_torch=False
    )


@pytest.fixture
def X2():
    return random_tensor_map_with_components(
        use_torch=False, use_metatensor_torch=False
    )


@pytest.mark.parametrize(
    "selector_class, skmatter_selector_class",
    [(FPS, skmatter.feature_selection.FPS), (CUR, skmatter.feature_selection.CUR)],
)
def test_fit(X1, selector_class, skmatter_selector_class):
    selector = selector_class(n_to_select=2)
    selector.fit(X1)
    support = selector.support[0].properties

    skmatter_selector = skmatter_selector_class(n_to_select=2)
    skmatter_selector.fit(X1[0].values)
    skmatter_support = skmatter_selector.get_support(indices=True)
    skmatter_support_labels = Labels(
        names=["properties"],
        values=np.array(
            [[support_i] for support_i in skmatter_support], dtype=np.int32
        ),
    )

    assert support == skmatter_support_labels


@pytest.mark.parametrize(
    "selector_class, skmatter_selector_class",
    [(FPS, skmatter.feature_selection.FPS), (CUR, skmatter.feature_selection.CUR)],
)
def test_transform(X1, selector_class, skmatter_selector_class):
    selector = selector_class(n_to_select=2)
    selector.fit(X1)
    X_trans = selector.transform(X1)

    skmatter_selector = skmatter_selector_class(n_to_select=2)
    skmatter_selector.fit(X1[0].values)
    X_trans_skmatter = skmatter_selector.transform(X1[0].values)

    assert_equal(X_trans[0].values, X_trans_skmatter)


@pytest.mark.parametrize("selector_class", [FPS, CUR])
def test_fit_transform(X1, selector_class):
    selector = selector_class(n_to_select=2)

    X_ft = selector.fit(X1).transform(X1)
    metatensor.equal_raise(selector.fit_transform(X1), X_ft)


@pytest.mark.parametrize("selector_class", [FPS])
def test_get_select_distance(X2, selector_class):
    selector = selector_class(n_to_select=3)
    selector.fit(X2)
    select_distance = selector.get_select_distance

    assert select_distance is not None

    # Check distances sorted in descending order, with an inf as the first
    # entry
    for block in select_distance:
        assert block.values[0][0] == np.inf
        for i, val in enumerate(block.values[0][1:], start=1):
            assert val < block.values[0][i - 1]


@pytest.mark.parametrize("selector_class", [FPS])
def test_get_select_distance_n_to_select(X2, selector_class):
    # Case 1: select all features for every block (n_to_select = -1)
    selector = selector_class(n_to_select=-1)
    selector.fit(X2)
    select_distance = selector.get_select_distance
    for block in select_distance:
        assert len(block.properties) == 5

    # Case 2: select subset of features but same for each block
    n = 2
    selector = selector_class(n_to_select=n)
    selector.fit(X2)
    select_distance = selector.get_select_distance
    for block in select_distance:
        assert len(block.properties) == n

    # Case 3: select subset of features but different for each block
    keys = X2.keys
    n = {tuple(key): 2 * i + 1 for i, key in enumerate(keys)}
    selector = selector_class(n_to_select=n)
    selector.fit(X2)
    select_distance = selector.get_select_distance
    for i, key in enumerate(keys):
        assert len(select_distance[key].properties) == 2 * i + 1


@pytest.mark.parametrize("selector_class", [CUR])
def test_get_select_distance_raises(X2, selector_class):
    selector = selector_class(n_to_select=3)
    selector.fit(X2)
    with assert_raises(ValueError):
        selector.get_select_distance
