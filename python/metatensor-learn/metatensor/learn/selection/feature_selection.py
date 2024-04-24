"""
Wrappers for the feature selectors of `scikit-matter`_.

.. _`scikit-matter`: https://scikit-matter.readthedocs.io/en/latest/selection.html
"""

from skmatter._selection import _CUR, _FPS

from ._selection import GreedySelector


class FPS(GreedySelector):
    """
    Transformer that performs Greedy Feature Selection using Farthest Point Sampling.

    If `n_to_select` is an `int`, all blocks will have this many features selected. In
    this case, `n_to_select` must be <= than the fewest number of features in any block.

    If `n_to_select` is a dict, it must have keys that are tuples corresponding to the
    key values of each block. In this case, the values of the `n_to_select` dict can be
    int that specify different number of features to select for each block.

    If `n_to_select` is -1, all features for every block will be selected. This is
    useful, for instance, for plotting Hausdorff distances, which can be accessed
    through the selector.haussdorf_at_select property after calling the fit() method.

    Refer to :py:class:`skmatter.feature_selection.FPS` for full documentation.
    """

    def __init__(
        self,
        initialize=0,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selector_class=_FPS,
            selection_type="feature",
            initialize=initialize,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )


class CUR(GreedySelector):
    """
    Transformer that performs Greedy Feature Selection with CUR.

    If `n_to_select` is an `int`, all blocks will have this many features selected. In
    this case, `n_to_select` must be <= than the fewest number of features in any block.

    If `n_to_select` is a dict, it must have keys that are tuples corresponding to the
    key values of each block. In this case, the values of the `n_to_select` dict can be
    int that specify different number of features to select for each block.

    If `n_to_select` is -1, all features for every block will be selected.

    Refer to :py:class:`skmatter.feature_selection.CUR` for full documentation.
    """

    def __init__(
        self,
        recompute_every=1,
        k=1,
        tolerance=1e-12,
        n_to_select=None,
        score_threshold=None,
        score_threshold_type="absolute",
        progress_bar=False,
        full=False,
        random_state=0,
    ):
        super().__init__(
            selector_class=_CUR,
            selection_type="feature",
            recompute_every=recompute_every,
            k=k,
            tolerance=tolerance,
            n_to_select=n_to_select,
            score_threshold=score_threshold,
            score_threshold_type=score_threshold_type,
            progress_bar=progress_bar,
            full=full,
            random_state=random_state,
        )
