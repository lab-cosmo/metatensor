from typing import Type, Union

import numpy as np
import skmatter._selection

import metatensor

from .._backend import Labels, TensorBlock, TensorMap


class GreedySelector:
    """
    Wraps :py:class:`skmatter._selection.GreedySelector` for a TensorMap.

    The class creates a selector for each block. The selection will be done based the
    values of each :py:class:`TensorBlock`. Gradients will not be considered for the
    selection.
    """

    def __init__(
        self,
        selector_class: Type[skmatter._selection.GreedySelector],
        selection_type: str,
        n_to_select: Union[int, dict],
        **selector_arguments,
    ) -> None:
        self._selector_class = selector_class
        self._selection_type = selection_type
        self._n_to_select = n_to_select
        self._selector_arguments = selector_arguments

        self._selector_arguments["selection_type"] = self._selection_type
        self._support = None
        self._select_distance = None

    @property
    def selector_class(self) -> Type[skmatter._selection.GreedySelector]:
        """
        The class to perform the selection. Usually one of 'FPS' or 'CUR'.
        """
        return self._selector_class

    @property
    def selection_type(self) -> str:
        """
        Whether to choose a subset of columns ('feature') or rows ('sample').
        """
        return self._selection_type

    @property
    def selector_arguments(self) -> dict:
        """
        Arguments passed to the ``selector_class``.
        """
        return self._selector_arguments

    @property
    def support(self) -> TensorMap:
        """
        TensorMap containing the support.
        """
        if self._support is None:
            raise ValueError("No selections. Call fit method first.")

        return self._support

    @property
    def get_select_distance(self) -> TensorMap:
        """
        Returns a TensorMap containing the Hausdorff distances.

        For each block, the metadata of the relevant axis (i.e. samples or properties,
        depending on whether sample or feature selection is being performed) is sorted
        and returned according to the Hausdorff distance, in descending order.
        """
        if self._selector_class == skmatter._selection._CUR:
            raise ValueError("Hausdorff distances not available for CUR in skmatter.")
        if self._select_distance is None:
            raise ValueError("No Hausdorff distances. Call fit method first.")

        return self._select_distance

    def fit(self, X: TensorMap, warm_start: bool = False) -> None:
        """
        Learn the features to select.

        :param X: the input training vectors to fit.
        :param warm_start: bool, whether the fit should continue after having already
            run, after increasing `n_to_select`. Assumes it is called with the same X.
        """
        # Check that we have only 0 or 1 comoponent axes
        if len(X.component_names) == 0:
            has_components = False
        elif len(X.component_names) == 1:
            has_components = True
        else:
            assert len(X.component_names) > 1
            raise ValueError("Can only handle TensorMaps with a single component axis.")

        support_blocks = []
        if self._selector_class == skmatter._selection._FPS:
            hausdorff_blocks = []
        for key, block in X.items():
            # Parse the n_to_select argument
            max_n = (
                len(block.properties)
                if self._selection_type == "feature"
                else len(block.samples)
            )
            if isinstance(self._n_to_select, int):
                if (
                    self._n_to_select == -1
                ):  # set to the number of samples/features for this block
                    tmp_n_to_select = max_n
                else:
                    tmp_n_to_select = self._n_to_select

            elif isinstance(self._n_to_select, dict):
                tmp_n_to_select = self._n_to_select[tuple(key.values)]
            else:
                raise ValueError("n_to_select must be an int or a dict.")

            if not (0 < tmp_n_to_select <= max_n):
                raise ValueError(
                    f"n_to_select ({tmp_n_to_select}) must > 0 and <= the number of "
                    f"{self._selection_type} for the given block ({max_n})."
                )

            selector = self.selector_class(
                n_to_select=tmp_n_to_select, **self.selector_arguments
            )

            # If the block has components, reshape to a 2D array such that the
            # components expand along the dimension *not* being selected.
            block_vals = block.values
            if has_components:
                n_components = len(block.components[0])
                if self._selection_type == "feature":
                    # Move components into samples
                    block_vals = block_vals.reshape(
                        (block_vals.shape[0] * n_components, block_vals.shape[2])
                    )
                else:
                    assert self._selection_type == "sample"
                    # Move components into features
                    block_vals = block.values.reshape(
                        (block_vals.shape[0], block_vals.shape[2] * n_components)
                    )

            # Fit on the block values
            selector.fit(block_vals, warm_start=warm_start)

            # Build the support TensorMap. In this case we want the mask to be a
            # list of bools, such that the original order of the metadata is
            # preserved.
            supp_mask = selector.get_support()
            if self._selection_type == "feature":
                supp_samples = Labels.single()
                supp_properties = Labels(
                    names=block.properties.names,
                    values=block.properties.values[supp_mask],
                )
            elif self._selection_type == "sample":
                supp_samples = Labels(
                    names=block.samples.names, values=block.samples.values[supp_mask]
                )
                supp_properties = Labels.single()

            supp_vals = np.zeros(
                [len(supp_samples), len(supp_properties)], dtype=np.int32
            )
            support_blocks.append(
                TensorBlock(
                    values=supp_vals,
                    samples=supp_samples,
                    components=[],
                    properties=supp_properties,
                )
            )

            if self._selector_class == skmatter._selection._FPS:
                # Build the Hausdorff TensorMap, only for FPS. In this case we want the
                # mask to be a list of int such that the samples/properties are
                # reordered according to the Hausdorff distance.
                haus_mask = selector.get_support(indices=True, ordered=True)
                if self._selection_type == "feature":
                    haus_samples = Labels.single()
                    haus_properties = Labels(
                        names=block.properties.names,
                        values=block.properties.values[haus_mask],
                    )
                elif self._selection_type == "sample":
                    haus_samples = Labels(
                        names=block.samples.names,
                        values=block.samples.values[haus_mask],
                    )
                    haus_properties = Labels.single()

                haus_vals = selector.hausdorff_at_select_[haus_mask].reshape(
                    len(haus_samples), len(haus_properties)
                )
                hausdorff_blocks.append(
                    TensorBlock(
                        values=haus_vals,
                        samples=haus_samples,
                        components=[],
                        properties=haus_properties,
                    )
                )

        self._support = TensorMap(X.keys, support_blocks)
        if self._selector_class == skmatter._selection._FPS:
            self._select_distance = TensorMap(X.keys, hausdorff_blocks)

        return self

    def transform(self, X: TensorMap) -> TensorMap:
        """
        Reduce X to the selected features.

        :param X: the input tensor.
        :returns: the selected subset of the input.
        """
        blocks = []
        for key, block in X.items():
            block_support = self.support.block(key)

            if self._selection_type == "feature":
                new_block = metatensor.slice_block(
                    block, "properties", block_support.properties
                )
            elif self._selection_type == "sample":
                new_block = metatensor.slice_block(
                    block, "samples", block_support.samples
                )
            blocks.append(new_block)

        return TensorMap(X.keys, blocks)

    def fit_transform(self, X: TensorMap, warm_start: bool = False) -> TensorMap:
        """
        Fit to data, then transform it.

        :param X: TensorMap of the training vectors.
        :param warm_start: bool, whether the fit should continue after having already
            run, after increasing `n_to_select`. Assumes it is called with the same X.
        """
        return self.fit(X, warm_start=warm_start).transform(X)
