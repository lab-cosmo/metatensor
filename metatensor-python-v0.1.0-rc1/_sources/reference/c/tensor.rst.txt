TensorMap
=========

.. doxygentypedef:: mts_tensormap_t

The following functions operate on :c:type:`mts_tensormap_t`:

- :c:func:`mts_tensormap`: create new tensor map
- :c:func:`mts_tensormap_copy`: copy existing tensor maps
- :c:func:`mts_tensormap_free`: free allocated tensor maps
- :c:func:`mts_tensormap_keys`: get the keys defined in a tensor map as :c:struct:`mts_labels_t`
- :c:func:`mts_tensormap_block_by_id`: get a :c:struct:`mts_block_t` in a tensor map from its index
- :c:func:`mts_tensormap_blocks_matching`: get a list of block indexes matching a selection
- :c:func:`mts_tensormap_keys_to_samples`: move entries from keys to sample labels
- :c:func:`mts_tensormap_keys_to_properties`: move entries from keys to properties labels
- :c:func:`mts_tensormap_components_to_properties`: move entries from component labels to properties labels


--------------------------------------------------------------------------------

.. doxygenfunction:: mts_tensormap

.. doxygenfunction:: mts_tensormap_copy

.. doxygenfunction:: mts_tensormap_free

.. doxygenfunction:: mts_tensormap_keys

.. doxygenfunction:: mts_tensormap_block_by_id

.. doxygenfunction:: mts_tensormap_blocks_matching

.. doxygenfunction:: mts_tensormap_keys_to_samples

.. doxygenfunction:: mts_tensormap_keys_to_properties

.. doxygenfunction:: mts_tensormap_components_to_properties
