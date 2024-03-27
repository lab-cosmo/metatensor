TensorBlock
===========

.. doxygentypedef:: mts_block_t

The following functions operate on :c:type:`mts_block_t`:

- :c:func:`mts_block`: create new blocks
- :c:func:`mts_block_copy`: copy existing blocks
- :c:func:`mts_block_free`: free allocated blocks
- :c:func:`mts_block_labels`: get one of the :c:struct:`mts_labels_t` associated with this block
- :c:func:`mts_block_data`: get one of the :c:struct:`mts_array_t` associated with this block
- :c:func:`mts_block_add_gradient`: add gradient data to this block
- :c:func:`mts_block_gradients_list`: get the list of gradients in this block

--------------------------------------------------------------------------------

.. doxygenfunction:: mts_block

.. doxygenfunction:: mts_block_copy

.. doxygenfunction:: mts_block_free

.. doxygenfunction:: mts_block_labels

.. doxygenfunction:: mts_block_data

.. doxygenfunction:: mts_block_add_gradient

.. doxygenfunction:: mts_block_gradients_list
