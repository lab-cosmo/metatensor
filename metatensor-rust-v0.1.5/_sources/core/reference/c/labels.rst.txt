Labels
======

.. doxygenstruct:: mts_labels_t
    :members:

The following functions operate on :c:type:`mts_labels_t`:

- :c:func:`mts_labels_create`: create the Rust-side data for the labels
- :c:func:`mts_labels_clone`: increment the reference count of the Rust-side data
- :c:func:`mts_labels_free`: decrement the reference count of the Rust-side data,
  and free the data when it reaches 0
- :c:func:`mts_labels_position`: get the position of an entry in the labels
- :c:func:`mts_labels_union`: get the union of two labels
- :c:func:`mts_labels_intersection`: get the intersection of two labels
- :c:func:`mts_labels_set_user_data`: store some data inside the labels for later retrieval
- :c:func:`mts_labels_user_data`: retrieve data stored earlier in the labels

--------------------------------------------------------------------------------

.. doxygenfunction:: mts_labels_create

.. doxygenfunction:: mts_labels_clone

.. doxygenfunction:: mts_labels_free

.. doxygenfunction:: mts_labels_position

.. doxygenfunction:: mts_labels_union

.. doxygenfunction:: mts_labels_intersection

.. doxygenfunction:: mts_labels_set_user_data

.. doxygenfunction:: mts_labels_user_data
