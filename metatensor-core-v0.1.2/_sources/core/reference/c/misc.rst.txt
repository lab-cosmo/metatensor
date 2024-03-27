Miscellaneous
=============

.. doxygenfunction:: mts_version

Error handling
^^^^^^^^^^^^^^

.. doxygenfunction:: mts_last_error

.. doxygenfunction:: mts_disable_panic_printing

.. doxygentypedef:: mts_status_t

.. doxygendefine:: MTS_SUCCESS

.. doxygendefine:: MTS_INVALID_PARAMETER_ERROR

.. doxygendefine:: MTS_BUFFER_SIZE_ERROR

.. doxygendefine:: MTS_INTERNAL_ERROR


Serialization
^^^^^^^^^^^^^

Tensors
-------

- :c:func:`mts_tensormap_save`: serialize and save a ``mts_tensormap_t`` to a file
- :c:func:`mts_tensormap_load`: load serialized ``mts_tensormap_t`` from a file
- :c:func:`mts_tensormap_save_buffer`: serialize and save a ``mts_tensormap_t``
  to a in-memory buffer
- :c:func:`mts_tensormap_load_buffer`: load serialized ``mts_tensormap_t`` from
  a in-memory buffer

.. doxygenfunction:: mts_tensormap_load

.. doxygenfunction:: mts_tensormap_save

.. doxygenfunction:: mts_tensormap_load_buffer

.. doxygenfunction:: mts_tensormap_save_buffer


.. doxygentypedef:: mts_create_array_callback_t

.. doxygentypedef:: mts_realloc_buffer_t


Labels
-------

- :c:func:`mts_labels_save`: serialize and save a ``mts_labels_t`` to a file
- :c:func:`mts_labels_load`: load serialized ``mts_labels_t`` from a file
- :c:func:`mts_labels_save_buffer`: serialize and save a ``mts_labels_t``
  to a in-memory buffer
- :c:func:`mts_labels_load_buffer`: load serialized ``mts_labels_t`` from
  a in-memory buffer

- :c:func:`mts_tensormap_load`: create the Rust-side data for the labels

.. doxygenfunction:: mts_labels_load

.. doxygenfunction:: mts_labels_save

.. doxygenfunction:: mts_labels_load_buffer

.. doxygenfunction:: mts_labels_save_buffer
