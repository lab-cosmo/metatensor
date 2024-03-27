Miscellaneous
=============

Error handling
^^^^^^^^^^^^^^

.. doxygenclass:: metatensor::Error
    :members:


N-dimensional arrays
^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: metatensor::NDArray
    :members:


``TensorMap`` serialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: metatensor::io::save(const std::string& path, const TensorMap& tensor)

.. doxygenfunction:: metatensor::io::save_buffer(const TensorMap& tensor)

.. doxygenfunction:: metatensor::io::load

.. doxygenfunction:: metatensor::io::load_buffer(const uint8_t* buffer, size_t buffer_count, mts_create_array_callback_t create_array)

.. doxygenfunction:: metatensor::io::load_buffer(const Buffer& buffer, mts_create_array_callback_t create_array)

.. doxygenfunction:: metatensor::details::default_create_array

``Labels`` serialization
^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenfunction:: metatensor::io::save(const std::string& path, const Labels& labels)

.. doxygenfunction:: metatensor::io::save_buffer(const Labels& labels)

.. doxygenfunction:: metatensor::io::load_labels

.. doxygenfunction:: metatensor::io::load_labels_buffer(const uint8_t* buffer, size_t buffer_count)

.. doxygenfunction:: metatensor::io::load_labels_buffer(const Buffer& buffer)
