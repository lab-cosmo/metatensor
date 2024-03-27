.. _cxx-api-reference:

C++ API reference
=================

Metatensor offers a C++ API, built on top of the :ref:`C API <c-api-reference>`.
You can the provided classes and functions in your own code by :ref:`installing
the corresponding shared library and header <install-c-lib>`, and then including
``metatensor.hpp`` and linking with ``-lmetatensor``. Alternatively, we provide a
cmake package config file, allowing you to do use metatensor like this (after
installation):

.. code-block:: cmake

    find_package(metatensor)

    # add executables/libraries
    add_executable(MyExecutable my_sources.cxx)
    add_library(MyLibrary my_sources.cxx)

    # Link to metatensor, this makes the header accessible and link to the right
    # libraries.
    #
    # The `metatensor` target will be an alias for `metatensor::shared`
    # or `metatensor::static` depending how you've installed the code.
    target_link_libraries(MyExecutable metatensor)
    target_link_libraries(MyLibrary metatensor)

    # alternatively, you can explicitly use the static or shared build of
    # metatensor. Unless you have a very specific need for a static build, we
    # recommend using the shared version of metatensor: this will allow to pass
    # data from your code to any other code using metatensor.

    # target_link_libraries(MyExecutable metatensor::shared)
    # target_link_libraries(MyExecutable metatensor::static)

The functions and types provided in ``metatensor.hpp`` can be grouped as follow:

.. toctree::
    :maxdepth: 1

    tensor
    labels
    block
    data
    misc
    CHANGELOG.md
