.. _c-api-reference:

C API reference
===============

``metatensor`` offers a C API that can be called from any language able to call
C functions (in particular, this includes Python, Fortran with ``iso_c_env``,
C++, and most languages used nowadays). Convenient wrappers of the C API are
also provided for :ref:`Python <python-api-reference>` users.

The C API is implemented in Rust. You can use these functions in your own code
by :ref:`installing the corresponding shared library and header
<install-c-lib>`, and then including ``metatensor.h`` and linking with
``-lmetatensor``. Alternatively, we provide a cmake package config file, allowing
you to do use ``metatensor`` like this (after installation):

.. code-block:: cmake

    find_package(metatensor)

    # add executables/libraries
    add_executable(MyExecutable my_sources.c)
    add_library(MyLibrary my_sources.c)

    # Links metatensor with MyExecutable, this makes the header accessible
    target_link_libraries(MyExecutable metatensor)

    # alternatively, use the static build of metatensor
    # target_link_libraries(MyExecutable metatensor::static)

The functions and types provided in ``metatensor.h`` can be grouped in five
main groups:

.. toctree::
    :maxdepth: 1

    tensor
    labels
    block
    data
    misc
    CHANGELOG.md
