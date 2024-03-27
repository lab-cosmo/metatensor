.. _c-api-reference:

C API reference
===============

.. note::

  This is the documentation for ``metatensor-core`` version
  |metatensor-core-version|. For other versions, look in the following pages:

  .. grid::
    :margin: 0 0 0 0

    .. grid-item-card:: Version 0.1.0
      :link: https://lab-cosmo.github.io/metatensor/metatensor-core-v0.1.0/reference/c/index.html
      :link-type: url
      :columns: 12 6 3 3
      :text-align: center
      :class-body: sd-p-2
      :class-title: sd-mb-0

    .. grid-item-card:: Version 0.1.1
      :link: https://lab-cosmo.github.io/metatensor/metatensor-core-v0.1.1/reference/c/index.html
      :link-type: url
      :columns: 12 6 3 3
      :text-align: center
      :class-body: sd-p-2
      :class-title: sd-mb-0


``metatensor`` offers a C API that can be called from any language able to call
C functions (in particular, this includes Python, Fortran with ``iso_c_env``,
C++, and most languages used nowadays). Convenient wrappers of the C API are
also provided for :ref:`Python <python-api-reference>` users.

The C API is implemented in Rust. You can use these functions in your own code
by :ref:`installing the corresponding shared library and header <install-c>`,
and then including ``metatensor.h`` and linking with ``-lmetatensor``.
Alternatively, we provide a cmake package config file, allowing you to do use
``metatensor`` like this (after installation):

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.16)
    project(my-project C)

    find_package(metatensor)

    # add executables/libraries
    add_executable(my-exe exe-source.c)
    add_library(my-lib lib-source.c)

    # Link to metatensor, this makes the header accessible and link to the right
    # libraries.
    #
    # The `metatensor` target will be an alias for `metatensor::shared`
    # or `metatensor::static` depending how you've installed the code.
    target_link_libraries(my-exe metatensor)
    target_link_libraries(my-lib metatensor)

    # alternatively, you can explicitly use the static or shared build of
    # metatensor. Unless you have a very specific need for a static build, we
    # recommend using the shared version of metatensor: this will allow to pass
    # data from your code to any other code using metatensor.

    # target_link_libraries(my-exe metatensor::shared)
    # target_link_libraries(my-exe metatensor::static)

The functions and types provided in ``metatensor.h`` can be grouped in five
main groups:

.. toctree::
    :maxdepth: 1

    tensor
    labels
    block
    data
    misc
