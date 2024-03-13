.. _cxx-api-torch:

TorchScript C++ API reference
=============================

To use metatensor's TorchScript C++ API from your own code, you should ``#include
<metatensor/torch.hpp>`` and use the functions and classes documented below. You
should then link with libtorch, and the both the ``metatensor_torch`` and
``metatensor`` shared libraries.

The easiest way to link to everything and set the right compiler flags is to use
CMake after :ref:`installing metatensor's C++ Torch interface <install-torch-cxx>`:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.16)
    project(my-project CXX)

    find_package(metatensor_torch)

    # add executables/libraries
    add_executable(my-exe exe-source.cpp)
    add_library(my-lib lib-source.cpp)

    # Link to metatensor, this makes the header accessible and link to the right
    # libraries.
    target_link_libraries(my-exe metatensor_torch)
    target_link_libraries(my-lib metatensor_torch)


.. toctree::
    :maxdepth: 1

    tensor
    block
    labels
    miscellaneous
