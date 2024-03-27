.. _torch-cxx-api-reference:

TorchScript C++ API reference
=============================

To use metatensor's TorchScript C++ API from your own code, you should ``#include
<metatensor/torch.hpp>`` and use the functions and classes documented below. You
should then link with libtorch, and the both the ``metatensor_torch`` and
``metatensor`` shared libraries.

The easiest way to link to everything and set the right compiler flags is to use
CMake after :ref:`installing metatensor's C++ Torch interface
<install-torch-script>`:

.. code-block:: cmake

    # This will find metatensor_torch, metatensor and Torch on your system
    find_package(metatensor_torch)

    # add executables/libraries
    add_executable(MyExecutable my_sources.cxx)
    add_library(MyLibrary my_sources.cxx)

    # Link to metatensor_torch, this makes the headers accessible
    # and link to the right libraries
    target_link_libraries(MyExecutable metatensor_torch)
    target_link_libraries(MyLibrary metatensor_torch)

.. toctree::
    :maxdepth: 1

    tensor
    block
    labels
    serialization
