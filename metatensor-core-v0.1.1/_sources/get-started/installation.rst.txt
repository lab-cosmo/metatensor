Installing metatensor
=====================

Metatensor is available fro multiple programming languages, and how to install
and use it will depend on the programming language you are using.


If you want to build metatensor from the source code you will need a Rust
compiler in addition to any language specific compiler. You can install Rust
using `rustup <https://rustup.rs/>`_ or the package manager of your operating
system.


.. _install-python-lib:

Installing the Python library
-----------------------------

For building and using the Python package clone the repository using `git
<https://git-scm.com>`_ and install metatensor using `pip
<https://pip.pypa.io>`_.

From source:

.. code-block:: bash

    # Make sure you are using the latest version of pip
    pip install --upgrade pip

    git clone https://github.com/lab-cosmo/metatensor
    cd metatensor
    pip install .

    # alternatively, the same thing in a single command
    pip install git+https://github.com/lab-cosmo/metatensor


Metatensor is also provided as prebuilt wheel which avoids the intermediate step
of building the package with a Rust compiler from the source code.

.. code-block:: bash

    pip install --upgrade pip
    pip install --extra-index-url https://luthaf.fr/temporary-wheels/ metatensor

.. _install-c-lib:

Installing the C and C++ library
--------------------------------

This installs a C-compatible shared library that can also be called from C++, as
well as CMake files that can be used with ``find_package(metatensor)``.

.. code-block:: bash

    git clone https://github.com/lab-cosmo/metatensor
    cd metatensor/metatensor-core
    mkdir build && cd build
    cmake ..
    # configure cmake if needed
    cmake --build . --target install

The build and installation can be configures with a few cmake options, using
``-D<OPTION>=<VALUE>`` on the cmake command line, or one of the cmake GUI
(``cmake-gui`` or ``ccmake``). Here are the main configuration options:

+---------------------------------------+-----------------------------------------------+----------------+
| Option                                | Description                                   | Default        |
+=======================================+===============================================+================+
| CMAKE_BUILD_TYPE                      | Type of build: debug or release               | release        |
+---------------------------------------+-----------------------------------------------+----------------+
| CMAKE_INSTALL_PREFIX                  | Prefix in which the library will be installed | ``/usr/local`` |
+---------------------------------------+-----------------------------------------------+----------------+
| INCLUDE_INSTALL_DIR                   | Path relative to ``CMAKE_INSTALL_PREFIX``     | ``include``    |
|                                       |  where the headers will be installed          |                |
+---------------------------------------+-----------------------------------------------+----------------+
| LIB_INSTALL_DIR                       | Path relative to ``CMAKE_INSTALL_PREFIX``     | ``lib``        |
|                                       | where the shared library will be installed    |                |
+---------------------------------------+-----------------------------------------------+----------------+
| BUILD_SHARED_LIBS                     | Default to installing and using a shared      | ON             |
|                                       | library instead of a static one               |                |
+---------------------------------------+-----------------------------------------------+----------------+
| METATENSOR_INSTALL_BOTH_STATIC_SHARED | Install both the shared and static version    | ON             |
|                                       | of the library                                |                |
+---------------------------------------+-----------------------------------------------+----------------+


Using the Rust library
----------------------

Add the following to your project ``Cargo.toml``

.. code-block:: toml

    [dependencies]
    metatensor = {git = "https://github.com/lab-cosmo/metatensor.git"}

.. _install-torch-script:

Installing the TorchScript bindings
-----------------------------------

For usage from Python
^^^^^^^^^^^^^^^^^^^^^

Building from source:

.. code-block:: bash

    # Make sure you are using the latest version of pip
    pip install --upgrade pip

    git clone https://github.com/lab-cosmo/metatensor
    cd metatensor/python/metatensor-torch
    pip install .

    # alternatively, the same thing in a single command
    pip install git+https://github.com/lab-cosmo/metatensor#subdirectory=python/metatensor-torch


For usage from C++
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/lab-cosmo/metatensor
    cd metatensor/metatensor-torch
    mkdir build && cd build
    cmake ..
    # configure cmake if needed
    cmake --build . --target install

Compiling the TorchScript bindings requires you to manually install some of the
dependencies:

- the C++ part of PyTorch, which you can install `on it's own
  <https://pytorch.org/get-started/locally/>`_. You can also use the
  installation that comes with a Python installation by adding the output of the
  command below to ``CMAKE_PREFIX_PATH``:

  .. code-block:: bash

    python -c "import torch; print(torch.utils.cmake_prefix_path)"

- :ref:`the C++ interface of metatensor <install-c-lib>`

If any of these dependencies is not in a standard location, you should specify
the installation directory when configuring cmake with ``CMAKE_PREFIX_PATH``.
Other useful configuration options are:

+--------------------------------------+-----------------------------------------------+----------------+
| Option                               | Description                                   | Default        |
+======================================+===============================================+================+
| CMAKE_BUILD_TYPE                     | Type of build: debug or release               | release        |
+--------------------------------------+-----------------------------------------------+----------------+
| CMAKE_INSTALL_PREFIX                 | Prefix in which the library will be installed | ``/usr/local`` |
+--------------------------------------+-----------------------------------------------+----------------+
| CMAKE_PREFIX_PATH                    | ``;``-separated list of path where CMake will |                |
|                                      | search for dependencies. This list should     |                |
|                                      | include the path to metatensor and torch      |                |
+--------------------------------------+-----------------------------------------------+----------------+
