.. _installation:

Installation
============

Metatensor is available fro multiple programming languages, and how to install
and use it will depend on the programming language you are using.

.. tab-set::

    .. tab-item:: Python
        :name: install-python

        Metatensor is split into multiple Python packages, each providing a
        subset of the functionality in a modular way. Most users will want to
        install all the packages, but installing individual packages is also
        supported.

        The simplest way to install metatensor is to use `pip`_, and run the
        following commands:

        .. code-block:: bash

            # Make sure you are using the latest version of pip
            pip install --upgrade pip

            pip install metatensor


        This will install the ``metatensor`` package, as well as
        ``metatensor-core``, and ``metatensor-operations``. We provide pre-built
        version of ``metatensor-core`` for Linux (x64), Windows (x64) and macOS
        (x64 and ARM64). The other packages are pure Python package that should
        install on any operating system.

        You can install any of these sub-packages one by one with

        .. code-block:: bash

            pip install metatensor-core
            pip install metatensor-operations

        **TorchScript integration**: If you also want to use the TorchScript
        bindings to metatensor, use this instead (see :ref:`this page
        <install-torch>` for more information).

        .. code-block:: bash

            pip install metatensor[torch]


        **Experimental packages**: ``metatensor-learn`` is still experimental
        and only the development version can be installed (see :ref:`this page
        <dev-install-python>` for more information):

        .. code-block:: bash

            pip install git+https://github.com/lab-cosmo/metatensor#subdirectory=python/metatensor-learn


    .. tab-item:: C and C++
        :name: install-c

        The main classes of metatensor are available from C and C++, and can be
        installed as a C-compatible shared library, with a C++ header wrapping
        the C library with a modern C++ API. The installation also comes with
        the required files for `CMake`_ integration, allowing you to use
        metatensor in your own CMake project with ``find_package(metatensor)``.

        To build and install the code, you'll need to find the latest release of
        ``metatensor-core`` on `GitHub releases
        <https://github.com/lab-cosmo/metatensor/releases>`_, and download the
        corresponding ``metatensor-core-cxx`` file in the release assets.

        You will also need to install a rust compiler and ``cargo`` either with
        `rustup`_ or the package manager of your operating system. Then, you can
        run the following commands:

        .. code-block:: bash

            cmake -E tar xf metatensor-core-cxx-*.tar.gz
            cd metatensor-core-cxx-*
            mkdir build && cd build

            # configure cmake here if needed
            cmake ..

            # build and install the code
            cmake --build . --target install

        The build and installation can be configures with a few CMake options,
        using ``-D<OPTION>=<VALUE>`` on the cmake command line, or one of the
        cmake GUI (``cmake-gui`` or ``ccmake``). Here are the main configuration
        options:

        +-------------------------------------------+-----------------------------------------------+----------------+
        | Option                                    | Description                                   | Default        |
        +===========================================+===============================================+================+
        | ``CMAKE_BUILD_TYPE``                      | Type of build: debug or release               | release        |
        +-------------------------------------------+-----------------------------------------------+----------------+
        | ``CMAKE_INSTALL_PREFIX``                  | Prefix in which the library will be installed | ``/usr/local`` |
        +-------------------------------------------+-----------------------------------------------+----------------+
        | ``INCLUDE_INSTALL_DIR``                   | Path relative to ``CMAKE_INSTALL_PREFIX``     | ``include``    |
        |                                           | where the headers will be installed           |                |
        +-------------------------------------------+-----------------------------------------------+----------------+
        | ``LIB_INSTALL_DIR``                       | Path relative to ``CMAKE_INSTALL_PREFIX``     | ``lib``        |
        |                                           | where the shared library will be installed    |                |
        +-------------------------------------------+-----------------------------------------------+----------------+
        | ``BUILD_SHARED_LIBS``                     | Default to installing and using a shared      | ON             |
        |                                           | library instead of a static one               |                |
        +-------------------------------------------+-----------------------------------------------+----------------+
        | ``METATENSOR_INSTALL_BOTH_STATIC_SHARED`` | Install both the shared and static version    | ON             |
        |                                           | of the library                                |                |
        +-------------------------------------------+-----------------------------------------------+----------------+



    .. tab-item:: Rust
        :name: install-rust

        To use metatensor from Rust, you can add following to your project
        ``Cargo.toml``

        .. code-block:: toml

            [dependencies]
            metatensor = "0.1"


        We have one feature that can be enabled with cargo: ``static``, which
        forces the code to use the static build of ``metatensor-core`` instead
        of a shared build. It is disabled by default. Enabling it will mean that
        your code might not be able to share data with other metatensor-enabled
        programs if they are using a different version of metatensor. For
        example if you are working on a Python extension with `PyO3
        <https://pyo3.rs/>`_, you should not use the ``static`` feature and
        instead have the code load the same shared library as the ``metatensor``
        Python package.


    .. tab-item:: TorchScript Python
        :name: install-torch

        The TorchScript bindings to metatensor are accessible in Python in the
        ``metatensor-torch`` package. You can install this at the same time you
        install the rest of metatensor with

        .. code-block:: bash

            # Make sure you are using the latest version of pip
            pip install --upgrade pip

            pip install metatensor[torch]

        or as a standalone package with

        .. code-block:: bash

            pip install metatensor-torch

        Due to the way PyTorch itself is structured and distributed, we can not
        provide pre-compiled versions of metatensor-torch on `PyPI
        <https://pypi.org/>`_, but only a source distribution that will be
        compiled on your machine. This local compilation will require a couple
        of additional dependencies.

        - a modern C++ compiler, able to handle C++17, such as:
            - gcc version 7 or above;
            - clang version 5 or above;
            - Microsoft Visual C++ (MSVC) compiler, version 19 (2015) or above.
        - if you want to use the CUDA version of PyTorch, you'll also need the
          `CUDA toolkit <https://developer.nvidia.com/cuda-toolkit>`_, including
          the NVIDIA compiler.

        By default, PyTorch is installed with CUDA support on Linux, even if you
        do not have a CUDA-compatible GPU, and will search for a CUDA toolkit
        when building extension (such as ``metatensor-torch``). If you don't
        want to install the CUDA toolkit in this case, you can use the CPU-only
        version of PyTorch with

        .. code-block:: bash

            pip install --extra-index-url https://download.pytorch.org/whl/cpu metatensor[torch]

        A similar index URL can be used to install the ROCm (AMD GPU) version of
        PyTorch, please refer to the `corresponding documentation
        <https://pytorch.org/get-started/locally/>`_.



    .. tab-item:: TorchScript C++
        :name: install-torch-cxx

        The TorchScript bindings to metatensor are also available as a C++
        library, which can be integrated in non-Python software (such as
        simulation engines) to use custom metatensor models directly in the
        software without relying on a Python interpreter. The code is installed
        as a shared library which register itself with torch when loaded, the
        corresponding header files and a CMake integration allowing you to use
        metatensor-torch in your code code with
        ``find_package(metatensor_torch)``.

        To build and install the code, you'll need to find the latest release of
        ``metatensor-torch`` on `GitHub releases
        <https://github.com/lab-cosmo/metatensor/releases>`_, and download the
        corresponding ``metatensor-torch-cxx`` file in the release assets. Then,
        you can run the following commands:

        .. code-block:: bash

            cmake -E tar xf metatensor-torch-cxx-*.tar.gz
            cd metatensor-torch-cxx-*
            mkdir build && cd build

            # configure cmake here if needed
            cmake ..

            # build and install the code
            cmake --build . --target install

        You will have to to manually install some of the dependencies of
        metatensor-torch yourself to compile this code, and any of the
        dependencies is not in a standard location, specify the installation
        directory when configuring cmake with ``CMAKE_PREFIX_PATH``. The
        following dependencies might have to be installed beforehand:

        - :ref:`the C++ interface <install-c>` of metatensor.
        - the C++ part of PyTorch, which you can install `on it's own
          <https://pytorch.org/get-started/locally/>`_. You can also use the
          same library as the Python version of torch by adding the output of
          the command below to ``CMAKE_PREFIX_PATH``:

          .. code-block:: bash

              python -c "import torch; print(torch.utils.cmake_prefix_path)"


        +--------------------------------------+-----------------------------------------------+----------------+
        | Option                               | Description                                   | Default        |
        +======================================+===============================================+================+
        | ``CMAKE_BUILD_TYPE``                 | Type of build: debug or release               | release        |
        +--------------------------------------+-----------------------------------------------+----------------+
        | ``CMAKE_INSTALL_PREFIX``             | Prefix in which the library will be installed | ``/usr/local`` |
        +--------------------------------------+-----------------------------------------------+----------------+
        | ``CMAKE_PREFIX_PATH``                | ``;``-separated list of path where CMake will |                |
        |                                      | search for dependencies. This list should     |                |
        |                                      | include the path to metatensor and torch      |                |
        +--------------------------------------+-----------------------------------------------+----------------+


Installing a development version
--------------------------------

Metatensor is developed on `GitHub <https://github.com/lab-cosmo/metatensor>`_.
If you want to install a development version of the code, you will need `git
<https://git-scm.com>`_ to fetch the latest version of the code. You will also
need a Rust compiler on top of any language specific compiler. You can install
Rust using `rustup`_ or the package manager of your operating system.


.. tab-set::
    .. tab-item:: Python
        :name: dev-install-python

        You can install a development version of all the metatensor sub-packages
        with:

        .. code-block:: bash

            # Make sure you are using the latest version of pip
            pip install --upgrade pip

            git clone https://github.com/lab-cosmo/metatensor
            cd metatensor
            pip install .

            # alternatively, the same thing in a single command
            pip install git+https://github.com/lab-cosmo/metatensor

        You can also install a single sub-package at the time with

        .. code-block:: bash

            git clone https://github.com/lab-cosmo/metatensor
            cd metatensor
            pip install ./python/metatensor-core
            pip install ./python/metatensor-operations
            pip install ./python/metatensor-learn
            pip install ./python/metatensor-torch

            # alternatively, the same thing in a single command
            pip install git+https://github.com/lab-cosmo/metatensor#subdirectory=python/metatensor-core
            pip install git+https://github.com/lab-cosmo/metatensor#subdirectory=python/metatensor-operations
            pip install git+https://github.com/lab-cosmo/metatensor#subdirectory=python/metatensor-learn
            pip install git+https://github.com/lab-cosmo/metatensor#subdirectory=python/metatensor-torch


    .. tab-item:: C and C++
        :name: dev-install-c

        You can install the development version of metatensor with the following
        (the same :ref:`cmake configuration options <install-c>` are available):

        .. code-block:: bash

            git clone https://github.com/lab-cosmo/metatensor
            cd metatensor/metatensor-core

            mkdir build && cd build

            # configure cmake here if needed
            cmake ..

            # build and install the code
            cmake --build . --target install


    .. tab-item:: Rust
        :name: dev-install-rust

        Add the following to your project ``Cargo.toml``

        .. code-block:: toml

            [dependencies]
            metatensor = {git = "https://github.com/lab-cosmo/metatensor.git"}



    .. tab-item:: TorchScript Python
        :name: dev-install-torch

        You can install a development version of the TorchScript bindings to
        metatensor with:

        .. code-block:: bash

            # Make sure you are using the latest version of pip
            pip install --upgrade pip

            git clone https://github.com/lab-cosmo/metatensor
            cd metatensor
            pip install .[torch]

            # alternatively, the same thing in a single command
            pip install "metatensor[torch] @ git+https://github.com/lab-cosmo/metatensor"


        If you want to install and update only the ``metatensor-torch`` package,
        you can do the following:

        .. code-block:: bash

            git clone https://github.com/lab-cosmo/metatensor
            cd metatensor
            pip install ./python/metatensor-torch

            # alternatively, the same thing in a single command
            pip install git+https://github.com/lab-cosmo/metatensor#subdirectory=python/metatensor-torch


    .. tab-item:: TorchScript C++
        :name: dev-install-torch-cxx

        You can install the development version of metatensor with the following
        (the same :ref:`cmake configuration options <install-torch-cxx>` are
        available):

        .. code-block:: bash

            git clone https://github.com/lab-cosmo/metatensor
            cd metatensor/metatensor-torch
            mkdir build && cd build

            # configure cmake here if needed
            cmake ..

            # build and install the code
            cmake --build . --target install



.. _pip: https://pip.pypa.io
.. _CMake: https://cmake.org
.. _rustup: https://rustup.rs
