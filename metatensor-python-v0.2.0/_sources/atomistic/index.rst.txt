.. _atomistic-models:

Atomistic applications
======================

While metatensor is a generic sparse data container able to store data and
metadata for multiple scientific fields, it comes from the field of atomistic
machine learning and as such offer some additional facilities for defining and
using machine learning models applied to atomistic systems.

Metatensor provides tools to build your own models (in the form of
:ref:`operations <python-api-operations>`), define new models architectures and
export models you just train to use them in arbitrary simulation engines. If you
want to train existing architectures with new data or re-use existing trained
models, look into the (work in progress!) metatensor-models_ project instead.

.. _metatensor-models: https://github.com/lab-cosmo/metatensor-models

.. grid::

    .. grid-item-card:: 💡 Tutorials
        :link: atomistic-tutorials
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Learn how to define your own models using metatensor, and how to use
        these models to run simulation in various simulation engines.

    .. grid-item-card:: 📋 Standard models outputs
        :link: atomistic-models-outputs
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Understand the metadata convention for specific models outputs, such as
        the potential energy.

    .. grid-item-card:: |Python-16x16| Python API reference
        :link: python-api-atomistic-models
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the classes and functions related to
        atomistic models in Python.

        +++
        Documentation for version |metatensor-torch-version|

    .. grid-item-card:: |Cxx-16x16| C++ API reference
        :link: cxx-api-atomistic-models
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the classes and functions related to
        atomistic models in C++.

        +++
        Documentation for version |metatensor-torch-version|

.. toctree::
    :maxdepth: 2
    :hidden:

    reference/index
    outputs
    ../examples/atomistic/index


Overview
--------

All the model facilities in metatensor are based on PyTorch and in particular
TorchScript; and as such are part of the ``metatensor-torch`` package. This
allow users to define new models with Python code (as a custom
:py:class:`torch.nn.Module` instance), train the models from Python, and export
them to TorchScript. The exported model can then be loaded into a C++ simulation
engine such as LAMMPS, GROMACS, *etc.* and executed without relying on a Python
installation.

Metatensor provides code for using atomistic systems as input of a machine
learning model with :py:class:`metatensor.torch.atomistic.System`, and exporting
trained models with
:py:class:`metatensor.torch.atomistic.MetatensorAtomisticModel`. Such models can
make predictions for various properties of the atomistic system, and return them
as a dictionary of :py:class:`metatensor.torch.TensorMap`, one such tensor map
for each property (i.e. energy, atomic charges, dipole, electronic density,
chemical shielding, *etc.*)
