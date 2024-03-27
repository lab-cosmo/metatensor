.. _atomistic-models:

Atomistic Models
================

While metatensor is a generic sparse data container able to store data and
metadata for multiple scientific fields, it comes from the field of atomistic
machine learning and as such offer some additional facilities for defining and
using machine learning models applied to atomistic systems.

Metatensor provides tools to build your own models (in the form of
:ref:`operations <python-api-operations>`), define new models architectures and
export models you just train to use them in arbitrary simulation engines. If you
want to train existing architectures with new data or re-use existing trained
models, look into the (work in progress) metatensor-models_ project instead!

.. _metatensor-models: https://github.com/lab-cosmo/metatensor-models

--------------------------------------------------------------------------------

All the model facilities in metatensor are based on PyTorch and in particular
TorchScript. This allow users to define new models with Python code (as a custom
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

This part of the documentation contains the full :ref:`Python
<python-api-atomistic-models>` and :ref:`C++ <cxx-api-atomistic-models>` API
references for atomistic models. It also defines how some :ref:`specific
properties <atomistic-models-outputs>` should be structured if they are returned
by a model.

.. toctree::
    :maxdepth: 2

    reference/index
    outputs
    ../examples/atomistic/index
