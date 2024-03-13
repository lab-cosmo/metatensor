.. _goals:

Metatensor's goals
==================

At its core, ``metatensor`` provides tools to efficiently store and manipulate
sparse arrays and their associated metadata. You can learn more about this in
the :ref:`core classes overview <core-classes-overview>`. With the creation of
metatensor, we have three main use cases in mind:

1. provide an exchange format for the atomistic machine learning ecosystem,
   making different players in this ecosystem more interoperable with one
   another and enhancing collaboration: see :ref:`metatensor-goal-exchange`;
2. make it easier and faster to develop new machine learning representations,
   models and algorithms: see :ref:`metatensor-goal-models`;
3. run large scale simulations using machine learning interatomic potentials,
   with fully customizable potentials, directly defined by the researchers
   running the simulations: see :ref:`metatensor-goal-simulation`;

.. _metatensor-goal-exchange:

Exchanging data
^^^^^^^^^^^^^^^

First, metatensor is a format to exchange data between different libraries in the
atomistic machine learning ecosystem. There is currently an explosion of
libraries and tools for atomistic machine learning, implementing new
representation, new models, and advanced research methods. Unfortunately each
one of these libraries lives mostly separated from the others, resulting in a
lot of duplicated effort. With metatensor, we want to provide a way for these
libraries to communicate with one another, by giving everyone a *lingua franca*,
a way to share data and metadata.

.. figure:: /../static/images/goal-exchange.*
    :width: 400px
    :align: center

    Illustration of machine learning workflows, going from some input data to a
    prediction. Metatensor enables the creation of workflows mixing different
    libraries together in new ways.

This goal is enabled by multiple features of metatensor: first, metatensor allows
storing data coming from many different sources, without requiring to first
convert the data to a specific format. Currently, we support data stored inside
numpy arrays, torch tensor (including tensors on GPU or other accelerators), as
well as arbitrary user-defined C, C++, and Rust array types. A second part of
this goal is achieved by also storing metadata together with the data,
communicating between libraries exactly **what** is stored in the different
arrays. We also store both data and gradients of this data with respect to
arbitrary parameters together, enabling for example training of models using
energy, forces and virial. Finally, we also make sure that the data storage is
as efficient as possible and can exploit the inherent sparsity of atomistic
data, in particular in gradients.

As a developer a library in the atomistic machine learning ecosystem, you can
provide conversion functions to and from metatensor
:py:class:`metatensor.TensorMap` (either inside your own code or in a small
conversion package) to enable using your library in conjunction with the rest of
the metatensor ecosystem!

.. admonition:: libraries using metatensor

    The following libraries use metatensor either as input, output or both:

    - `rascaline <https://github.com/Luthaf/rascaline/>`_: a library computing
      physics-inspired representations of atomic systems, the computed
      representations are given in metatensor format;
    - `torch_spex <https://github.com/lab-cosmo/torch_spex/>`_: pure PyTorch
      implementation of spherical expansion representation, with GPU and
      learnable representations support, which outputs to metatensor format;
    - `metatensor-models <https://github.com/lab-cosmo/metatensor-models/>`_: an
      end-to-end training and evaluation library for models based on metatensor;
    - `Q-stack <https://github.com/lcmd-epfl/Q-stack/>`_: library of pre- and
      post-processing tasks for Quantum Machine Learning; can output some of its
      data in metatensor format;

.. _metatensor-goal-models:

Defining custom models
^^^^^^^^^^^^^^^^^^^^^^

The second objective of metatensor is to provide functionalities to be a tool
for developing new models. While it is possible to use metatensor to only
exchange data between libraries (and immediately convert everything to
library-specific formats); we also provide tools to operate directly on
metatensor data. This enable models to handle sparse data and have low memory
consumption; as well as keeping rich metadata around for easier debugging and
understanding of the model behavior.

One part of these tools is the set of low-level :ref:`operations
<metatensor-operations>` we provide as part of the Python interface to
metatensor. By using combining multiple operations, you can build custom machine
learning models, using data and representations coming from arbitrary
metatensor-compatible libraires in the ecosystem. Using these operations allow
you to keep your data in metatensor format across the whole ML pipeline;
ensuring the metadata is kept up to date with the data, and gradients are
automatically updated to stay consistent with the values.

Another part of these tools is the :ref:`learning utilities <metatensor-learn>`,
which provide high level building blocks for machine learning models, with API
similar to PyTorch or scikit-learn. These blocks enable you do define and train
models with a few lines of code and a familiar API.

.. warning::

    The learning utilities are still an early work in progress, with a lot more
    building blocks to be included.

.. table:: Where similar functionalities is provided by different packages
    :widths: auto

    +-------------+----------------------------------+---------------------------+----------------------------------------------+
    |  Package    | Core data class                  |  Operations               |  Machine learning models facilities          |
    +=============+==================================+===========================+==============================================+
    |  numpy      | :py:class:`numpy.ndarray`        | :py:func:`numpy.sum`      |  `scikit-learn`_                             |
    +-------------+----------------------------------+---------------------------+----------------------------------------------+
    |  torch      | :py:class:`torch.Tensor`         | :py:func:`torch.add`      | :py:class:`torch.nn.Module`,                 |
    |             |                                  |                           | :py:class:`torch.utils.data.Dataset`         |
    +-------------+----------------------------------+---------------------------+----------------------------------------------+
    |  metatensor | :py:class:`metatensor.TensorMap` | :py:func:`metatensor.add` | :py:class:`metatensor.learn.nn.ModuleMap`,   |
    |             |                                  |                           | :py:class:`metatensor.learn.Dataset`         |
    +-------------+----------------------------------+---------------------------+----------------------------------------------+


.. _scikit-learn: https://scikit-learn.org/

.. _metatensor-goal-simulation:

Running atomistic simulations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One particularly interesting class of machine learning model for atomistic
modelling is machine learning interatomic potentials (MLIPs). Using the
capacities provided by the first two goals of metatensor, researchers should be
able to created and train such MLIPs and customize various parts of the model.

The final objective of metatensor is to allow using these custom models inside
large scale molecular simulation engines. To do this, we integrate metatensor
with `TorchScript <https://pytorch.org/docs/stable/jit.html>`_, and use the
facilities of TorchScript to export the model from Python and then load and
execute it inside the simulation engine. Have a look at the
:ref:`atomistic-models` section for more information!

.. figure:: /../static/images/goal-simulations.*
    :width: 500px
    :align: center

    Different steps in the workflow of running simulations with metatensor.
    Defining a model, training a model and running simulations with it can be
    done by different users; and the same metatensor-based model can be used
    with multiple simulation engines.
