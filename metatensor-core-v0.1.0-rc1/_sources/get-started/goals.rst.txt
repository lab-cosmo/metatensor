.. _userdoc-goals:

Goals of this library
=====================

With the creation of metatensor, we have three main use cases in mind:

1. provide an exchange format for the atomistic machine learning ecosystem,
   making different players in this ecosystem more interoperable with one
   another and enhancing collaboration;
2. make it easier and faster to prototype new machine learning representations,
   models and algorithms applied to atomistic modeling;
3. run large scale simulations using machine learning interatomic potentials,
   with fully customizable potentials, directly defined by the researchers
   running the simulations.

If you agree with any of these goals, you might find metatensor useful! We try to
make metatensor usable for any single one of these goals, it is perfectly fine to
only use it for a subset of its capacities.

Metatensor as an exchange format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, metatensor is a format to exchange data between different libraries in the
atomistic machine learning ecosystem. There is currently an explosion of
libraries and tools for atomistic machine learning, implementing new
representation, new models, and advanced research methods. Unfortunately each
one of these libraries lives mostly separated from the others, resulting in a
lot of duplicated effort. With metatensor, we want to provide a way for these
libraries to communicate with one another, by giving everyone a *lingua franca*,
a way to share data and metadata.

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
the metatensor-compatible libraries!

.. TODO: add illustration

.. admonition:: libraries using metatensor

    The following libraries use metatensor either as input, output or both:

    - `equisolve <https://github.com/lab-cosmo/equisolve/>`_: a companion to
      metatensor implementing common ML models that take all the data as metatensor
      TensorMaps;
    - `rascaline <https://github.com/Luthaf/rascaline/>`_: a library computing
      physics-inspired representations of atomic systems. Rascaline always outputs
      its representations in metatensor format;
    - `Q-stack <https://github.com/lcmd-epfl/Q-stack/>`_: library of pre- and
      post-processing tasks for Quantum Machine Learning; can output some of its
      data in metatensor format;


Metatensor as a prototyping tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second objective of metatensor is to provide functionalities to be a
prototyping tool for new models. While it is possible to use metatensor to only
exchange data between libraries (and immediately convert everything to library
specific format); we also provide tools to operate on metatensor data, staying in
the metatensor format.

We call these tools :ref:`operations <python-api-operations>`, and they
available in the Python interface to metatensor. By using combining multiple
operations, you can build custom machine learning models, using data and
representations coming from arbitrary metatensor-compatible libraires in the
ecosystem. Using these operations allow you to keep your data in metatensor
format across the whole ML pipeline; meaning the metadata is kept up to date
with the data, and arbitrary gradients are automatically updated to stay
consistent with the values.

.. TODO: add illustration


Metatensor for running simulation with custom models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One particularly interesting class of machine learning model for atomistic
modelling is machine learning interatomic potentials (MLIPs). Using the
capacities provided by the first two goals of metatensor, researchers should be
able to created and train such MLIPs and customize various parts of the model.

The final objective of metatensor is to allow using these custom models inside
large scale molecular simulation engines. To do this, we plan to integrate
metatensor with `TorchScript <https://pytorch.org/docs/stable/jit.html>`_, and
use the facilities of TorchScript to export the model from Python and then load
and execute it inside the simulation engine. This is a planned feature, not
implemented yet.

.. TODO: add illustration
