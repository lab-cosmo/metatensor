.. _atomistic-models:

Atomistic applications
======================

While metatensor is a generic sparse data container able to store data and
metadata for multiple scientific fields, it comes from the field of atomistic
machine learning and as such offer some additional facilities for defining and
using machine learning models applied to atomistic systems.

The main goal here is to define and train models once, and then be able to
re-use them across many different simulation engines (such as LAMMPS, GROMACS,
*etc.*). We strive to achieve this goal without imposing any structure on the
model itself, and to allow any model architecture to be used.

This part of metatensor focusses on exporting and importing fully working,
already trained models. There are some tools elsewhere to define new models (in
the :ref:`operations <python-api-operations>` and :ref:`learn
<learn-api-reference>` submodules). If you want to train existing architectures
with new data or re-use existing trained models, look into the (work in
progress!) metatensor-models_ project instead.

.. _metatensor-models: https://github.com/lab-cosmo/metatensor-models

.. grid::

    .. grid-item-card:: ⚛️ Overview
        :link: atomistic-overview
        :link-type: ref
        :columns: 12 12 12 12
        :margin: 0 3 0 0

        Why should you use metatensor to define and export your model? What is
        the point of the interface? How can you use models that follow the
        interface in your own simulation code?

        All of this and more will find answers in this overview!

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

    overview
    reference/index
    outputs
    ../examples/atomistic/index
