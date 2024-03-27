Metatensor
==========

Metatensor is a specialized data storage format for all your atomistic machine
learning needs, and more. Think numpy ``ndarray`` or pytorch ``Tensor`` equipped
with extra metadata for atomic — and other particles — systems. The core of this
library is written in Rust and we provide API for C, C++, and Python.

The main class of metatensor is the :py:class:`metatensor.TensorMap` data
structure, illustrated below. This class defines a block-sparse data format,
where each :py:class:`block <metatensor.TensorBlock>` is stored using
`coordinate`_ sparse storage. The schematic below represents a TensorMap made of
multiple TensorBlocks, and the overall data format is explained further in the
:ref:`getting started <userdoc-core-concepts>` section of this documentation. If
you are using metatensor from Python, we additionally provide a collection of
mathematical, logical and other utility :ref:`operations
<python-api-operations>` to make working with TensorMaps more convenient.

.. _coordinate: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)

.. image:: ../static/images/TensorMap.*
    :width: 400px
    :align: center

Why metatensor
--------------

With the creation of metatensor, we want to achieve three goals:

1. provide an interchange format for the atomistic machine learning ecosystem,
   making different players in this ecosystem more interoperable with one
   another;
2. make it easier and faster to prototype new machine learning representations,
   models and algorithms applied to atomistic modeling;
3. run large scale simulations using machine learning interatomic potentials,
   with fully customizable potentials, directly defined by the researchers
   running the simulations.

For more information on these goals and how we are trying to fulfill them,
please read the corresponding :ref:`documentation page <userdoc-goals>`.
Metatensor is still in the alpha phase of software development, so expect some
rough corners and sharp edges.

Development team
----------------

Metatensor is developed in the `COSMO laboratory`_ at `EPFL`_, and made
available to everyone under the `BSD 3-clauses license <LICENSE_>`_. We welcome
contributions from anyone, and we provide some :ref:`developer documentation
<devdoc>` for newcomers.

.. _COSMO laboratory: https://www.epfl.ch/labs/cosmo/
.. _EPFL: https://www.epfl.ch/
.. _LICENSE: https://github.com/lab-cosmo/metatensor/blob/master/LICENSE

--------------------------------------------------------------------------------

.. toctree::
   :maxdepth: 2
   :caption: Content:

   get-started/index
   reference/index
   devdoc/index
