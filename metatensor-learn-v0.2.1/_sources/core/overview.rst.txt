.. _core-classes-overview:

Overview
========

.. py:currentmodule:: metatensor

.. _concept-TensorMap:

TensorMap
^^^^^^^^^

The core type of metatensor is the :py:class:`TensorMap`: a high dimensional
block-sparse tensor containing both data and metadata. A TensorMap contains a
list of blocks (represented as :ref:`concept-TensorBlock`), each associated with
a key; and the set of all keys is stored in a :ref:`concept-Labels` object. Both
these building block for ``TensorMap`` are explained in more details below.


The keys can contain multiple dimensions (in the illustration below we have two
dimensions named ``key_1`` and ``key_2``), and each entry in the keys has one
integer value for each dimension. Here for example, the first block is
associated with ``key_1 = 0`` and ``key_2 = 0``, while the second block is
associated with ``key_1 = 0`` and ``key_2 = 1``, and so on.

.. figure:: ../../static/images/TensorMap.*
    :width: 600px
    :align: center

    Illustration of a metatensor TensorMap object, made of a set of keys and
    associated :ref:`concept-TensorBlock`.


Different key dimensions can have different purposes, but some typical keys
dimensions you'll encounter when working with atomistic data are the following:

- **atomic types dimensions**: when using a one-hot encoding of different
  atomic types (or atomic elements) in a structure, the resulting data is
  sparse, containing implicit zeros if a given type is not present in a
  structure. This is the case of the ``center_type`` and various
  ``neighbor_type`` key dimensions produced by `rascaline`_.
- **symmetry markers**: Another use case for metatensor is to store and
  manipulate equivariant data, i.e. data that transforms in a known, specific
  way when the corresponding atomic structure is transformed. This is typically
  used to represent the symmetry property of some data with respect to
  rotations, by decomposing the properties of interest on a basis of spherical
  harmonics. When handling this kind of data, it is convenient to store and
  manipulate the data corresponding to different spherical harmonics (or
  generally different irreducible representations of the symmetry group)
  separately. This is the case of the ``o3_lambda`` key dimension produced by
  `rascaline`_: different blocks will contain the :math:`\lambda = 1` and
  :math:`\lambda = 2` parts of an equivariant representation.

.. _rascaline: https://github.com/Luthaf/rascaline/

.. _concept-Labels:

Labels
^^^^^^

A fundamental part of metatensor is to carry simultaneously the data used in
machine learning and the associated metadata. The first kind of metadata we
encountered was the keys of a :py:class:`TensorMap`, stored as an instance of
the :py:class:`Labels` class. This class is also used to store all other
metadata in metatensor, i.e. all the metadata associated with a given
:py:class:`TensorBlock`.


.. _fig-labels:

.. figure:: ../../static/images/Labels.*
    :width: 600px
    :align: center

    Illustration of two different :py:class:`Labels` instances, corresponding to
    potential *samples* (green, on the left) and *properties* (red, on the
    right) of a :py:class:`TensorBlock`.


A set of :py:class:`Labels` can be seen as a two dimensional array of integers,
where each row corresponds to an entry in the data, and each column is a
*dimension*, which is named. For example, in the illustration above, the set of
Labels on the left has two dimensions (``structure`` and ``center``), and 10
entries (10 rows); while the Labels on the right has four dimensions and 8
entries. Depending on the language you use, :py:class:`Labels` entries and
dimensions' names can be accessed and manipulated in different ways, please
refer to the corresponding :ref:`API documentation <python-api-reference>` for
more information.

.. _concept-TensorBlock:

TensorBlock
^^^^^^^^^^^

The final core object of metatensor is the :py:class:`TensorBlock`, containing a
dense array of data and metadata describing the different axes of this array.
The simplest possible TensorBlock is represented below, and contains three things:

- a 2-dimensional **data** array;
- metadata describing the rows of this array, called **samples** and stored as a
  set of :py:class:`Labels`;
- metadata describing the columns of this array, called **properties**, also
  stored as a set of :py:class:`Labels`.

The samples store information about **what objects** does the data represent,
while properties store information about **how** these objects are represented.
Taking a couple of examples for clarity:

- if we are storing the energy of a list of systems in a TensorBlock, the
  samples would contain only a single ``"system"`` dimension, and multiple
  entries — one per structure — going from 0 to ``len(systems)``. The
  properties would contain a single ``"energy"`` dimension with a single entry,
  which value does not carry information;
- if we are storing increasing powers of the bond lengths between pairs of atom
  in a structure (:math:`(r_{ij})^k` for :math:`k \in [1, n]`), the samples
  would contain the index of the ``"first_atom"`` (:math:`i`) and the
  ``"second_atom"`` (:math:`j`); while the properties would contain the value of
  ``"k"``. The data array would contain the values of :math:`(r_{ij})^k`.
- if we are storing an atom-centered machine learning representation, the
  samples would contain the index of the atom ``"atom"`` and the index of the
  corresponding ``"system"``; while the properties would contain information
  about the e.g. the basis functions used to define the representation. The
  :ref:`Labels figure <fig-labels>` above contains an example of samples and
  properties that one would find in machine learning representation.

In general, for a 2-dimensional data array, the value at index ``(i, j)`` is
described by the ``i``:superscript:`th` entry of the samples and the
``j``:superscript:`th` entry of the properties.

.. figure:: ../../static/images/TensorBlock-Basic.*
    :width: 300px
    :align: center

    Illustration of the simplest possible :py:class:`TensorBlock`: a two
    dimensional data array, and two :py:class:`Labels` describing these two
    axis. The metadata associated with the first axis (rows) describes
    **samples**, while the metadata associated with the second axis (columns)
    describes **properties**.

In addition to all this metadata, metatensor also carries around some data. This
data can be stored in various arrays types, all integrated with metatensor.
Metatensor then manipulate these arrays in an opaque way, without knowing what's
inside. This allows to integrate metatensor with multiple third-party libraries
and ecosystems, for example having the data live on GPU, or using memory-mapped
data arrays.

.. admonition:: Advanced functionalities: integrating new array types with metatensor

    Currently, the following array types are integrated with metatensor:

    - `Numpy's ndarray`_ from Python,
    - `PyTorch's Tensor`_ from Python and C++, including full support for
      autograd and different device (data living on CPU memory, GPU memory, …),
    - `Rust's ndarray`_ from Rust, more specifically ``ndarray::ArrayD<f64>``,
    - A very bare-bone N-dimensional array in metatensor C++ API:
      :cpp:class:`metatensor::SimpleDataArray`

    It is possible to integrate new array types with metatensor, look into the
    :py:func:`metatensor.data.register_external_data_wrapper` function in Python, the
    :c:struct:`mts_array_t` struct in C, the :cpp:class:`metatensor::DataArrayBase`
    abstract base class in C++, and the `metatensor::Array`_ trait in Rust.

.. _Numpy's ndarray: https://numpy.org/doc/stable/reference/arrays.ndarray.html
.. _PyTorch's Tensor: https://pytorch.org/docs/stable/tensors.html
.. _Rust's ndarray: https://docs.rs/ndarray/
.. _metatensor::Array: ../reference/rust/metatensor/trait.Array.html

Gradients
---------

In addition to storing data and metadata together, a :py:class:`TensorBlock` can
also store values and gradients together. The gradients are stored in another
:py:class:`TensorBlock`, associated with a **parameter** name, describing with
respect to **what** the gradients are taken. Regarding metadata, the gradient
properties always match the values properties; while the gradient sample are
different from the value samples. The gradient samples contains both what a
given row in the data is the gradient **of**, and **with respect to** what the
gradient is taken. As illustrated below, multiple gradient rows can be gradients
of the same values row, but with respect to different things (here the positions
of different particles in the system).

.. figure:: ../../static/images/TensorBlock-Gradients.*
    :width: 550px
    :align: center

    Illustration of gradients stored inside a :py:class:`TensorBlock`.


.. TODO: explain how the gradient sample works in a separate tutorial

Components
----------

There is one more thing :py:class:`TensorBlock` can contain. When working with
vectorial data, we also handle vector **components** in both data and metadata.
In its most generic form, a :py:class:`TensorBlock` contains a
:math:`N`-dimensional data array (with :math:`N \geqslant 2`), and :math:`N` set
of :py:class:`Labels`. The first Labels describe the *samples*, the last Labels
describe the *properties*, and all the remaining Labels describe vectorial
**components** (matching all remaining axes of the data array, from the second
to one-before-last).

For example, gradients with respect to positions are actually a bit more complex
than the illustration above. They always contain a supplementary axis in the
data for the :math:`x/y/z` direction of the gradient, associated with a
**component** :py:class:`Labels`. Getting back to the example where we store
energy in the :py:class:`TensorBlock` values, the gradient (i.e. the negative of
the forces) samples describe with respect to which atom position we are taking
gradient, and the component :py:class:`Labels` allow to find the :math:`x/y/z`
component of the forces.

.. figure:: ../../static/images/TensorBlock-Components.*
    :width: 400px
    :align: center

    Illustration of a :py:class:`TensorBlock` containing **components** as an
    extra set of metadata to describe supplementary axes of the data array.

Another use-case for components is the storage of equivariant data, where a
given irreducible representation might have multiple elements. For example, when
handling spherical harmonics (which are the irreducible representation of the
`group of 3D rotations`_ :math:`SO(3)`), all the spherical harmonics
:math:`Y_\lambda^\mu` with the same angular momentum :math:`\lambda` and
corresponding :math:`\mu` should be considered simultaneously: the different
:math:`\mu` are **components** of a single irreducible representation.

.. _group of 3D rotations: https://en.wikipedia.org/wiki/3D_rotation_group

When handling the gradients of equivariant data, we quickly realize that we
might need more than one component in a given :py:class:`TensorBlock`. Gradients
with respect to positions of an equivariant representation based on spherical
harmonics will need both the gradient direction :math:`x/y/z` and the spherical
harmonics :math:`m` as components. This impacts metadata associated with
:py:class:`TensorBlock` in two ways:

- :py:class:`TensorBlock` can have an arbitrary number of components associated
  with the values, which will always occur "*in between*" samples and properties
  metadata;
- when values in a :py:class:`TensorBlock` have components, and gradient with
  respect to some parameter would add more components, the resulting gradient
  components will contain first the new, gradient-specific components, and then
  all of the components already present in the values.
