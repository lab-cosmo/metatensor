.. image:: /../static/images/metatensor-horizontal.png
   :class: only-light sd-mb-4
   :width: 600px

.. image:: /../static/images/metatensor-horizontal-dark.png
   :class: only-dark sd-mb-4
   :width: 600px


``metatensor`` is a library for defining, manipulating, storing, and sharing
arrays with many, potentially sparse, indices. Think numpy's ``ndarray`` or
PyTorch's ``Tensor`` with additional metadata and block-sparse storage.

``metatensor`` was designed to work with data in atomistic machine learning and
makes it easy, memory efficient, and fast to keep track of spherical harmonics
orders, neighboring atoms indices, atomic types, and much more. It can also
store gradients, keeping them together with the associated values.


.. grid::

    .. grid-item-card:: 🚀 Getting started
        :link: installation
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Install the right version of metatensor for your programming language!
        The core of this library is written in Rust and we provide API for C,
        C++, and Python.

        +++
        |Python-32x32| |Cxx-32x32| |C-32x32| |Rust-32x32|

    .. grid-item-card:: 💡 Our goals
        :link: goals
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Learn about the core goals of metatensor, and what the library is about:

        - an exchange format for ML data;
        - a tool to develop new ML models;
        - an interface for atomistic simulations.

    .. grid-item-card:: 🛠️ Core classes
        :link: core-classes-overview
        :link-type: ref
        :columns: 12 12 4 4
        :margin: 0 3 0 0

        .. py:currentmodule:: metatensor

        Explore the core types of metatensor: :py:class:`TensorMap`,
        :py:class:`TensorBlock` and :py:class:`Labels`, and discover how to used
        them.

        +++
        |Python-32x32| |Cxx-32x32| |C-32x32| |Rust-32x32|


    .. grid-item-card:: 📈 Operations
        :link: metatensor-operations
        :link-type: ref
        :columns: 12 12 4 4
        :margin: 0 3 0 0

        Use `operations` to manipulate the core types of metatensor and write
        new algorithms operating on metatensor's sparse data.

        +++
        |Python-32x32|


    .. grid-item-card:: 🔥 TorchScript interface
        :link: metatensor-torch
        :link-type: ref
        :columns: 12 12 4 4
        :margin: 0 3 0 0

        Learn about the TorchScript version of metatensor, used to export and
        execute custom models inside non-Python software.

        +++
        |Python-32x32| |Cxx-32x32|


    .. grid-item-card:: 🧑‍💻 Learning utilities
        :link: metatensor-learn
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Use the utility class with the same API as torch or scikit-learn to
        train models using metatensor!

        +++
        |Python-32x32|


    .. grid-item-card:: ⚛️ Running atomistic simulations
        :link: atomistic-models
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Learn about the facilities provided to define atomistic models, and use
        them to run molecular dynamics simulations and more!

        +++
        |Python-32x32| |Cxx-32x32|



.. toctree::
   :maxdepth: 2
   :hidden:

   goals
   installation
   core/index
   operations/index
   torch/index
   learn/index
   atomistic/index
   devdoc/index
