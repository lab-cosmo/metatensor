.. _metatensor-operations:

Operations
==========

.. py:currentmodule:: metatensor

The Python API for metatensor also provides functions which operate on
:py:class:`TensorMap`, :py:class:`TensorBlock`, and :py:class:`Labels` and can
be used to build your own Machine Learning models. These operations are provided
in the ``metatensor-operations`` python package, which is installed by default
when doing ``pip install metatensor``.

The operations are implemented in Python, and handle the extra metadata
(including sparsity and gradients) of metatensor. Actual manipulations of the
data itself will call the corresponding functions from `numpy`_ or `PyTorch`_,
depending on how the arrays are stored in the various :py:class:`TensorBlock`.

.. _numpy: https://numpy.org/
.. _PyTorch: https://pytorch.org/

The list of all operations currently implemented is available in the API
reference below. If you need any other operation, please `open an issue
<https://github.com/lab-cosmo/metatensor/issues>`_!

.. grid::

    .. grid-item-card:: 🔥 Using the operations with PyTorch
        :link: operations-and-torch
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Learn how the operations interact with PyTorch, and in particular with
        PyTorch's automatic differentiation framework when handling gradients.

    .. grid-item-card:: |Python-16x16| Operations API reference
        :link: python-api-operations
        :link-type: ref
        :columns: 12 12 6 6
        :margin: 0 3 0 0

        Read the documentation for all the functions in the
        ``metatensor-operations`` Python package.

        +++
        Documentation for version |metatensor-operations-version|

.. toctree::
    :maxdepth: 2
    :hidden:

    torch
    reference/index

.. toctree::
    :maxdepth: 1
    :hidden:

    CHANGELOG.md
