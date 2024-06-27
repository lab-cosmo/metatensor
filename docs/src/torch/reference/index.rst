.. _python-api-torch:

API reference
=============

.. note::

  This is the documentation for ``metatensor-torch`` version
  |metatensor-torch-version|. For other versions, look in the following pages:

  .. grid::
    :margin: 0 0 0 0

    .. grid-item-version:: 0.5.2
        :tag-prefix: metatensor-torch-v
        :url-suffix: torch/reference/index.html

    .. grid-item-version:: 0.5.1
        :tag-prefix: metatensor-torch-v
        :url-suffix: torch/reference/index.html

    .. grid-item-version:: 0.5.0
        :tag-prefix: metatensor-torch-v
        :url-suffix: torch/reference/index.html

    .. grid-item-version:: 0.4.0
        :tag-prefix: metatensor-torch-v
        :url-suffix: torch/reference/index.html

    .. grid-item-version:: 0.3.0
        :tag-prefix: metatensor-torch-v
        :url-suffix: torch/reference/index.html

    .. grid-item-version:: 0.2.1
        :tag-prefix: metatensor-torch-v
        :url-suffix: torch/reference/index.html

    .. grid-item-version:: 0.2.0
        :tag-prefix: metatensor-torch-v
        :url-suffix: reference/torch/index.html

    .. grid-item-version:: 0.1.0
        :tag-prefix: metatensor-torch-v
        :url-suffix: reference/torch/index.html


.. py:currentmodule:: metatensor.torch

The classes and functions in the TorchScript API are kept as close as possible
to the classes and functions of the pure Python API, with the explicit goal that
changing from

.. code-block:: python

    import metatensor
    from metatensor import TensorMap, TensorBlock, Labels

to

.. code-block:: python

    import metatensor.torch as metatensor
    from metatensor.torch import TensorMap, TensorBlock, Labels

should be 80% of the work required to make a model developed in Python with
:py:mod:`metatensor.operations` compatible with TorchScript. In particular, all
the :ref:`operations <python-api-operations>` are also available in the
``metatensor.torch`` module under the same name. All the functions have the same
behavior, but the versions in ``metatensor.torch`` are annotated with the types
from ``metatensor.torch``, and compatible with TorchScript compilation. For
example :py:func:`metatensor.add()` is available as ``metatensor.torch.add()``.

The :ref:`learn <python-api-learn>` module is also re-exported inside
``metatensor.torch.learn``, with the same functionalities as
``metatensor.learn``.

The documentation for the usual core classes of metatensor can be found in the
following pages:

.. toctree::
    :maxdepth: 1

    tensor
    block
    labels
    serialization


.. this is linked directly from torch/index.rst
.. toctree::
    :maxdepth: 1
    :hidden:

    cxx/index
