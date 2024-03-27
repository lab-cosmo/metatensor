.. _python-api-atomistic-models:

API reference
=============

.. note::

  This is the documentation for the atomistic capabilities of
  ``metatensor-torch`` version |metatensor-torch-version|. For other versions,
  look in the following pages:

  .. grid::
    :margin: 0 0 0 0

    .. grid-item-version:: 0.2.1
        :tag-prefix: metatensor-torch-v
        :url-suffix: atomistic/reference/index.html

    .. grid-item-version:: 0.2.0
        :tag-prefix: metatensor-torch-v
        :url-suffix: atomistic/reference/index.html


The atomistic machine learning models capabilities of metatensor are part of the
:ref:`TorchScript API <torch-api-reference>`, and are installed when
:ref:`installing the TorchScript API <install-torch>`.

These capabilities can be used from either Python or C++. The Python API is
documented below, and is part of the ``metatensor.torch.atomistic`` module.

.. toctree::
    :maxdepth: 1

    systems
    models/index
    ase


C++ API reference
-----------------

The C++ API can be accessed by including ``<metatensor/torch/atomistic.hpp>``.

.. toctree::
    :maxdepth: 1

    cxx/index
