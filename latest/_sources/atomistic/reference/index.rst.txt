.. _python-api-atomistic-models:

API reference
=============

.. note::

  This is the documentation for the atomistic capabilities of
  ``metatensor-torch`` version |metatensor-torch-version|. For other versions,
  look in the following pages:

  .. grid::
    :margin: 0 0 0 0

    .. grid-item-version:: 0.3.0
        :tag-prefix: metatensor-torch-v
        :url-suffix: atomistic/reference/index.html

    .. grid-item-version:: 0.2.1
        :tag-prefix: metatensor-torch-v
        :url-suffix: atomistic/reference/index.html

    .. grid-item-version:: 0.2.0
        :tag-prefix: metatensor-torch-v
        :url-suffix: atomistic/reference/index.html


The atomistic machine learning models capabilities of metatensor are part of the
:ref:`TorchScript API <torch-api-reference>`, and are installed when
:ref:`installing the TorchScript API <install-torch>`. We provide code for using
atomistic systems as input of a machine learning model with
:py:class:`metatensor.torch.atomistic.System`, and exporting trained models with
:py:class:`metatensor.torch.atomistic.MetatensorAtomisticModel`.

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
