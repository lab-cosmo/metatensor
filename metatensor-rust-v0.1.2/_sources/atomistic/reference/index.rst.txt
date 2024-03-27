.. _python-api-atomistic-models:

API reference
=============

.. note::

  This is the documentation for the atomistic capabilities of
  ``metatensor-torch`` version |metatensor-torch-version|. For other versions,
  look in the following pages:

  .. grid::
    :margin: 0 0 0 0

    .. grid-item-card:: Version 0.2.0
      :link: https://lab-cosmo.github.io/metatensor/metatensor-torch-v0.2.0/reference/torch/index.html
      :link-type: url
      :columns: 12 6 3 3
      :text-align: center
      :class-body: sd-p-2
      :class-title: sd-mb-0


The atomistic machine learning models capabilities of metatensor are part of the
:ref:`TorchScript API <torch-api-reference>`, and are installed when
:ref:`installing the TorchScript API <install-torch>`.

These capabilities can be used from either Python or C++. The Python API is
documented below, and is part of the ``metatensor.torch.atomistic`` module.

.. toctree::
    :maxdepth: 1

    systems
    models
    ase


C++ API reference
-----------------

The C++ API can be accessed by including ``<metatensor/torch/atomistic.hpp>``.

.. toctree::
    :maxdepth: 1

    cxx/index
