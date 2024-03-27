Models
======

Most of the code in ``metatensor.torch.atomistic`` is here to define and export
models and to store model metadata. The corresponding classes are documented
below:

.. toctree::
    :maxdepth: 1

    export
    metadata

We also provide a couple of functions to work with the models:

.. autofunction:: metatensor.torch.atomistic.check_atomistic_model

.. autofunction:: metatensor.torch.atomistic.unit_conversion_factor

.. _known-quantities-units:

Known quantities and units
--------------------------

The following quantities and units can be used with metatensor models. Adding
new units and quantities is very easy, please contact us if you need something
else! In the mean time, you can create
:py:class:`metatensor.torch.atomistic.ModelOutput` with quantities that are not
in this table. A warning will be issued and no unit conversion will be
performed.

When working with one of the quantity in this table, the unit you use must be
one of the registered unit.

+----------------+-------------------------------------------------------------+
|   quantity     | units                                                       |
+================+=============================================================+
|   **length**   | angstrom, Bohr, nanometer, nm                               |
+----------------+-------------------------------------------------------------+
|   **energy**   | eV, meV, Hartree, kcal/mol, kJ/mol                          |
+----------------+-------------------------------------------------------------+
