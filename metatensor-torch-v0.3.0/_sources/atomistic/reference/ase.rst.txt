Atomic Simulation Environment (ASE) integration
===============================================

.. py:currentmodule:: metatensor.torch.atomistic

The code in ``metatensor.torch.atomistic.ase_calculator`` defines a class that
allow using :py:class:`MetatensorAtomisticModel` which predict the energy of a
system as an ASE `calculator`_; enabling the use of machine learning interatomic
potentials to drive simulations inside ASE.

Additionally, it allow using arbitrary models with prediction targets which are
not just the energy, through the
:py:meth:`ase_calculator.MetatensorCalculator.run_model` function.

.. _calculator: https://wiki.fysik.dtu.dk/ase/ase/calculators/calculators.html

.. autoclass:: metatensor.torch.atomistic.ase_calculator.MetatensorCalculator
    :show-inheritance:
    :members:
