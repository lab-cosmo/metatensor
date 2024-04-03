Exporting models
================

.. py:currentmodule:: metatensor.torch.atomistic

Exporting models to work with any metatensor-compatible simulation engine is
done with the :py:class:`MetatensorAtomisticModel` class. This class takes in an
arbitrary :py:class:`torch.nn.Module`, with a forward functions that follows the
:py:class:`ModelInterface`. In addition to the actual model, you also need to
define some information about the model, using :py:class:`ModelMetadata` and
:py:class:`ModelCapabilities`.

.. autoclass:: metatensor.torch.atomistic.ModelInterface
    :members:
    :show-inheritance:

.. autoclass:: metatensor.torch.atomistic.MetatensorAtomisticModel
    :members:
