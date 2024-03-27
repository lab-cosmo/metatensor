Information about models
------------------------

.. py:currentmodule:: metatensor.torch.atomistic

Here are the classes that are used to store and use information about the
atomistic models.

- :py:class:`ModelMetadata` stores metadata about the model: name, authors,
  references, *etc.*
- :py:class:`ModelCapabilities` stores information about what a model can do.
  Part of that is the full set of outputs the model can produce, stored in
  :py:class:`ModelOutput`;
- :py:class:`ModelEvaluationOptions` is used by the simulation engine to request
  the model to do some things. This is handled by
  :py:class:`MetatensorAtomisticModel`, and transformed into the arguments given
  to :py:meth:`ModelInterface.forward`.

--------------------------------------------------------------------------------

.. autoclass:: metatensor.torch.atomistic.ModelMetadata
    :members:

.. autoclass:: metatensor.torch.atomistic.ModelOutput
    :members:

.. autoclass:: metatensor.torch.atomistic.ModelCapabilities
    :members:

.. autoclass:: metatensor.torch.atomistic.ModelEvaluationOptions
    :members:
