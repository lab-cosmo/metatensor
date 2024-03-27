.. _atomistic-models-outputs:

Standard model outputs
======================

In order for multiple simulation engines to be able to exploit atomic properties
computing by arbitrary metatensor atomistic models, we need all the models to
return data with specific metadata. If your model returns one of the output
defined in this documentation, then the model should follow the metadata
structure described here.

For other kind of output, you are free to use any relevant metadata structure,
but if multiple people are producing the same kind of outputs, they are
encouraged to come together, define the metadata they need and add a new section
to this page.


Energy
------

Energy (and its gradients: forces/virial) is associated with the ``"energy"``
key in the model outputs, and must have the following metadata:

- **keys**: the energy keys must have a single dimension named ``"_"``, with a
  single entry set to ``0``. The energy is always a single block
  :py:class:`metatensor.torch.TensorMap`.
- **samples**: if doing ``per_atom`` output, the sample names must be
  ``["system", "atom"]``, otherwise the sample names must be ``["system"]``.
  ``"structure"`` must range from 0 to the number of systems given as input to
  the model. ``"atom"`` must range between 0 and the number of atoms in the
  corresponding system. If ``selected_atoms`` is provided, then only the
  selected atoms for each system should be part of the samples.
- **components**: the energy must not have any components.
- **properties**: the energy must have a single property dimension named
  ``"energy"``, with a single entry set to ``0``.
