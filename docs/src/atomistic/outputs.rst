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
^^^^^^

Energy is associated with the ``"energy"`` key in the model outputs, and must
have the following metadata:

.. list-table:: Metadata for energy output
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - keys
    - ``"_"``
    - the energy keys must have a single dimension named ``"_"``, with a single
      entry set to ``0``. The energy is always a
      :py:class:`metatensor.torch.TensorMap` with a single block.

  * - samples
    - ``["system", "atom"]`` or ``["system"]``
    - if doing ``per_atom`` output, the sample names must be ``["system",
      "atom"]``, otherwise the sample names must be ``["system"]``.

      ``"system"`` must range from 0 to the number of systems given as input to
      the model. ``"atom"`` must range between 0 and the number of
      atoms/particles in the corresponding system. If ``selected_atoms`` is
      provided, then only the selected atoms for each system should be part of
      the samples.

  * - components
    -
    - the energy must not have any components

  * - properties
    - ``"energy"``
    - the energy must have a single property dimension named ``"energy"``, with
      a single entry set to ``0``.

Energy gradients
----------------

Most of the time when writing an atomistic model compatible with metatensor,
gradients will be handled implicitly and computed by the simulation engine using
a backward pass. Additionally, it is possible for the model to support explicit,
forward mode gradients

The following gradients can be defined and requested with
``explicit_gradients``:

- **"positions"** (:math:`r_j`) gradients will contain the negative of the
  forces :math:`F_j`.

  .. math::

      \frac{\partial E}{\partial r_j} = -F_j

.. list-table:: Metadata for positions energy's gradients
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - samples
    - ``["sample", "system", "atom"]``
    - ``"sample"`` indicates which of the values samples we are taking the
      gradient of, and ``("system", "atom")`` indicates which of the atom's
      positions we are taking the gradients with respect to; i.e. :math:`j` in
      the equation above.

  * - components
    - ``"xyz"``
    - there must be a single component named ``"xyz"`` with values 0, 1, 2;
      indicating the direction of the displacement of the atom in the gradient
      samples.

- **"strain"** (:math:`\epsilon`) gradients will contain the stress
  :math:`\sigma` acting on the system, multiplied by the volume :math:`V`
  (sometimes also called the *virial* of this system)

  .. math::

    \frac{\partial E}{\partial \epsilon} = V \sigma

.. list-table:: Metadata for strain energy's gradients
  :widths: 2 3 7
  :header-rows: 1

  * - Metadata
    - Names
    - Description

  * - **samples**
    - ``"sample"``
    - There is a single gradient sample dimension, ``"sample"`` indicating which
      of the values samples we are taking the gradient of.

  * - **components**
    - ``["xyz_1", "xyz_2"]``
    - Both ``"xyz_1"`` and ``"xyz_2"`` have values ``[0, 1, 2]``, and correspond
      to the two axes of the 3x3 strain matrix :math:`\epsilon`.
