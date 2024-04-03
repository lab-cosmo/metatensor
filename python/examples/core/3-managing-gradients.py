"""
.. _core-tutorial-gradients:

Managing gradients
==================

Another big difference between metatensor and other more generic data storage formats is
its ability to store and manipulate one or more gradients of the data together with the
data itself. In this tutorial, we will see how and where gradients are stored, and what
one can do with them.

.. note::

    Metatensor supports multiple ways of managing gradients: explicit forward gradients,
    presented in this tutorial; and implicit backward mode gradients (only when the data
    is stored in :py:class:`torch.Tensor`). Both can be mixed as well, to compute
    backward mode gradients of explicit forward mode gradients when training a model
    with gradient data (forces and virial).

    In general, the explicit forward gradients presented here are mainly relevant to you
    if you are working within the numpy ecosystem; and the implicit backward gradients
    are more interesting if you are working in the PyTorch ecosystem.

    .. TODO add tutorial explaining the difference in more details and link to it here

The code used to generate :download:`spherical-expansion.npz` is in :ref:`the first
tutorial <core-tutorial-first-steps>`, and the code for :download:`radial-spectrum.npz`
is shown :ref:`in the second <core-tutorial-sparsity>`. Notice how in both cases, the
data was computed with ``gradients=["positions"]``, meaning the gradients with respect
to atomic positions are included.

.. py:currentmodule:: metatensor

"""

# %%
#

import metatensor
from metatensor import TensorBlock, TensorMap


# %%
#
# Amazing gradients and where to find them
# ----------------------------------------
#
# In the first :ref:`tutorial <core-tutorial-first-steps>`, we have seen how metatensor
# stores data and metadata together inside :py:class:`TensorBlock`; and groups multiple
# blocks to form a full :py:class:`TensorMap`. To refresh our memory, let's load some
# data (the radial spectrum from the :ref:`sparsity tutorial <core-tutorial-sparsity>`).
# It is a tensor map with two dimensions for the keys; and 5 blocks:

# sphinx_gallery_thumbnail_path = '../static/images/TensorBlock-Gradients.*'

radial_spectrum = metatensor.load("radial-spectrum.npz")
print(radial_spectrum)

# %%
#
# If we look at one of the block, we can see that is contains gradients with respect to
# ``"positions"``:

block = radial_spectrum.block(center_type=7, neighbor_type=7)
print(block)

# %%
#
# Gradients are stored inside normal :py:class:`TensorBlock`, with their own set of
# metadata in the samples, components and properties dimensions.

gradient = block.gradient("positions")
print(gradient)

# %%
#
# The samples are different from the values blocks (the block to which this gradient it
# attached to): there is a first ``"sample"`` dimension, followed by a pair of indices
# ``(structure, atom)``.
#
# The ``"sample"`` dimension is always present in gradients, and indicate which of the
# samples in the values block we are taking the gradient of. Here, the first row of the
# gradients will contain a gradient of the first sample in the values; with respect to
# the position of atom 4; while the last row of the gradients contains a gradient of the
# second row of the values with respect to the position of atom 5.

print(gradient.samples)

# %%
#
# Re-using the notation from the previous tutorial, the values contain :math:`\rho_i`,
# for a given atomic center :math:`i`.

print(block.samples)

# %%
#
# If we look a the samples for the values, we can express the four samples in this
# gradient block as
#
# - :math:`\nabla_4 \rho_4`: gradient of the representation of atom 4 with respect to
#   the position of atom 4;
# - :math:`\nabla_5 \rho_4`: gradient of the representation of atom 4 with respect to
#   the position of atom 5;
# - :math:`\nabla_4 \rho_5`: gradient of the representation of atom 5 with respect to
#   the position of atom 4;
# - :math:`\nabla_5 \rho_5`: gradient of the representation of atom 5 with respect to
#   the position of atom 5.
#
# You'll realize that some of the combinations of atoms are missing here: there is no
# gradient of :math:`\rho_4` with respect to the positions of atom 0, 1, 2, *etc.* This
# is another instance of the data sparsity that metatensor enable: only the non-zero
# gradients are actually stored in memory.
#
# .. figure:: /../static/images/TensorBlock-Gradients.*
#     :width: 400px
#     :align: center
#
#     Visual illustration of the gradients, and how multiple gradient row/gradient
#     samples can correspond to the same row/sample in the values.
#

# %%
#
# The gradient block can also differ from the values block in the components: here the
# values have no components, but the gradient have one, representing the x/y/z cartesian
# coordinate direction of the gradient with respect to positions.

print(gradient.components)

# %%
#
# Finally, the gradient properties are guaranteed to be the same as the values
# properties.

print(block.properties == gradient.properties)

# %%
#
# The gradient block also contains the data for the gradient, in the ``values``
# attribute. Here the gradients are zeros everywhere except in the x direction because
# in the original input, the N\ :sub:`2` molecule was oriented along the x axis.

print(gradient.values)

# %%
#
# What if the values have components?
# -----------------------------------
#
# We have seen that the gradient samples are related to the values samples with the
# ``sample`` dimension; and that the gradient are allowed to have custom ``components``.
# You might be wondering what happen if the values already have some components!
#
# Let's load such an example, the spherical expansion from the :ref:`first steps
# tutorial <core-tutorial-first-steps>`:

spherical_expansion = metatensor.load("spherical-expansion.npz")
print(spherical_expansion)

# %%
#
# In the :py:class:`TensorMap` above, the value blocks already have a set of components
# corresponding to the :math:`m` index of spherical harmonics:

block = spherical_expansion.block(2)
print(block.components)

# %%
#
# If we look at the gradients with respect to positions, we see that they contain two
# sets of components: the same ``xyz`` component as the radial spectrum example
# earlier; and the same ``o3_mu`` as the values.

gradient = block.gradient("positions")
print(gradient)

# %%
#

print("first set of components:", gradient.components[0])
print("second set of components:", gradient.components[1])


# %%
#
# In general, the gradient blocks are allowed to have additional components when
# compared to the values, but these extra components must come first, and are followed
# by the same set of components as the values.


# %%
#
# Using gradients in calculations
# -------------------------------
#
# Now that we know about gradient storage in metatensor, we should try to compute a new
# set of values and their corresponding gradients.
#
# Let's compute the square of the radial spectrum, :math:`h(r) = \rho^2(r)`, and the
# corresponding gradients with respect to atomic positions. The chain rules tells us
# that the gradient should be
#
# .. math::
#
#     \nabla h(r) = 2\ \rho(r) * \nabla \rho(r)
#
# Since the calculation can happen block by block, let's define a function to compute a
# new :math:`h(r)` block:


def compute_square(block: TensorBlock) -> TensorBlock:
    # compute the new values
    new_values = block.values**2

    # store the new values in a block
    new_block = TensorBlock(
        values=new_values,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )

    # compute the new gradient
    gradient = block.gradient("positions")

    # `block.values[gradient.samples["sample"]]` gives us an array with a shape
    # compatible with `gradient.values`; using the right row in the values to compute
    # the a given row of the gradients. ``None`` creates an additional dimension to
    # match the components of the gradients.
    broadcasted_values = block.values[gradient.samples["sample"], None, :]
    new_gradient_values = 2.0 * broadcasted_values * gradient.values

    new_gradient = TensorBlock(
        values=new_gradient_values,
        samples=gradient.samples,
        components=gradient.components,
        properties=gradient.properties,
    )

    # store the gradient in the new block
    new_block.add_gradient("positions", new_gradient)

    return new_block


# %%
#
# One issue when applying the equation above blindly is that ``block.values`` (i.e.
# :math:`\rho(r)`) and ``gradient.values`` (i.e. :math:`\nabla \rho(r)`) have different
# shape. Fortunately, we already know how to match them: ``gradient.samples["sample"]``
# contains the indices of ``block.values`` matching each row of ``gradient.values``.

gradient = radial_spectrum.block(2).gradient("positions")
print(gradient.samples["sample"])

# %%
#
# We can now apply this function on all the blocks, and reconstruct a new
# :py:class:`TensorMap`:

blocks = [compute_square(block) for block in radial_spectrum.blocks()]
squared = TensorMap(keys=radial_spectrum.keys, blocks=blocks)

# %%
#
# ``squares`` has the same shape and sparsity pattern as ``radial_spectrum``, but
# contains different values:

print(squared)

# %%
#
rs_block = radial_spectrum.block(2)
squared_block = squared.block(2)

print("radial_spectrum block:", rs_block)
print("square block:", squared_block)


# %%
#

print("radial_spectrum values:", rs_block.values)
print("square values:", squared_block.values)

# %%
#

print("radial_spectrum gradient:", rs_block.gradient("positions").values[:, 0])
print("square gradient:", squared_block.gradient("positions").values[:, 0])

# %%
#
# .. tip::
#
#     We provide many functions that operate on :py:class:`TensorMap` and
#     :py:class:`TensorBlock` as part of the :ref:`metatensor-operations
#     <metatensor-operations>` module (installed by default with the main
#     ``metatensor`` package). These operations already support the different sparsity
#     levels of metatensor, and support for explicit forward gradients. In general you
#     will not have to write the type of code from this tutorial yourself, and you
#     should use the corresponding operation.
#
#     For example, ``squared`` from this tutorial can be calculated with:
#
#     .. code-block:: python
#
#         squared = metatensor.multiply(radial_spectrum, radial_spectrum)
#
#         # alternatively
#         squared = metatensor.pow(radial_spectrum, 2)
#

squared_operations = metatensor.multiply(radial_spectrum, radial_spectrum)
print(metatensor.equal(squared_operations, squared))
