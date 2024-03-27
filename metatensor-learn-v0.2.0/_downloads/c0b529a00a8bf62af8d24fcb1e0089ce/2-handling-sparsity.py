r"""
.. _core-tutorial-sparsity:

Handling sparsity
=================

The one sentence introduction to metatensor mentions that this is a "self-describing
**sparse** tensor data format". The :ref:`previous tutorial <core-tutorial-first-steps>`
explained the self-describing part of the format, and in this tutorial we will explore
what makes metatensor a sparse format; and how to remove the corresponding sparsity when
required.

Like in the :ref:`previous tutorial <core-tutorial-first-steps>`, we will load the data
we need from a file. The code used to generate this file can be found below:

.. details:: Show the code used to generate the :file:`radial-spectrum.npz` file

    ..

        The data was generated with `rascaline`_, a package to compute atomistic
        representations for machine learning applications.

        .. _rascaline: https://luthaf.fr/rascaline/latest/index.html

        .. literalinclude:: radial-spectrum.py.example
            :language: python

The file contains a representation of two molecules called the radial spectrum. The atom
:math:`i` is represented by the radial spectrum :math:`R_i^\alpha`, which is an
expansion of the neighbor density :math:`\rho_i^\alpha(r)` on a set of radial basis
functions :math:`f_n(r)`

.. math::

    R_i^\alpha(n) = \int f_n(r) \rho_i(r) dr

The density :math:`\rho_i^\alpha(r)` associated with all neighbors of species
:math:`\alpha` of the atom :math:`i` (each neighbor is replaced with a Gaussian function
centered on the neighbor :math:`g(r_{ij})`) is defined as:

.. math::

    \rho_i^\alpha(r) = \sum_{j \in \text{ neighborhood of i }} g(r_{ij})
        \delta_{\alpha_j,\alpha}


The exact mathematical details above don't matter too much for this tutorial, the main
point being that this representation treats atomic species as completely independent,
effectively using the neighbor species :math:`\alpha` for `one-hot encoding`_.

.. _one-hot encoding: https://en.wikipedia.org/wiki/One-hot

.. py:currentmodule:: metatensor

"""

# %%
#

import ase
import ase.visualize.plot
import matplotlib.pyplot as plt
import numpy as np

import metatensor


# %%
#
# We will work on the radial spectrum representation of three molecules in our system:
# a carbon monoxide, an oxygen molecule and a nitrogen molecule.

atoms = ase.Atoms(
    "COO2N2",
    positions=[(0, 0, 0), (1.2, 0, 0), (0, 6, 0), (1.1, 6, 0), (6, 0, 0), (7.3, 0, 0)],
)

fig, ax = plt.subplots(figsize=(3, 3))
ase.visualize.plot.plot_atoms(atoms, ax)
ax.set_axis_off()
plt.show()

# %%
#
# Sparsity in ``TensorMap``
# -------------------------
#
# The radial spectrum representation has two keys: ``central_species`` indicating the
# species of the central atom (atom :math:`i` in the equations); and
# ``neighbor_species`` indicating the species of the neighboring atoms (atom :math:`j`
# in the equations)

radial_spectrum = metatensor.load("radial-spectrum.npz")

print(radial_spectrum)

# %%
#
# This shows the first level of sparsity in ``TensorMap``: block sparsity.
#
# Out of all possible combinations of ``central_species`` and ``neighbor_species``, some
# are missing such as ``central_species=7, neighbor_species=8``. This is because we are
# using a spherical cutoff of 2.5 Å, and as such there are no oxygen neighbor atoms
# close enough to the nitrogen centers. This means that all the corresponding radial
# spectrum coefficients :math:`R_i^\alpha(n)` will be zero (since the neighbor density
# :math:`\rho_i^\alpha(r)` is zero everywhere).
#
# Instead of wasting memory space by storing all of these zeros explicitly, we simply
# avoid creating the corresponding blocks from the get-go and save a lot of memory!


# %%
#
# Let's now look at the block containing the representation for oxygen centers and
# carbon neighbors:

block = radial_spectrum.block(species_center=8, species_neighbor=6)

# %%
#
# Naively, this block should contain samples for all oxygen atoms (since
# ``species_center=8``); in practice we only have a single sample!

print(block.samples)

# %%
#
# There is a second level of sparsity happening here, using a format related to
# `coordinate sparse arrays (COO format) <COO_>`_. Since there is only one oxygen atom
# with carbon neighbors, we only include this atom in the samples, and the
# density/radial spectrum coefficient for all the other oxygen atoms is assumed to be
# zero.
#
# .. _COO: https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)


# %%
#
# Making the data dense again
# ---------------------------
#
# Sometimes, we might have to use data in a sparse metatensor format with code that does
# not understands this sparsity. One solution is to convert the data to a dense format,
# making the zeros explicit as much as possible. Metatensor provides functionalities to
# convert sparse data to a dense format for the keys sparsity; and metadata to convert
# to a dense format for sample sparsity.
#
# First, the sample sparsity can be removed block by block by creating a new array full
# of zeros, and copying the data according to the indices in ``block.samples``

dense_block_data = np.zeros((len(atoms), block.values.shape[1]))

# only copy the non-zero data stored in the block
dense_block_data[block.samples["center"]] = block.values

print(dense_block_data)


# %%
#
# Alternatively, we can undo the keys sparsity with
# :py:meth:`TensorMap.keys_to_samples` and :py:meth:`TensorMap.keys_to_properties`,
# which merge multiple blocks along the samples or properties dimensions respectively.
#
# Which one of these functions to call will depend on the data you are handling.
# Typically, one-hot encoding (the ``species_neighbors`` key here) should be merged
# along the properties dimension; and keys that define subsets of the samples
# (``species_center``) should be merged along the samples dimension.

dense_radial_spectrum = radial_spectrum.keys_to_samples("species_center")
dense_radial_spectrum = dense_radial_spectrum.keys_to_properties("species_neighbor")

# %%
#
# After calling these two functions, we now have a :py:class:`TensorMap` with a single
# block and no keys:

print(dense_radial_spectrum)

block = dense_radial_spectrum.block()

# %%
#
# We can see that the resulting dense data array contains a lot of zeros (and has a well
# defined block-sparse structure):

with np.printoptions(precision=3):
    print(block.values)

# %%
#
# And using the metadata attached to the block, we can understand which part of the data
# is zero and why. For example, the lower-right corner of the array corresponds to
# nitrogen atoms (the last two samples):

print(block.samples.print(max_entries=-1))

# %%
#
# And these two bottom rows are zero everywhere, except in the part representing the
# nitrogen neighbor density:

print(block.properties.print(max_entries=-1))
