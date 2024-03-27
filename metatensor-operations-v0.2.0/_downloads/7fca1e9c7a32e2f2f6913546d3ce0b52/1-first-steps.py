"""
.. _core-tutorial-first-steps:

First steps with metatensor
===========================

.. |CO2| replace:: CO\\ :sub:`2`

This tutorial explores how data is stored inside metatensor's ``TensorMap``, and how to
access the associated metadata. This is a companion to the :ref:`core classes overview
<core-classes-overview>` page of this documentation, presenting the same concepts with
code examples.

To this end, we will need some data in metatensor format, which for the sake of
simplicity will be loaded from a file. The code used to generate this file can be found
below:

.. details:: Show the code used to generate the :download:`spherical-expansion.npz`
             file, or use the link to download it

    ..

        The data was generated with `rascaline`_, a package to compute atomistic
        representations for machine learning applications.

        .. _rascaline: https://luthaf.fr/rascaline/latest/index.html

        .. literalinclude:: spherical-expansion.py.example
            :language: python

The :py:class:`TensorMap` stored in the file contains a machine learning representation
(the spherical expansion) of all the atoms in a |CO2| molecule. You don't need to know
anything the spherical expansion to follow this tutorial!

.. py:currentmodule:: metatensor

"""

# %%
#

import ase
import ase.visualize.plot
import matplotlib.pyplot as plt

import metatensor


# %%
#
# For reference, we are working with a representation of this |CO2| molecule:

co2 = ase.Atoms(
    "CO2",
    positions=[(0, 0, 0), (-0.2, -0.65, 0.94), (0.2, 0.65, -0.94)],
)


fig, ax = plt.subplots(figsize=(3, 3))
ase.visualize.plot.plot_atoms(co2, ax)
ax.set_axis_off()
plt.show()

# %%
#
# The main entry point: ``TensorMap``
# -----------------------------------
#
# We'll start by loading our data with :py:func:`metatensor.load`. The ``tensor``
# returned by this function is a :py:class:`TensorMap`, the core class of metatensor.

# sphinx_gallery_thumbnail_path = '../static/images/TensorMap.*'

tensor = metatensor.load("spherical-expansion.npz")
print(type(tensor))

# %%
#
# Looking at the tensor tells us that it is composed of 12 blocks, each associated with
# a key:

print(tensor)

# %%
#
# We can see that here, the keys of the :py:class:`TensorMap` have three named
# *dimensions*:
#
# - ``spherical_harmonics_l``, representing the angular momentum index in the spherical
#   harmonics (the :math:`l` in :math:`Y_l^m`) used by the machine learning
#   representation;
# - ``species_center`` represents the atomic species of the central atom in
#   consideration. For |CO2|, we have both carbons (species 6) and oxygens (species 8);
# - ``species_neighbor`` represents the atomic species of the neighbor atoms considered
#   by the machine learning representation, in this case it takes the values 6 and 8 as
#   well.
#
#
# These keys can be accessed with :py:attr:`TensorMap.keys`, and they are an instance of
# the :py:class:`Labels` class:

keys = tensor.keys
print(type(keys))


# %%
#
# ``Labels`` to store metadata
# ----------------------------
#
# One of the :ref:`main goals of metatensor <metatensor-goal-exchange>` is to be able to
# store both data and metadata together. We've just encountered the first example of
# this metadata as the :py:class:`TensorMap` keys! In general, most metadata will be
# stored in the :py:class:`Labels` class. Let's explore this class a bit.
#
# As already mentioned, :py:class:`Labels` can have multiple dimensions, and each
# dimension has a name. We can look at all the dimension names simultaneously with
# :py:func:`Labels.names`:
print(keys.names)

# %%
#
# :py:class:`Labels` then contains multiple entries, each entry being described by a set
# of integer values, one for each dimension of the labels.

print(keys.values)

# %%
#
# We can access all the values taken by a given dimension/column in the labels with
# :py:func:`Labels.column` or by indexing with a string:

print(keys["spherical_harmonics_l"])

# %%
#

print(keys.column("species_center"))

# %%
#
# We can also access individual entries in the labels by iterating over them or indexing
# with an integer:

print("Entries with spherical_harmonics_l=2:")
for entry in keys:
    if entry["spherical_harmonics_l"] == 2:
        print("    ", entry)

print("\nEntry at index 3:")
print("    ", keys[3])


# %%
#
# ``TensorBlock`` to store the data
# ---------------------------------
#
# Each entry in the :py:attr:`TensorMap.keys` is associated with a
# :py:class:`TensorBlock`, which contains the actual data and some additional metadata.
# We can extract the block from a key by indexing our :py:class:`TensorMap`, or with the
# :py:func:`TensorMap.block`

# this is equivalent to `block = tensor[tensor.keys[0]]`
block = tensor[0]

block = tensor.block(spherical_harmonics_l=1, species_center=8, species_neighbor=6)

print(block)

# %%
#
# Each block contains some data, stored inside the :py:attr:`TensorBlock.values`. Here,
# the values contains the different coefficients of the spherical expansion, i.e. our
# atomistic machine learning representation.
#
# The problem with this array is that we do not know what the different numbers
# correspond to: different libraries might be using different convention and storage
# order, and one has to read documentation carefully if they want to use this kind of
# data. Metatensor helps by making this data self-describing; by attaching metadata to
# each element of the array indicating what exactly we are working with.

print(block.values)

# %%
#
# The metadata is attached to the different array axes, and stored in
# :py:class:`Labels`. The array must have at least two axes but can have more if
# required. Here, we have three:

print(block.values.shape)

# %%
#
# The **first** dimension of the ``values`` array is described by the
# :py:attr:`TensorBlock.samples` labels, and correspond to **what** is being described.
# This follows the usual convention in machine learning, using the different rows of the
# array to store separate samples/observations.
#
# Here, since we are working with a per-atom representation, the samples contain the
# index of the structure and atomic center in this structure. Since we are looking at a
# block for ``species_center=8``, we have two samples, one for each oxygen atom in our
# single |CO2| molecule.

print(block.samples)

# %%
#
# The **last** dimension of the ``values`` array is described by the
# :py:attr:`TensorBlock.properties` labels, and correspond to **how** we are
# describing our subject. Here, we are using a radial basis, indexed by an integer
# ``n``:

print(repr(block.properties))

# %%
#
# Finally, each **intermediate** dimension of the ``values`` array is described by one
# set of :py:attr:`TensorBlock.components` labels. These dimensions correspond to one or
# more *vectorial components* in the data. Here the only component corresponds to the
# different :math:`m` number in spherical harmonics :math:`Y_l^m`, going from -1 to 1
# since we are looking at the block for ``spherical_harmonics_l = 1``:

print(block.components)

# %%
#
# All this metadata allow us to know exactly what each entry in the ``values``
# corresponds to. For example, we can see that the value at position ``(1, 0, 3)``
# corresponds to:
#
# - the center at index 2 inside the structure at index 0;
# - the ``m=-1`` part of the spherical harmonics;
# - the coefficients on the ``n=3`` radial basis function.

print("value =", block.values[1, 0, 3])
print("sample =", block.samples[1])
print("component =", block.components[0][0])
print("property =", block.properties[3])


# %%
#
# Wrapping it up
# --------------
#
# .. figure:: /../static/images/TensorMap.*
#     :width: 400px
#     :align: center
#
#     Illustration of the structure of a :py:class:`TensorMap`, with multiple keys and
#     blocks.

# %%
#
# To summarize this tutorial, we saw that a :py:class:`TensorMap` contains multiple
# :py:class:`TensorBlock`, each associated with a key. The key describes the block, and
# what kind of data will be found inside.
#
# The blocks contains the actual data, and multiple set of metadata, one for each axis
# of the data array.
#
# - The rows are described by ``samples`` labels, which describe **what** is being
#   stored;
# - the (generalized) columns are described by ``properties``, which describe **how**
#   the data is being represented;
# - Additional axes of the array correspond to vectorial ``components`` in the data.
#
# All the metadata is stored inside :py:class:`Labels`, where each entry is described by
# the integer values is takes along some named dimensions.
#
# For a more visual approach to this data organization, you can also read the :ref:`core
# classes overview <core-classes-overview>`.
#
# We have learned how metatensor organizes its data, and what makes it a "self
# describing data format". In the :ref:`next tutorial <core-tutorial-sparsity>`, we will
# explore what makes metatensor :py:class:`TensorMap` a "sparse data format".
