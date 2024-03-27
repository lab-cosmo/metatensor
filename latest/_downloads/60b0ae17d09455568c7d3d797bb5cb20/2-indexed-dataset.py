"""
.. _learn-tutorial-indexed-dataset-dataloader:

Using IndexedDataset
====================

.. py:currentmodule:: metatensor.torch.learn.data

"""

import os

import torch

from metatensor.learn.data import DataLoader, Dataset, IndexedDataset


# %%
#
# Review of the standard Dataset
# ------------------------------
#
# The previous tutorial, :ref:`learn-tutorial-dataset-dataloader`, showed how to define
# a :py:class:`Dataset` able to handle both torch tensor and metatensor TensorMap. We
# saw that in-memory, on-disk, or mixed in-memory/on-disk datasets can be defined.
# DataLoaders are then defined on top of these Dataset objects.
#
# In all cases, however, each data sample is accessed by a numeric integer index, which
# ranges from 0 to ``len(dataset) - 1``. Let's use a simple example to review this.
#
# Again let's define some dummy data as before. Our x data as a list of random tensors,
# and our y data as a list of integers that enumerate the samples.
#
# For the purposes of this tutorial, we will only focus on an in-memory dataset, though
# the same principles apply to on-disk and mixed datasets.

n_samples = 5
x_data = [torch.randn(3) for _ in range(n_samples)]
y_data = [i for i in range(n_samples)]

dataset = Dataset(x=x_data, y=y_data)

# %%
#
# A sample is accessed by its numeric index. As the length of the lists passed as kwargs
# is 5, both for ``x`` and ``y``, the valid indices are [0, 1, 2, 3, 4].
#
# Let's retrieve the 4th sample (index 3) and print it. The value of the "y" data field
# should be 3.

print(dataset[3])

# %%
#
# What if we wanted to access samples by something other than an integer index part of a
# continuous range?
#
# For instance, what if we wanted to access samples by:
#    1. a string id, or other arbitrary hashable object?
#    2. an integer index that is not defined inside a continuous range?
#
# In these cases, we can use an IndexedDataset instead.

# %%
#
# IndexedDataset
# --------------
#
# First let's define a Dataset where the samples are indexed by arbitrary unique
# indexes, such as strings, integers, and tuples.
#
# Suppose the unique indexes for our 5 samples are:

sample_id = [
    "cat",
    4,
    ("small", "cow"),
    "dog",
    0,
]

# Build an IndexedDataset, specifying the unique sample indexes with ``sample_id``
dataset = IndexedDataset(
    x=x_data,
    y=y_data,
    sample_id=sample_id,
)

# %%
#
# Now, when we access the dataset, we can access samples by their unique sample index
# using the ``get_sample`` method. This method takes a single argument, the sample
# index, and returns the corresponding sample.

print(dataset.get_sample("dog"))
print(dataset.get_sample(4))
print(dataset.get_sample(("small", "cow")))

# %%
#
# Note that using ``__getitem__``, i.e. ``dataset[4]``, will return the sample passed to
# the constructor at position 5. In this case, the sample indexes map to the numeric
# indices as follows:
#
# 0. ``"cat"``
# 1. ``4``
# 2. ``("small", "cow")``
# 3. ``"dog"``
# 4. ``0``
#
# Thus, accessing the unique sample index ``"cat"`` can be done equivalently with
# either of:

print(dataset[0])
print(dataset.get_sample("cat"))


# %%
#
# Note that the named tuple returned in both cases contains the unique sample index as
# the ``sample_id`` field, which precedes all other data fields. This is in contrast to
# the standard Dataset, which only returns the passed data fields and not the index.
#
# A :py:class:`DataLoader` can be constructed on top of an :py:class:`IndexedDataset`
# in the same way as a :py:class:`Dataset`. Batches are accessed by iterating over the
# :py:class:`DataLoader`, though this time the ``Batch`` named tuple returned by the
# data loader will contain the unique sample indexes ``sample_id`` as the first field.

dataloader = DataLoader(dataset, batch_size=2)

# Iterate over batches
for batch in dataloader:
    print(batch)

# %%
#
# As before, we can create separate variables in the iteration pattern
for ids, x, y in dataloader:
    print(ids, x, y)

# %%
#
# On-disk :py:class:`IndexedDataset` with arbitrary sample indexes
# -----------------------------------------------------------------
#
# When defining an :py:class:`IndexedDataset` with data fields on-disk, i.e. to be
# loaded lazily, the sample indexes passed as the ``sample_id`` kwarg to the
# constructor are used as the arguments to the load function.
#
# To demonstrate this, as we did in the previous tutorial, let's save the ``x`` data to
# disk and build a mixed in-memory/on-disk :py:class:`IndexedDataset`.
#
# For instance, the below code will save sone x data for the sample ``"dog"`` at
# relative path ``"data/x_dog.pt"``.

# Create a directory to save the dummy x data to disk
os.makedirs("data", exist_ok=True)

for i, x in zip(sample_id, x_data):
    torch.save(x, f"data/x_{i}.pt")

# %%
#
# We can now define a load function to load data from disk. This should take the unique
# sample index as a single argument, and return the corresponding data in memory.


def load_x(sample_id):
    """
    Loads the x data for the sample indexed by `sample_id` from disk and
    returns the object in memory
    """
    print(f"loading x for sample {sample_id}")
    return torch.load(f"data/x_{sample_id}.pt")


# %%
#
# Now when we define an IndexedDataset, the 'x' data field can be passed as a
# callable.

mixed_dataset = IndexedDataset(x=load_x, y=y_data, sample_id=sample_id)
print(mixed_dataset.get_sample("dog"))
print(mixed_dataset.get_sample(("small", "cow")))


# %%
#
# Using an IndexedDataset: subset integer ranges
# ----------------------------------------------
#
# One could also define an IndexedDataset where the samples indices are integers forming
# a possibly shuffled and non-continuous subset of a larger continuous range of numeric
# indices.
#
# For instance, imagine we have a global Dataset of 1000 samples, with indices [0, ...,
# 999], but only want to build a dataset for samples with indices [4, 7, 200, 5, 999],
# in that order. We can pass these indices kwarg ``sample_id``.

# Build an IndexedDataset, specifying the subset sample indexes in a specific order
sample_id = [4, 7, 200, 5, 999]
dataset = IndexedDataset(x=x_data, y=y_data, sample_id=sample_id)

# %%
#
# Now, when we access the dataset, we can access samples by their unique sample index
# using the `get_sample` method. This method takes a single argument, the sample index,
# and returns the corresponding sample.
#
# Again, the numeric index can be used equivalently to access the sample, and again note
# that the ``Sample`` named tuple includes the ``sample_id`` field.

# These return the same sample
print(dataset.get_sample(5))
print(dataset[4])

# %%
#
# And finally, the DataLoader behaves as expected:

dataloader = DataLoader(dataset, batch_size=2)

for batch in dataloader:
    print(batch)
