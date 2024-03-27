"""
.. _learn-tutorial-dataset-dataloader:

Datasets and data loaders
=========================

.. py:currentmodule:: metatensor.learn.data

This tutorial shows how to define :py:class:`Dataset` and :py:class:`DataLoader`
compatible with PyTorch and containing metatensor data (i.e. data stored in
:py:class:`metatensor.torch.TensorMap`) in addition to more usual types of data.
"""

import os

import torch

from metatensor.learn.data import DataLoader, Dataset
from metatensor.torch import Labels, TensorBlock, TensorMap


# %%
#
# Let's define a simple dummy dataset with two fields, named 'x' and 'y'. Every field in
# the `Dataset` must be a list of objects corresponding to the different samples in this
# dataset.
#
# Let's define our x data as a list of random tensors, and our y data as a list of
# integers enumerating the samples.

n_samples = 5
x_data = [torch.randn(3) for _ in range(n_samples)]
y_data = [i for i in range(n_samples)]

# %%
#
# In-memory dataset
# -----------------
#
# We are ready to build out first dataset. The simplest use case is when all data is in
# memory. In this case, we can pass the data directly to the :py:class:`Dataset`
# constructor as keyword arguments, named and ordered according to how we want the data
# to be returned when we access samples in the dataset.

in_memory_dataset = Dataset(x=x_data, y=y_data)

# %%
#
# We can now access samples in the dataset. The returned object is a named tuple with
# fields corresponding to the keyword arguments given to the :py:class:``Dataset`
# constructor (here ``x`` and ``y``).

print(in_memory_dataset[0])

# %%
#
# One can also iterate over the samples in the dataset as follows:

for sample in in_memory_dataset:
    print(sample)

# %%
#
# Any number of named data fields can be passed to the Dataset constructor, as long as
# they are all uniquely named, and are all lists of the same length. The elements of
# each list can be any type of object (integer, string, torch Tensor, etc.), as long as
# it is the type same for all samples in the respective field.
#
# For example, here we are creating a dataset of torch tensors (``x``), integers
# (``y``), and strings (``z``).

bigger_dataset = Dataset(x=x_data, y=y_data, z=["a", "b", "c", "d", "e"])
print(bigger_dataset[0])
print("Sample 4, z field:", bigger_dataset[4].z)

# %%
#
# Mixed in-memory / on-disk dataset
# ---------------------------------
#
# Now suppose we have a large dataset, where the x data is too large to fit in
# memory. In this case, we might want to lazily load data when training a model
# with minibatches.
#
# Let's save the x data to disk to simulate this use case.

# Create a directory to save the dummy x data to disk
os.makedirs("data", exist_ok=True)

for i, x in enumerate(x_data):
    torch.save(x, f"data/x_{i}.pt")

# %%
#
# In order for the x data to be loaded lazily, we need to give the ``Dataset`` a
# ``load`` function that loads a single sample into memory. This can a function of
# arbitrary complexity, taking a single argument which is the numeric index (between
# ``0`` and ``len(dataset)``) of the sample to load


def load_x(sample_id):
    """
    Loads the x data for the sample indexed by `sample_id` from disk and returns the
    object in memory
    """
    print(f"loading x for sample {sample_id}")
    return torch.load(f"data/x_{sample_id}.pt")


print("load_x called with sample index 0:", load_x(0))

# %%
#
# Now when we define a dataset, the 'x' data field can be passed as a callable.

mixed_dataset = Dataset(x=load_x, y=y_data)
print(mixed_dataset[3])

# %%
#
# On-disk dataset
# ---------------
#
# Finally, suppose we have a large dataset, where both the x and y data are too large to
# fit in memory. In this case, we might want to lazily load all data when training a
# model with minibatches.
#
# Let's save the y data to disk as well to simulate this use case.

for i, y in enumerate(y_data):
    torch.save(y, f"data/y_{i}.pt")


def load_y(sample_id):
    """
    Loads the y data for the sample indexed by `sample_id` from disk and
    returns the object in memory
    """
    print(f"loading y for sample {sample_id}")
    return torch.load(f"data/y_{sample_id}.pt")


print("load_y called with sample index 0:", load_y(0))

# %%
#
# Now when we define a dataset, as all the fields are to be lazily loaded, we need to
# indicate how many samples are in the dataset with the ``size`` argument.
#
# Internally, the Dataset class infers the unique sample indexes as a continuous integer
# sequence starting from 0 to ``size - 1`` (inclusive). In this case, sample indexes are
# therefore [0, 1, 2, 3, 4]. These indexes are used to lazily load the data upon access.
on_disk_dataset = Dataset(x=load_x, y=load_y, size=n_samples)
print(on_disk_dataset[2])

# %%
#
# Building a Dataloader
# ---------------------
#
# Now let's see how we can use the Dataset class to build a DataLoader.
#
# Metatensor's ``DataLoader`` class is a wrapper around the PyTorch ``DataLoader``
# class, and as such can be initialized with a ``Dataset`` object. It will also inherit
# all of the default arguments from the PyTorch DataLoader class.

in_memory_dataloader = DataLoader(in_memory_dataset)

# %%
#
# We can now iterate over the DataLoader to access batches of samples from the
# dataset. With no arguments passed, the default batch size is 1 and the samples
# are not shuffled.

for batch in in_memory_dataloader:
    print(batch.y)

# %%
#
# As an alternative syntax, the data fields can be unpacked into separate variables in
# the for loop.

for x, y in in_memory_dataloader:
    print(x, y)

# %%
#
# We can also pass arguments to the DataLoader constructor to change the batch
# size and shuffling of the samples.
in_memory_dataloader = DataLoader(in_memory_dataset, batch_size=2, shuffle=True)

for batch in in_memory_dataloader:
    print(batch.y)

# %%
#
# Data loaders for cross-validation
# ---------------------------------
#
# One can use the usual torch :py:func:`torch.utils.data.random_split` function to split
# a ``Dataset`` into train, validation, and test subsets for cross-validation purposes.
# ``DataLoader`` s can then be constructed for each subset.


# Perform a random train/val/test split of the Dataset,
# in the relative proportions (60% / 20% / 20%)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    in_memory_dataset, [0.6, 0.2, 0.2]
)

# Construct DataLoaders for each subset
train_dataloader = DataLoader(train_dataset)
val_dataloader = DataLoader(val_dataset)
test_dataloader = DataLoader(test_dataset)

# As the Dataset was initialized with 5 samples, the split should be 3:1:1
print(f"Dataset size: {len(on_disk_dataset)}")
print(f"Training set size: {len(train_dataloader)}")
print(f"Validation set size: {len(val_dataloader)}")
print(f"Test set size: {len(test_dataloader)}")


# %%
#
# Working with :py:class:`torch.Tensor` and :py:class:`metatensor.torch.TensorMap`
# --------------------------------------------------------------------------------
#
# As the :py:class:`Dataset` and :py:class:`DataLoader` classes exist to interface
# metatensor and torch, let's explore how they behave when using
# :py:class:`torch.Tensor` and :py:class:`metatensor.torch.TensorMap` objects as the
# data.
#
# We'll consider some dummy data consisting of the following fields:
#
# - **descriptor**: a list of random TensorMap objects
# - **scalar**: a list of random floats
# - **vector**: a list of random torch Tensors

# Create a dummy descriptor as a TensorMap
descriptor = [
    TensorMap(
        keys=Labels(
            names=["key_1", "key_2"],
            values=torch.tensor([[1, 2]]),
        ),
        blocks=[
            TensorBlock(
                values=torch.randn((1, 3)),
                samples=Labels("sample_id", torch.tensor([[sample_id]])),
                components=[],
                properties=Labels("p", torch.tensor([[1], [4], [5]])),
            )
        ],
    )
    for sample_id in range(n_samples)
]

# Create dummy scalar and vectorial target properties as torch Tensors
scalar = [float(torch.rand(1, 1)) for _ in range(n_samples)]
vector = [torch.rand(1, 3) for _ in range(n_samples)]

# Build the Dataset
dataset = Dataset(
    scalar=scalar,
    vector=vector,
    descriptor=descriptor,
)
print(dataset[0])

# %%
#
# Merging samples in a batch
# --------------------------
#
# As is typically customary when working with torch Tensors, we want to vertically stack
# the samples in a minibatch into a single Tensor object. This allows passing a single
# Tensor object to a model, rather than a tuple of Tensor objects. In a similar way,
# sparse data stored in metatensor TensorMap objects can also be vertically stacked,
# i.e. joined along the samples axis, into a single TensorMap object.
#
# The default ``collate_fn`` used by :py:class:`DataLoader`
# (:py:func:`metatensor.learn.data.group_and_join`), vstacks (respectively joins along
# the samples axis) data fields that correspond :py:class:`torch.Tensor` (respectively
# :py:class:`metatensor.torch.TensorMap`). For all other data types, the data is left as
# tuple containing all samples in the current batch in order.

batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size)

# %%
#
# We can look at a single ``Batch`` object (i.e. a named tuple, returned by the
# ``DataLoader.__iter__()``) to see this in action.

batch = next(iter(dataloader))

# TensorMaps for each sample in the batch joined along the samples axis
# into a single TensorMap
print("batch.descriptor =", batch.descriptor)

# `scalar` data are float objects, so are just grouped and returned in a tuple
print("batch.scalar =", batch.scalar)
assert len(batch.scalar) == batch_size

# `vector` data are torch Tensors, so are vertically stacked into a single
# Tensor
print("batch.vector =", batch.vector)

# %%
#
# Advanced functionality: IndexedDataset
# --------------------------------------
#
# What if we wanted to explicitly define the sample indexes used to store and access
# samples in the dataset? See the next tutorial,
# :ref:`learn-tutorial-indexed-dataset-dataloader`, for more details!
