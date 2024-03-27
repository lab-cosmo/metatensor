.. _torch-api-reference:

TorchScript API reference
=========================

.. py:currentmodule:: metatensor.torch

We provide a special PyTorch C++ extension exporting all of the core metatensor
types in a way compatible with the TorchScript compiler, allowing users to save
and load models based on metatensor everywhere TorchScript is supported. In
particular, this allow to define, train and save a model from Python, and then
load it with pure C++ code, without requiring a Python interpreter. Please refer
to the :ref:`installation instructions <install-torch-script>` to know how to
install the Python and C++ sides of this library.

The classes and functions in the TorchScript API are kept as close as possible
to the classes and functions of the pure Python API, with the explicit goal that
changing from

.. code-block:: python

    import metatensor
    from metatensor import TensorMap, TensorBlock, Labels

to

.. code-block:: python

    import metatensor.torch as metatensor
    from metatensor.torch import TensorMap, TensorBlock, Labels

should be 80% of the work required to make a model developed in Python with
:py:mod:`metatensor.operations` compatible with TorchScript.

The documentation for the three usual core classes of metatensor can be found in
the following pages:

.. toctree::
    :maxdepth: 1

    tensor
    block
    labels
    serialization

All the :ref:`operations <python-api-operations>` are also available in the
``metatensor.torch`` module under the same name. All the functions have the same
behavior, but the versions in ``metatensor.torch`` are annotated with the types
from ``metatensor.torch``, and compatible with TorchScript compilation. For
example :py:func:`metatensor.add()` is available as ``metatensor.torch.add()``.

--------------------------------------------------------------------------------

If you want to use metatensor's TorchScript API from C++, you might be interested
in the following documentation:

.. toctree::
    :maxdepth: 1

    cxx/index

.. _python-vs-torch:

Differences between Python and TorchScript API
----------------------------------------------

The Python and TorchScript interfaces to metatensor are very similar, and have
some level of overlap (it is possible to store data in torch ``Tensor`` with the
Python API), so a big question is when/why should you use one or the other, and
what's the difference between them.

First, the Python and TorchScript API are separate to allow using the Python API
without torch, for example in a pure numpy workflow, or with other array types
such as `Jax`_ arrays, `cupy`_ arrays, *etc.*.

While this works great for a lot of use cases based on torch (defining models,
training them with autograd, …), the TorchScript compiler is more restrictive in
what code it accepts and does not support calling into arbitrary native code
like the Python API does. The TorchScript version of metatensor is here to fill
this gap, teaching the TorchScript compiler what's going on inside metatensor and
how to translate code using metatensor to TorchScript.

Another different is that while the Python API supports storing data in multiple
ways, including storing it on GPU through torch Tensor, the metadata is always
stored on CPU, inside the metatensor shared library. The TorchScript API changes
this to enable storing both the data and metadata on GPU, minimizing data
transfer and making the models faster.

If you only care about PyTorch, we would recommend using the TorchScript API
from the start, to make sure you will be able to export your code to
TorchScript. If you are not using PyTorch, or if you want to write code in an
engine agnostic way, we recommend using the Python API.

.. _Jax: https://jax.readthedocs.io/en/latest/
.. _cupy: https://cupy.dev
