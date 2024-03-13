.. _operations-and-torch:

Operations and PyTorch
======================

`PyTorch`_ is a very popular framework for machine learning, providing multiple
tools to make writing and training models easier. There are two ways to use
the operations with PyTorch:

- Using the pure Python version of metatensor, one can store values in a
  :py:class:`metatensor.TensorBlock` using :py:class:`torch.Tensor`. In this
  case, all operations will be compatible with torch autograd (automatic
  gradient tracking and differentiation). This allows to train models from
  Python, but not to export the models to run without the Python interpreter.
  When running a model with the pure Python version of metatensor, you should use
  the operations from ``metatensor.<operation_name>``.

- When using the :ref:`TorchScript version of metatensor <python-api-torch>`,
  one can also compile the Python code to TorchScript and then run the model
  without a Python interpreter. This is particularly useful to export and then
  use an already trained model, for example to run molecular simulations. If you
  want to do this, you should use classes and operations from
  ``metatensor.torch``, i.e. :py:class:`metatensor.torch.TensorMap` and using
  the operation from ``metatensor.torch.<operation_name>``. All the operation
  are available in the ``metatensor.torch`` module.

.. _PyTorch: https://pytorch.org/


Handling of gradients in the operations
---------------------------------------

There are two ways in which the gradients of some values can be computed with
metatensor operations. Let's consider for example an operation that takes one
TensorMap :math:`X` and returns some transformation of that TensorMap
:math:`y = f(X)`.

1) if you are using :py:class:`torch.Tensor` as arrays — either with the pure
   Python (``metatensor``) or TorchScript (``metatensor.torch``) backend — then
   all transformations will be recorded in the computational graph of the output
   data. This means that if ``y_block.values.requires_grad`` is ``True``;
   ``y_block.values.grad_fn`` will be set for all blocks in :math:`y`, and
   calling ``y_block.values.backward()`` will propagate the gradient through the
   transformations applied by the operation.
2) your input TensorMap :math:`X` contains :ref:`explicit gradients
   <core-tutorial-gradients>`, stored in ``x_block.gradient(<parameter>)`` for
   all blocks. The operation will forward propagate these gradients (or raise an
   error if they can not do so), and the output blocks will contain the same set
   of explicit gradients, now containing the gradients of :math:`y` with respect
   to the same parameters.


These two methods can be used together: you can store explicit gradients in
``x_block.gradient(<parameter>)`` using :py:class:`torch.Tensor`, forward
propagate these gradients to some final quantity; then compute a loss taking
into account the gradients of this quantity (e.g. :math:`\ell = |y -
y^\text{ref}|^2 + |\nabla y - \nabla y^\text{ref}|^2`); and finally call
``backward`` on :math:`\ell`. This would allow to train a model on gradients of
a quantity replacing a double backward propagation with a single forward and a
single backward propagation of gradients.
