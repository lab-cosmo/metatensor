import re
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

from ._backend import torch_jit_is_scripting, torch_jit_script


def parse_version(version):
    match = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
    if match:
        return tuple(map(int, match.groups()))
    else:
        raise ValueError("Invalid version string format")


try:
    import torch
    from torch import Tensor as TorchTensor

    torch_dtype = torch.dtype
    torch_device = torch.device
    torch_version = parse_version(torch.__version__)

except ImportError:

    class TorchTensor:
        pass

    class torch_dtype:
        pass

    class torch_device:
        pass

    torch_version = (0, 0, 0)


UNKNOWN_ARRAY_TYPE = (
    "unknown array type, only numpy arrays and torch tensors are supported"
)


def _check_all_torch_tensor(arrays: List[TorchTensor]):
    for array in arrays:
        if not isinstance(array, TorchTensor):
            raise TypeError(
                f"expected argument to be a torch.Tensor, but got {type(array)}"
            )


def _check_all_np_ndarray(arrays):
    for array in arrays:
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"expected argument to be a np.ndarray, but got {type(array)}"
            )


def sum(array, axis: Optional[int] = None):
    """
    Returns the sum of the elements in the array at the axis.

    It is equivalent of np.sum(array, axis=axis) and torch.sum(tensor, dim=axis)
    """
    if isinstance(array, TorchTensor):
        return torch.sum(array, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.sum(array, axis=axis).astype(array.dtype)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def abs(array):
    """
    Returns the absolute value of the elements in the array.

    It is equivalent of np.abs(array) and torch.abs(tensor)
    """
    if isinstance(array, TorchTensor):
        return torch.abs(array)
    elif isinstance(array, np.ndarray):
        return np.abs(array).astype(array.dtype)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def all(a, axis: Optional[int] = None):
    """Test whether all array elements along a given axis evaluate to True.

    This function has the same behavior as
    ``np.all(array,axis=axis)``.
    """
    if isinstance(a, TorchTensor):
        # torch.all has two implementation, and picks one depending if more than one
        # parameter is given. The second one does not supports setting dim to `None`
        if axis is None:
            return torch.all(a)
        else:
            return torch.all(a, dim=axis)
    elif isinstance(a, np.ndarray):
        return np.all(a, axis=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def allclose(
    a: TorchTensor,
    b: TorchTensor,
    rtol: float,
    atol: float,
    equal_nan: bool = False,
):
    """Compare two arrays using ``allclose``

    This function has the same behavior as
    ``np.allclose(array1, array2, rtol, atol, equal_nan)``.
    """
    if isinstance(a, TorchTensor):
        _check_all_torch_tensor([b])
        return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    elif isinstance(a, np.ndarray):
        _check_all_np_ndarray([b])
        return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def argsort_labels_values(labels_values, reverse: bool = False):
    """
    Similar to :py:func:`np.argsort`, but sort the rows as one aggregated
    tuple.

    :param labels_values: numpy.array or torch.Tensor
    :param reverse: if true, order is descending

    :return: indices corresponding to the sorted values in ``labels_values``
    """
    if isinstance(labels_values, TorchTensor):
        if labels_values.shape[1] == 1:
            # Index is appended at the end to get
            # the indices corresponding to the
            # sorted values. Because we append the indices at
            # the end and since metadata is unique, we do not affect
            # the sorted order. Using `torch.argsort` is not an option
            # because only on Tensors and sorting a Tensor is
            # different than sorting a List(Tuple)
            data = [(int(row[0]), i) for i, row in enumerate(labels_values)]
            return torch.tensor(
                [i[-1] for i in sort_list_2(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 2:
            data = [
                (int(row[0]), int(row[1]), i) for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_3(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 3:
            data = [
                (int(row[0]), int(row[1]), int(row[2]), i)
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_4(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 4:
            data = [
                (int(row[0]), int(row[1]), int(row[2]), int(row[3]), i)
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_5(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 5:
            data = [
                (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), i)
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_6(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 6:
            data = [
                (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    i,
                )
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_7(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 7:
            data = [
                (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                    i,
                )
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_8(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 8:
            data = [
                (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                    int(row[7]),
                    i,
                )
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_9(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        if labels_values.shape[1] == 9:
            data = [
                (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                    int(row[7]),
                    int(row[8]),
                    i,
                )
                for i, row in enumerate(labels_values)
            ]
            return torch.tensor(
                [i[-1] for i in sort_list_10(data, reverse=reverse)],
                dtype=torch.int64,
                device=labels_values.device,
            )
        else:
            raise Exception("labels_values.shape[1]> 9 is not supported")
    elif isinstance(labels_values, np.ndarray):
        # Index is appended at the end to get the indices corresponding to the
        # sorted values. Because we append the indices at the end and since metadata
        # is unique, we do not affect the sorted order.
        list_tuples: List[List[int]] = labels_values.tolist()
        for i in range(len(labels_values)):
            list_tuples[i].append(i)
        list_tuples.sort(reverse=reverse)
        return np.array(list_tuples)[:, -1]
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def bincount(input, weights: Optional[TorchTensor] = None, minlength: int = 0):
    """Count number of occurrences of each value in array of non-negative ints.
    Equivalent of ``numpy.bitcount(input, weights, minlength)``

    Args:
        input (array_like): Input array.
        weights (array_like, optional): Weights, array of the same shape as input.
                                        Defaults to None.
        minlength (int, optional): A minimum number of bins for the output array.
                                        Defaults to 0.
    Note:
        In the main code this function is only used with ``np.ndarray`` as an input,
        since the indexes comes from labels which are always ``np.ndarray``. If you
        want to use the result of ``bincount`` to operate with ``TorchTensor``, you
        should follow this with a call to `_dispatch.array_like_data` to transform it
        in a ``TorchTensor`` with the desired properties.
    """
    if isinstance(input, TorchTensor):
        if weights is not None:
            _check_all_torch_tensor([weights])
        return torch.bincount(input, weights=weights, minlength=minlength)
    elif isinstance(input, np.ndarray):
        if weights is not None:
            _check_all_np_ndarray([weights])
        return np.bincount(input, weights=weights, minlength=minlength)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def bool_array_like(bool_list: List[bool], like):
    """
    Converts the input list of bool to a numpy array or torch tensor
    based on the type of `like`.
    """
    if isinstance(like, TorchTensor):
        return torch.tensor(bool_list, dtype=torch.bool, device=like.device)
    elif isinstance(like, np.ndarray):
        return np.array(bool_list).astype(bool)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def concatenate(arrays: List[TorchTensor], axis: int):
    """
    Concatenate a group of arrays along a given axis.

    This function has the same behavior as ``numpy.concatenate(arrays, axis)``
    and ``torch.concatenate(arrays, axis)``.
    """
    if isinstance(arrays[0], TorchTensor):
        _check_all_torch_tensor(arrays)
        return torch.cat(arrays, axis)
    elif isinstance(arrays[0], np.ndarray):
        _check_all_np_ndarray(arrays)
        return np.concatenate(arrays, axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def copy(array):
    """Returns a copy of ``array``.
    The new data is not shared with the original array"""
    if isinstance(array, TorchTensor):
        return array.clone()
    elif isinstance(array, np.ndarray):
        return array.copy()
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def detach(array):
    """Returns a new array, detached from the underlying computational graph, if any"""
    if isinstance(array, TorchTensor):
        return array.detach()
    elif isinstance(array, np.ndarray):
        # nothing to do
        return array
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def dot(A, B):
    """Compute dot product of two arrays.

    This function has the same behavior as  ``np.dot(A, B.T)``, and assumes the
    second array is 2-dimensional.
    """
    if isinstance(A, TorchTensor):
        _check_all_torch_tensor([B])
        assert len(B.shape) == 2
        return A @ B.T
    elif isinstance(A, np.ndarray):
        _check_all_np_ndarray([B])
        shape1 = A.shape
        assert len(B.shape) == 2
        # Using matmul/@ is the recommended way in numpy docs for 2-dimensional
        # matrices
        if len(shape1) == 2:
            return A @ B.T
        else:
            return np.dot(A, B.T)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def empty_like(array, shape: Optional[List[int]] = None, requires_grad: bool = False):
    """
    Create an uninitialized array, with the given ``shape``, and similar dtype,
    device and other options as ``array``.

    If ``shape`` is :py:obj:`None`, the array shape is used instead.
    ``requires_grad`` is only used for torch tensors, and set the corresponding
    value on the returned array.

    This is the equivalent to ``np.empty_like(array, shape=shape)``.
    """
    if isinstance(array, TorchTensor):
        if shape is None:
            shape = array.size()
        return torch.empty(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        ).requires_grad_(requires_grad)
    elif isinstance(array, np.ndarray):
        return np.empty_like(array, shape=shape, subok=False)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def eye_like(array, size: int):
    """
    Create an identity matrix with the given ``size``, and the same
    dtype and device as ``array``.
    """

    if isinstance(array, TorchTensor):
        return torch.eye(size).to(array.dtype).to(array.device)
    elif isinstance(array, np.ndarray):
        return np.eye(size, dtype=array.dtype)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def get_device(array):
    """
    Returns the device of the array if it is a
    ``torch.Tensor``, or "cpu" if it is a ``numpy.ndarray``.
    """

    if isinstance(array, TorchTensor):
        return array.device
    elif isinstance(array, np.ndarray):
        return "cpu"
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def index_add(output_array, input_array, index):
    """Accumulates in `output_array`
    the elements of `array`
    by adding to the indices in the order given in index.

    it is equivalent of torch's:

    output_array.index_add_(0, torch.tensor(index),input_array)

    """
    index = to_index_array(index)
    if isinstance(input_array, TorchTensor):
        if not isinstance(index, TorchTensor):
            index = torch.tensor(index).to(device=input_array.device)

        _check_all_torch_tensor([output_array, input_array, index])
        output_array.index_add_(0, index, input_array)
    elif isinstance(input_array, np.ndarray):
        _check_all_np_ndarray([output_array, input_array, index])
        np.add.at(output_array, index, input_array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def int_array_like(int_list: List[int], like):
    """
    Converts the input list of int to a numpy array or torch tensor
    based on the device of `like`.

    If the backend is torch and the device is "meta",
    the device is set to "cpu". This is useful in case where
    we create labels for a block that is on the meta device.
    In that case, `int_list` are the labels, and `like` are the block
    values.
    """
    if isinstance(like, TorchTensor):
        if like.device.type == "meta":
            device = torch.device("cpu")
        else:
            device = like.device
        return torch.tensor(int_list, dtype=torch.int64, device=device)
    elif isinstance(like, np.ndarray):
        return np.array(int_list).astype(np.int64)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def lstsq(X, Y, rcond: Optional[float], driver: Optional[str] = None):
    """
    Computes a solution to the least squares problem of a system of linear
    equations.

    Computes the vector x that approximately solves the equation ``array1 @ x =
    array2``. This function has the same behavior as ``numpy.linalg.lstsq(X,
    Y)``.

    :param rcond: Cut-off ratio for small singular values of a.
        WARNING: the default rcond=None for numpy and torch is different
        numpy -> rcond is the machine precision times max(M, N).
                with M, N being the dimensions of array1
        torch -> rcond is the machine precision,
                to have this behavior in numpy use
                rcond=-1

    :param driver: Used only in torch (ignored if numpy is used).
            Chooses the LAPACK/MAGMA function that will be used.
            Possible values: for CPU 'gels', 'gelsy', 'gelsd', 'gelss'.
                             for GPU  the only valid driver is 'gels',
                             which assumes that A is full-rank
            see https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html
            for a full description
            If None, 'gelsy' is used for CPU inputs
            and 'gels' for CUDA inputs. Default: None
    """
    if isinstance(X, TorchTensor):
        _check_all_torch_tensor([Y])
        return torch.linalg.lstsq(X, Y, rcond=rcond, driver=driver)[0]
    elif isinstance(X, np.ndarray):
        _check_all_np_ndarray([Y])
        return np.linalg.lstsq(X, Y, rcond=rcond)[0]
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def mask(array, axis: int, mask):
    """
    Applies a boolean mask along the specified axis.

    This function indexes an `array` along an `axis` using the boolean values inside
    `mask`. Only the indices for which `mask` is True will be part of the output array.

    This operation is useful because array[..., mask] — i.e. indexing with a mask after
    an ellipsis — is not supported in torchscript.
    """
    indices = where(mask)[0]  # use _dispatch.where to find the indices
    if isinstance(array, TorchTensor):
        return torch.index_select(array, dim=axis, index=indices)
    elif isinstance(array, np.ndarray):
        if isinstance(indices, TorchTensor):
            indices = indices.detach().cpu().numpy()
        return np.take(array, indices, axis=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def nan_to_num(
    X,
    nan: float = 0.0,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
):
    """Equivalent to np.nan_to_num(X, nan, posinf, neginf)"""
    if isinstance(X, TorchTensor):
        return torch.nan_to_num(X, nan=nan, posinf=posinf, neginf=neginf)
    elif isinstance(X, np.ndarray):
        return np.nan_to_num(X, nan=nan, posinf=posinf, neginf=neginf)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def norm(array, axis=None):
    """Compute the 2-norm (Frobenius norm for matrices) of the input array.

    This calls the equivalent of ``np.linalg.norm(array, axis=axis)``, see this
    function for more documentation.
    """
    if isinstance(array, TorchTensor):
        return np.linalg.norm(array, axis=axis)
    elif isinstance(array, np.ndarray):
        return torch.linalg.norm(array, dim=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def ones_like(array, shape: Optional[List[int]] = None, requires_grad: bool = False):
    """
    Create an array filled with ones, with the given ``shape``, and similar
    dtype, device and other options as ``array``.

    If ``shape`` is :py:obj:`None`, the array shape is used instead.
    ``requires_grad`` is only used for torch tensors, and set the corresponding
    value on the returned array.

    This is the equivalent to ``np.ones_like(array, shape=shape)``.
    """

    if isinstance(array, TorchTensor):
        if shape is None:
            shape = array.size()
        return torch.ones(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        ).requires_grad_(requires_grad)
    elif isinstance(array, np.ndarray):
        return np.ones_like(array, shape=shape, subok=False)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def rand_like(array, shape: Optional[List[int]] = None, requires_grad: bool = False):
    """
    Create an array with values randomly sampled from the uniform distribution
    in the ``[0, 1)`` interval, with the given ``shape``, and similar dtype,
    device and other options as ``array``.

    If ``shape`` is :py:obj:`None`, the array shape is used instead.
    ``requires_grad`` is only used for torch tensors, and set the corresponding
    value on the returned array.
    """

    if isinstance(array, TorchTensor):
        if shape is None:
            shape = array.shape
        return torch.rand(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        ).requires_grad_(requires_grad)
    elif isinstance(array, np.ndarray):
        if shape is None:
            shape = array.shape
        return np.random.rand(*shape).astype(array.dtype)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def requires_grad(array, value: bool):
    """
    Set ``requires_grad`` to ``value`` on ``array``. This does nothing on numpy arrays.
    """
    if isinstance(array, TorchTensor):
        if value and array.requires_grad:
            warnings.warn(
                "setting `requires_grad=True` again on a Tensor will detach the Tensor",
                stacklevel=1,  # show the warning as coming from the operation
            )
        return array.detach().requires_grad_(value)
    elif isinstance(array, np.ndarray):
        if value:
            warnings.warn(
                "`requires_grad=True` does nothing for numpy arrays",
                stacklevel=1,  # show the warning as coming from the operation
            )
        return array
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def sign(array):
    """
    Returns an indication of the sign of the elements in the array.

    It is equivalent of np.sign(array) (as defined in v2.0) and torch.sgn(tensor)
    """
    if isinstance(array, TorchTensor):
        return torch.sgn(array)
    elif isinstance(array, np.ndarray):
        if np.issubdtype(array.dtype, np.complexfloating):
            return array / np.abs(array)
        else:
            return np.sign(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def solve(X, Y):
    """
    Computes the solution of a square system of linear equations with a unique
    solution.

    This function has the same behavior as ``numpy.linalg.solve(X, Y)``.
    """
    if isinstance(X, TorchTensor):
        _check_all_torch_tensor([Y])
        result = torch.linalg.solve(X, Y)
        return result
    elif isinstance(X, np.ndarray):
        _check_all_np_ndarray([Y])
        return np.linalg.solve(X, Y)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def sqrt(array):
    """Compute the square root  of the input array.

    This calls the equivalent of ``np.sqrt(array)``, see this
    function for more documentation.
    """
    if isinstance(array, TorchTensor):
        return torch.sqrt(array)
    elif isinstance(array, np.ndarray):
        return np.sqrt(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def stack(arrays: List[TorchTensor], axis: int):
    """
    Stack a group of arrays along a new axis.

    This function has the same behavior as ``numpy.stack(arrays, axis)``
    and ``torch.stack(arrays, axis)``.
    """
    if isinstance(arrays[0], TorchTensor):
        _check_all_torch_tensor(arrays)
        return torch.stack(arrays, axis)
    elif isinstance(arrays[0], np.ndarray):
        _check_all_np_ndarray(arrays)
        return np.stack(arrays, axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def take(array, indices, axis: int):
    """
    See :py:func:`torch.index_select` or :py:func:`numpy.take` as reference.

    Because numpy and torch have different APIs we went for the more limited one
    in torch not supporting ``mode`` argument. The argument ``out`` is not supported
    by TorchScript, or at least it is nontrivial to add

    :param array: the array the elements are returned from
    :param indices: the indices to take long an axes
    :param axis: axis of array to take

    :return: the elments at the indices in the array along dimension specified by the
    axis
    """
    if isinstance(array, TorchTensor):
        return torch.index_select(array, dim=axis, index=indices)
    elif isinstance(array, np.ndarray):
        return np.take(array, indices, axis=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def to(
    array,
    backend: Optional[str] = None,
    dtype: Optional[torch_dtype] = None,
    device: Optional[Union[str, torch_device]] = None,
):
    """Convert the array to the specified backend."""

    # Convert torch Tensor
    if isinstance(array, TorchTensor):
        if backend is None:  # Infer the target backend
            backend = "torch"
        if dtype is None:
            dtype = array.dtype
        if device is None:
            device = array.device
        if isinstance(device, str):
            device = torch.device(device)

        # Perform the conversion
        if backend == "torch":
            return array.to(dtype=dtype).to(device=device)

        elif backend == "numpy":
            if torch_jit_is_scripting():
                raise ValueError("cannot call numpy conversion when torch-scripting")
            else:
                return array.detach().cpu().numpy()

        else:
            raise ValueError(f"Unknown backend: {backend}")

    # Convert numpy array
    elif isinstance(array, np.ndarray):
        if backend is None:  # Infer the target backend
            backend = "numpy"

        # Perform the conversion
        if backend == "numpy":
            return np.array(array, dtype=dtype)

        elif backend == "torch":
            return torch.tensor(array, dtype=dtype, device=device)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    else:
        # Only numpy and torch arrays currently supported
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def _to_index_array_checks(array):
    if len(array.shape) != 1:
        raise ValueError("Index arrays must be 1D")

    if isinstance(array, TorchTensor):
        if torch.is_floating_point(array):
            raise ValueError("Index arrays must be integers")
    elif isinstance(array, np.ndarray):
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError("Index arrays must be integers")
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


if torch_version >= (2, 0, 0):

    def to_index_array(array):
        """
        Returns an array that is suitable for indexing a dimension of
        a different array.
        """
        _to_index_array_checks(array)
        return array

else:

    def to_index_array(array):
        """
        Returns an array that is suitable for indexing a dimension of
        a different array, converting torch data to long/64-bit integers
        """
        _to_index_array_checks(array)

        if isinstance(array, TorchTensor):
            return array.to(torch.long)
        elif isinstance(array, np.ndarray):
            return array
        else:
            raise TypeError(UNKNOWN_ARRAY_TYPE)


def unique(array, axis: Optional[int] = None):
    """Find the unique elements of an array."""
    if isinstance(array, TorchTensor):
        return torch.unique(array, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.unique(array, axis=axis)


def unique_with_inverse(array, axis: Optional[int] = None):
    """Return the unique entries of `array`, along with inverse indices.

    Specifying return_inverse=True explicitly seems to be necessary, as
    there is apparently no way to mark something as a compile-time constant
    in torchscript.
    """
    if isinstance(array, TorchTensor):
        return torch.unique(array, return_inverse=True, dim=axis)
    elif isinstance(array, np.ndarray):
        return np.unique(array, return_inverse=True, axis=axis)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def where(array):
    """Return the indices where `array` is True.

    This function has the same behavior as ``np.where(array)``.
    """
    if isinstance(array, TorchTensor):
        return torch.where(array)
    elif isinstance(array, np.ndarray):
        return np.where(array)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


def zeros_like(array, shape: Optional[List[int]] = None, requires_grad: bool = False):
    """
    Create an array filled with zeros, with the given ``shape``, and similar
    dtype, device and other options as ``array``.

    If ``shape`` is :py:obj:`None`, the array shape is used instead.
    ``requires_grad`` is only used for torch tensors, and set the corresponding
    value on the returned array.

    This is the equivalent to ``np.zeros_like(array, shape=shape)``.
    """
    if isinstance(array, TorchTensor):
        if shape is None:
            shape = array.size()

        return torch.zeros(
            shape,
            dtype=array.dtype,
            layout=array.layout,
            device=array.device,
        ).requires_grad_(requires_grad)
    elif isinstance(array, np.ndarray):
        return np.zeros_like(array, shape=shape, subok=False)
    else:
        raise TypeError(UNKNOWN_ARRAY_TYPE)


@torch_jit_script
def sort_list_2(
    data: List[Tuple[int, int]], reverse: bool = False
) -> List[Tuple[int, int]]:
    # Using the `reverse` flag of sorted does not work with TorchScript
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_3(
    data: List[Tuple[int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int]]:
    # Using the `reverse` flag of sorted does not work with TorchScript
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_4(
    data: List[Tuple[int, int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int, int]]:
    # Using the `reverse` flag of sorted does not work with TorchScript
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_5(
    data: List[Tuple[int, int, int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int, int, int]]:
    # Using the `reverse` flag of sorted does not work with TorchScript
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_6(
    data: List[Tuple[int, int, int, int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int, int, int, int]]:
    # Using the `reverse` flag of sorted does not work with TorchScript
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_7(
    data: List[Tuple[int, int, int, int, int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int, int, int, int, int]]:
    # Using the `reverse` flag of sorted does not work with TorchScript
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_8(
    data: List[Tuple[int, int, int, int, int, int, int, int]], reverse: bool = False
) -> List[Tuple[int, int, int, int, int, int, int, int]]:
    # Using the `reverse` flag of sorted does not work with TorchScript
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_9(
    data: List[Tuple[int, int, int, int, int, int, int, int, int]],
    reverse: bool = False,
) -> List[Tuple[int, int, int, int, int, int, int, int, int]]:
    # Using the `reverse` flag of sorted does not work with TorchScript
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))


@torch_jit_script
def sort_list_10(
    data: List[Tuple[int, int, int, int, int, int, int, int, int, int]],
    reverse: bool = False,
) -> List[Tuple[int, int, int, int, int, int, int, int, int, int]]:
    # Using the `reverse` flag of sorted does not work with TorchScript
    if reverse:
        return list(sorted(data))[::-1]
    else:
        return list(sorted(data))
