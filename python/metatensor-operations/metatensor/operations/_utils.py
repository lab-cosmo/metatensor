from typing import List, Union

from ._backend import TensorBlock, TensorMap


class NotEqualError(Exception):
    """Exception used to indicate that two metatensor objects are different"""

    pass


def _check_same_keys(a: TensorMap, b: TensorMap, fname: str) -> bool:
    """
    Returns true if the keys of 2 TensorMaps are the same, without specification of the
    order, and false otherwise.
    """
    return not bool(_check_same_keys_impl(a, b, fname))


def _check_same_keys_raise(a: TensorMap, b: TensorMap, fname: str) -> None:
    """
    If the keys of 2 TensorMaps are not the same, raises a NotEqualError, otherwise
    returns None.
    """
    message = _check_same_keys_impl(a, b, fname)
    if message != "":
        raise NotEqualError(message)


def _check_same_keys_impl(a: TensorMap, b: TensorMap, fname: str) -> str:
    """
    Checks if the keys of 2 TensorMaps are the same, without specification of the order.
    Returns an empty str if they are the same, otherwise returns a str message of a
    meaningful error.

    The functions verifies that the key names and length are the same, and that the key
    values are the same without specification of exact order.

    :param a: first :py:class:`TensorMap` for check
    :param b: second :py:class:`TensorMap` for check
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    """

    keys_a = a.keys
    keys_b = b.keys

    if keys_a.names != keys_b.names:
        return (
            f"inputs to '{fname}' should have the same keys names, "
            f"got '{keys_a.names}' and '{keys_b.names}'"
        )

    if len(keys_a) != len(keys_b):
        return (
            f"inputs to '{fname}' should have the same number of blocks, "
            f"got {len(keys_a)} and {len(keys_b)}"
        )

    if not all([keys_b[i] in keys_a for i in range(len(keys_b))]):
        return f"inputs to '{fname}' should have the same keys"

    return ""


def _check_blocks(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Union[List[str], str] = "all",
) -> bool:
    """
    Checks if the metadata of 2 TensorBlocks are the same. If not, returns false.
    Otherwise returns true.
    """
    return not bool(_check_blocks_impl(a, b, fname, check))


def _check_blocks_raise(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Union[List[str], str] = "all",
) -> None:
    """
    Checks if the metadata of two TensorBlocks are the same. If not, raises a
    NotEqualError. If invalid `check` are given, this function raises a `ValueError`.
    Otherwise returns it None.

    The message associated with the exception will contain more information on where the
    two :py:class:`TensorBlock` differ. See :py:func:`_check_blocks_impl` for more
    information on when two :py:class:`TensorBlock` are considered as equal.

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    :param check: Which parts of the metadata to check. This can be a list containing
        any of ``'samples'``, ``'components'``, and ``'properties'``; or the string
        ``'all'`` to check everything. Defaults to ``'all'``.

    :raises: :py:class:`metatensor.NotEqualError` if the metadata of the blocks are
        different
    :raises: :py:class:`ValueError` if an invalid prop name in :param check: is
        given. See :param check: description for valid prop names
    """
    message = _check_blocks_impl(a, b, fname, check)
    if message != "":
        raise NotEqualError(message)


def _check_blocks_impl(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Union[List[str], str] = "all",
) -> str:
    """
    Check if metadata between two TensorBlocks is consistent for an operation.

    The functions verifies that that the metadata of the given check is the same, in
    terms of length, dimension names, and order of the values.

    If they are not the same, an error message as a str is returned. Otherwise, an empty
    str is returned.

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    :param check: Which parts of the metadata to check. This can be a list containing
        any of ``'samples'``, ``'components'``, and ``'properties'``; or the string
        ``'all'`` to check everything. Defaults to ``'all'``.
    """
    if isinstance(check, str):
        if check == "all":
            metadata_to_check = ["samples", "components", "properties"]
        else:
            raise ValueError("`check` must be a list of strings or 'all'")
    else:
        metadata_to_check = check

    for metadata in metadata_to_check:
        if metadata == "samples":
            if not a.samples == b.samples:
                return (
                    f"inputs to '{fname}' should have the same samples, "
                    "but they are not the same or not in the same order"
                )

        elif metadata == "properties":
            if not a.properties == b.properties:
                return (
                    f"inputs to '{fname}' should have the same properties, "
                    "but they are not the same or not in the same order"
                )

        elif metadata == "components":
            if len(a.components) != len(b.components):
                return f"inputs to '{fname}' have a different number of components"

            for c1, c2 in zip(a.components, b.components):
                if not c1 == c2:
                    return (
                        f"inputs to '{fname}' should have the same components, "
                        "but they are not the same or not in the same order"
                    )
        else:
            raise ValueError(
                f"'{metadata}' does not refer to metadata to check, "
                "choose from 'samples', 'properties' and 'components'"
            )
    return ""


def _check_same_gradients(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Union[List[str], str] = "all",
) -> bool:
    """
    Check if metadata between the gradients of 2 TensorBlocks is consistent for an
    operation. If they are the same, true is returned, otherwise false.

    The functions verifies that that the metadata of the given check is the same, in
    terms of length, dimension names, and order of the values.

    If check is None it only checks if the ``'parameters'`` are consistent.

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    :param check: Which parts of the metadata to check. This can be a list containing
        any of ``'samples'``, ``'components'``, and ``'properties'``; or the string
        ``'all'`` to check everything. Defaults to ``'all'``. If you only want to check
        if the two blocks have the same gradients, pass an empty list ``check=[]``.
    """
    return not bool(_check_same_gradients_impl(a, b, fname, check))


def _check_same_gradients_raise(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Union[List[str], str] = "all",
) -> None:
    """
    Check if two TensorBlocks gradients have identical metadata.

    The message associated with the exception will contain more information on where the
    gradients of the two :py:class:`TensorBlock` differ. See
    :py:func:`_check_same_gradients_impl` for more information on when gradients of
    :py:class:`TensorBlock` are considered as equal.

    :param a: first :py:class:`TensorBlock` for check
    :param b: second :py:class:`TensorBlock` for check
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    :param check: Which parts of the metadata to check. This can be a list containing
        any of ``'samples'``, ``'components'``, and ``'properties'``; or the string
        ``'all'`` to check everything. Defaults to ``'all'``. If you only want to check
        if the two blocks have the same gradients, pass an empty list ``check=[]``.

    :raises: :py:class:`metatensor.NotEqualError` if the gradients of the blocks are
        different
    :raises: :py:class:`ValueError` if an invalid prop name in :param check: is
        given. See :param check: description for valid prop names
    """
    message = _check_same_gradients_impl(a, b, fname, check)
    if message != "":
        raise NotEqualError(message)


def _check_same_gradients_impl(
    a: TensorBlock,
    b: TensorBlock,
    fname: str,
    check: Union[List[str], str] = "all",
) -> str:
    """
    Check if metadata between the gradients of two TensorBlocks is consistent for an
    operation.

    The functions verifies that that the 2 TensorBlocks have the same gradient
    parameters, then checks the metadata of the given ``check`` is the same, in terms of
    length, dimension names, and order of the values.

    If they are not the same, an error message as a str is returned. Otherwise, an empty
    str is returned. If the 2 blocks have no gradients, an empty string is returned.

    :param a: first :py:class:`TensorBlock` whose gradients are to be checked
    :param b: second :py:class:`TensorBlock` whose gradients are to be checked
    :param fname: name of the function where the check is performed. The input will be
        used to generate a meaningful error message.
    :param check: Which parts of the metadata to check. This can be a list containing
        any of ``'samples'``, ``'components'``, and ``'properties'``; or the string
        ``'all'`` to check everything. Defaults to ``'all'``. If you only want to check
        if the two blocks have the same gradients, pass an empty list ``check=[]``.
    """
    if isinstance(check, str):
        if check == "all":
            metadata_to_check = ["samples", "components", "properties"]
        else:
            raise ValueError("`check` must be a list of strings or 'all'")
    else:
        metadata_to_check = check

    err_msg = f"inputs to '{fname}' should have the same gradients: "
    gradients_list_a = a.gradients_list()
    gradients_list_b = b.gradients_list()

    if len(gradients_list_a) != len(gradients_list_b) or (
        not all([parameter in gradients_list_b for parameter in gradients_list_a])
    ):
        return f"inputs to '{fname}' should have the same gradient parameters"

    for parameter, grad_a in a.gradients():
        grad_b = b.gradient(parameter)

        for metadata in metadata_to_check:
            err_msg_1 = (
                f"gradient '{parameter}' {metadata} are not the same or not in the "
                "same order"
            )

            if metadata == "samples":
                if not grad_a.samples == grad_b.samples:
                    return err_msg + err_msg_1

            elif metadata == "properties":
                if not grad_a.properties == grad_b.properties:
                    return err_msg + err_msg_1

            elif metadata == "components":
                if len(grad_a.components) != len(grad_b.components):
                    extra = (
                        f"gradient '{parameter}' have different number of components"
                    )
                    return err_msg + extra

                for c1, c2 in zip(grad_a.components, grad_b.components):
                    if not c1 == c2:
                        return err_msg + err_msg_1
            else:
                raise ValueError(
                    f"{metadata} is not a valid property to check, "
                    "choose from 'samples', 'properties' and 'components'"
                )
    return ""


def _check_gradient_presence_raise(
    block: TensorBlock,
    parameters: List[str],
    fname: str,
) -> None:
    """
    For a single TensorBlock checks if each of the passed ``parameters`` are present as
    parameters of its gradients. If all of them are present, None is returned. Otherwise
    a ValueError is raised.
    """
    for parameter in parameters:
        if parameter not in block.gradients_list():
            raise ValueError(
                f"requested gradient '{parameter}' in '{fname}' is not defined "
                "in this tensor"
            )
