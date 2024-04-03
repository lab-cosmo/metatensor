from . import _dispatch
from ._backend import Labels, torch_jit_script


@torch_jit_script
def one_hot(labels: Labels, dimension: Labels):
    """Generates one-hot encoding from a Labels object.

    This function takes two ``Labels`` objects as inputs. The first
    is the one to be converted to one-hot-encoded format, and the
    second contains the name of the label to be extracted and all
    possible values of the one-hot encoding.

    :param labels:
        A ``Labels`` object from which one label will be extracted and
        transformed into a one-hot-encoded array.
    :param dimension:
        A ``Labels`` object that contains a single dimension. The name of
        this label is the same that will be selected from ``labels``,
        and its values correspond to all possible values that the label
        can take.

    :return:
        A two-dimensional ``numpy.ndarray`` or ``torch.Tensor`` containing
        the one-hot encoding along the selected dimension: its first
        dimension matches the one in ``labels``, while the second contains 1
        at the position corresponding to the original label and 0
        everywhere else

    >>> import numpy as np
    >>> import metatensor
    >>> from metatensor import Labels

    >>> # Let's say we have 6 atoms, whose chemical identities
    >>> # are C, H, H, H, C, H:
    >>> original_labels = Labels(
    ...     names=["atom", "types"],
    ...     values=np.array([[0, 6], [1, 1], [2, 1], [3, 1], [4, 6], [5, 1]]),
    ... )
    >>> # Set up a Labels object with all possible elements,
    >>> # including, for example, also O:
    >>> possible_labels = Labels(names=["types"], values=np.array([[1], [6], [8]]))
    >>> # Get the one-hot encoded labels:
    >>> one_hot_encoding = metatensor.one_hot(original_labels, possible_labels)
    >>> print(one_hot_encoding)
    [[0 1 0]
     [1 0 0]
     [1 0 0]
     [1 0 0]
     [0 1 0]
     [1 0 0]]
    """

    if len(dimension.names) != 1:
        raise ValueError(
            "only one label dimension can be extracted as one-hot "
            "encoding. The `dimension` labels contains "
            f"{len(dimension.names)} names"
        )

    name = dimension.names[0]

    indices = _dispatch.zeros_like(dimension.values, [len(labels)])
    labels_name = labels.column(name)
    for i in range(labels_name.shape[0]):
        entry = labels_name[None, i]
        position = dimension.position(entry)
        if position is None:
            raise ValueError(
                f"{name}={entry[0]} is present in the labels, but was not found in "
                "the dimension"
            )
        indices[i] = position

    one_hot_array = _dispatch.eye_like(dimension.values, len(dimension))[
        _dispatch.to_index_array(indices)
    ]
    return one_hot_array
