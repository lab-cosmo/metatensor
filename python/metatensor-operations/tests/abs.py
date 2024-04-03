"""Unit tests for the :py:func:`metatensor.abs` function."""

import numpy as np
import pytest

import metatensor
from metatensor import TensorBlock, TensorMap

from . import utils


def tensor_complex():
    """Manipulate tensor to be a suitable test for the `abs` function."""

    tensor = utils.tensor()

    blocks = []
    for block in tensor:
        new_block = TensorBlock(
            values=-block.values + 3j,
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=-gradient.values - 2j,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=gradient.properties,
                ),
            )

        blocks.append(new_block)

    return TensorMap(keys=tensor.keys, blocks=blocks)


def tensor_complex_result():
    blocks = []
    tensor = tensor_complex()
    for block in tensor:
        new_block = TensorBlock(
            values=np.abs(block.values).astype(np.complex128),
            samples=block.samples,
            components=block.components,
            properties=block.properties,
        )

        sign = block.values[0, 0] / np.abs(block.values[0, 0])

        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=sign * gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=gradient.properties,
                ),
            )

        blocks.append(new_block)

    return TensorMap(keys=tensor.keys, blocks=blocks)


@pytest.mark.parametrize("gradients", (True, False))
def test_abs(gradients):
    if not gradients:
        tensor = metatensor.remove_gradients(tensor_complex())
        tensor_result = metatensor.remove_gradients(tensor_complex_result())
    else:
        tensor = tensor_complex()
        tensor_result = tensor_complex_result()

    tensor_abs = metatensor.abs(tensor)
    metatensor.allclose_raise(tensor_abs, tensor_result)

    # Check the tensors haven't be modified in place
    assert not metatensor.equal(tensor_abs, tensor)
