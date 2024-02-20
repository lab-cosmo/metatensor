import os

import metatensor
from metatensor import TensorBlock, TensorMap

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

tensor = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        use_numpy=True,
    )


def _incontiguous_tensor(tensor: TensorMap) -> TensorMap:
    """
    Make a non-contiguous TensorMap by reversing the order in all the main value
    blocks and the gradient blocks.
    """

    keys = tensor.keys
    new_blocks = []

    for key, block in tensor.items():
        # Create a new TM with a non-contig array
        new_block = _incontiguous_block(block)
        new_blocks.append(new_block)

    return TensorMap(keys=keys, blocks=new_blocks)


def _incontiguous_block(block: TensorBlock) -> TensorBlock:
    """
    Make a non-contiguous block by reversing the order in both the main value block and
    the gradient block(s).
    """
    new_vals = block.values.copy()[::-1, ::-1]
    new_block = TensorBlock(
        values=new_vals,
        samples=block.samples,
        components=block.components,
        properties=block.properties,
    )
    # Make gradients non-contig
    for param, gradient in block.gradients():
        new_grad = gradient.values.copy()[::-1, ::-1]
        new_gradient = TensorBlock(
            values=new_grad,
            samples=gradient.samples,
            components=gradient.components,
            properties=gradient.properties,
        )
        new_block.add_gradient(param, new_gradient)

    return new_block


def test_is_contiguous_block():
    assert not metatensor.is_contiguous_block(_incontiguous_block(tensor.block(0)))
    assert metatensor.is_contiguous_block(
        metatensor.make_contiguous_block(_incontiguous_block(tensor.block(0)))
        )


def test_is_contiguous():
    assert not metatensor.is_contiguous(_incontiguous_tensor(tensor))
    assert metatensor.is_contiguous(
        metatensor.make_contiguous(_incontiguous_tensor(tensor))
        )
    
    
    