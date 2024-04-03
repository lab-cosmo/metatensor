import os

import numpy as np
import pytest

import metatensor
from metatensor import Labels, NotEqualError, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture
def test_tensor_map_1() -> TensorMap:
    return metatensor.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )


@pytest.fixture
def test_tensor_map_2() -> TensorMap:
    return metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        # the npz is using DEFLATE compression, metatensor only supports STORED
        use_numpy=True,
    )


@pytest.fixture
def test_tensor_block_1(test_tensor_map_1) -> TensorBlock:
    return test_tensor_map_1.block(0)


@pytest.fixture
def test_tensor_block_2(test_tensor_map_2) -> TensorBlock:
    return test_tensor_map_2.block(0)


@pytest.fixture
def tensor_map() -> TensorMap:
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    block_1 = TensorBlock(
        values=np.full((3, 1, 1), 1.0),
        samples=Labels(["s"], np.array([[0], [2], [4]])),
        components=[Labels(["c"], np.array([[0]]))],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((2, 1, 1), 11.0),
            samples=Labels(["sample", "g"], np.array([[0, -2], [2, 3]])),
            components=[Labels(["c"], np.array([[0]]))],
            properties=block_1.properties,
        ),
    )

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 2.0),
        samples=Labels(["s"], np.array([[0], [1], [3]])),
        components=[Labels(["c"], np.array([[0]]))],
        properties=Labels(["p"], np.array([[3], [4], [5]])),
    )
    block_2.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((3, 1, 3), 12.0),
            samples=Labels(
                ["sample", "g"],
                np.array([[0, -2], [0, 3], [2, -2]]),
            ),
            components=[Labels(["c"], np.array([[0]]))],
            properties=block_2.properties,
        ),
    )

    block_3 = TensorBlock(
        values=np.full((4, 3, 1), 3.0),
        samples=Labels(["s"], np.array([[0], [3], [6], [8]])),
        components=[Labels.range("c", 3)],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_3.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((1, 3, 1), 13.0),
            samples=Labels(
                ["sample", "g"],
                np.array([[1, -2]]),
            ),
            components=[Labels.range("c", 3)],
            properties=block_3.properties,
        ),
    )

    block_4 = TensorBlock(
        values=np.full((4, 3, 1), 4.0),
        samples=Labels(["s"], np.array([[0], [1], [2], [5]])),
        components=[Labels.range("c", 3)],
        properties=Labels(["p"], np.array([[0]])),
    )
    block_4.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.full((2, 3, 1), 14.0),
            samples=Labels(
                ["sample", "g"],
                np.array([[0, 1], [3, 3]]),
            ),
            components=[Labels.range("c", 3)],
            properties=block_4.properties,
        ),
    )

    # TODO: different number of components?

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array([[0, 0], [1, 0], [2, 2], [2, 3]]),
    )

    return TensorMap(keys, [block_1, block_2, block_3, block_4])


def test_self(test_tensor_map_1):
    """check if the metadata of the same tensor are equal"""
    assert metatensor.equal_metadata(test_tensor_map_1, test_tensor_map_1)


def test_self_raise(test_tensor_map_1):
    """check no error raise the metadata of the same tensor are equal"""
    metatensor.equal_metadata_raise(test_tensor_map_1, test_tensor_map_1)


def test_self_block(test_tensor_block_1):
    """check if the metadata of the same tensor are equal"""
    assert metatensor.equal_metadata_block(test_tensor_block_1, test_tensor_block_1)


def test_self_block_raise(test_tensor_block_1):
    """check no error raise if the metadata of the same tensor are equal"""
    metatensor.equal_metadata_block_raise(test_tensor_block_1, test_tensor_block_1)


def test_two_tensors(test_tensor_map_1, test_tensor_map_2):
    """check if the metadata of two tensor maps are equal"""
    assert not metatensor.equal_metadata(test_tensor_map_1, test_tensor_map_2)


def test_two_tensors_raise(test_tensor_map_1, test_tensor_map_2):
    """check error raise if the metadata of two tensor maps are equal"""
    error_message = "should have the same keys names"
    with pytest.raises(NotEqualError, match=error_message):
        metatensor.equal_metadata_raise(test_tensor_map_1, test_tensor_map_2)


def test_two_tensors_block(test_tensor_block_1, test_tensor_block_2):
    """check if the metadata of two tensor maps are equal"""
    assert not metatensor.equal_metadata_block(test_tensor_block_1, test_tensor_block_2)


def test_two_tensors_block_raise(test_tensor_block_1, test_tensor_block_2):
    """check error raise if the metadata of two tensor maps are equal"""
    error_message = (
        "inputs to 'equal_metadata_block_raise' have a different number of components"
    )
    with pytest.raises(NotEqualError, match=error_message):
        metatensor.equal_metadata_block_raise(test_tensor_block_1, test_tensor_block_2)


def test_after_drop(test_tensor_map_1):
    """check if dropping an existing block changes the metadata"""
    new_key = Labels(
        test_tensor_map_1.keys.names, np.array([tuple(test_tensor_map_1.keys[0])])
    )
    new_tensor = metatensor.drop_blocks(test_tensor_map_1, new_key)
    assert not metatensor.equal_metadata(test_tensor_map_1, new_tensor)


def test_after_drop_raise(test_tensor_map_1):
    """check if dropping an existing block changes the metadata"""
    new_key = Labels(
        test_tensor_map_1.keys.names, np.array([tuple(test_tensor_map_1.keys[0])])
    )
    new_tensor = metatensor.drop_blocks(test_tensor_map_1, new_key)
    error_message = "should have the same number of blocks, got 17 and 16"
    with pytest.raises(NotEqualError, match=error_message):
        metatensor.equal_metadata_raise(test_tensor_map_1, new_tensor)


def test_single_nonexisting_meta(test_tensor_map_1):
    """check behavior if non existing metadata is provided"""
    # wrong metadata key alone
    wrong_meta = "wrong"
    error_message = (
        "'wrong' does not refer to metadata to check, choose from "
        "'samples', 'properties' and 'components'"
    )

    with pytest.raises(ValueError, match=error_message):
        metatensor.equal_metadata_raise(
            tensor_1=test_tensor_map_1,
            tensor_2=test_tensor_map_1,
            check=[wrong_meta],
        )

    # wrong metadata key with another correct one
    correct_meta = "properties"
    with pytest.raises(ValueError, match=error_message):
        metatensor.equal_metadata_raise(
            tensor_1=test_tensor_map_1,
            tensor_2=test_tensor_map_1,
            check=[correct_meta, wrong_meta],
        )


def test_single_nonexisting_meta_block(test_tensor_block_1):
    """check behavior if non existing metadata is provided"""
    # wrong metadata key alone
    wrong_meta = "wrong"
    error_message = (
        "'wrong' does not refer to metadata to check, choose from "
        "'samples', 'properties' and 'components'"
    )
    with pytest.raises(ValueError, match=error_message):
        metatensor.equal_metadata_block_raise(
            block_1=test_tensor_block_1,
            block_2=test_tensor_block_1,
            check=[wrong_meta],
        )

    # wrong metadata key with another correct one
    correct_meta = "properties"
    with pytest.raises(ValueError, match=error_message):
        metatensor.equal_metadata_block_raise(
            block_1=test_tensor_block_1,
            block_2=test_tensor_block_1,
            check=[correct_meta, wrong_meta],
        )


def test_key_order(test_tensor_map_1):
    """check changing the key order"""
    keys = test_tensor_map_1.keys
    new_keys = Labels(keys.names, keys.values[::-1])
    new_blocks = [test_tensor_map_1[key].copy() for key in new_keys]
    new_tensor = TensorMap(keys=new_keys, blocks=new_blocks)
    assert metatensor.equal_metadata(test_tensor_map_1, new_tensor)


def test_samples_order(test_tensor_map_1):
    """Test changing the order of the values of the samples should yield False"""
    new_blocks = []
    for key in test_tensor_map_1.keys:
        block = test_tensor_map_1[key].copy()
        samples = Labels(block.samples.names, block.samples.values[::-1])
        new_block = TensorBlock(
            values=block.values,
            samples=samples,
            properties=block.properties,
            components=block.components,
        )
        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=new_block.properties,
                ),
            )
        new_blocks.append(new_block)

    new_tensor = TensorMap(keys=test_tensor_map_1.keys, blocks=new_blocks)
    assert not metatensor.equal_metadata(test_tensor_map_1, new_tensor)


def test_samples_order_block(test_tensor_block_1):
    """Changing the order of the values of the samples should yield False"""
    block = test_tensor_block_1.copy()
    samples = Labels(block.samples.names, block.samples.values[::-1])
    new_block = TensorBlock(
        values=block.values,
        samples=samples,
        properties=block.properties,
        components=block.components,
    )
    for parameter, gradient in block.gradients():
        new_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient.values,
                samples=gradient.samples,
                components=gradient.components,
                properties=new_block.properties,
            ),
        )

    assert not metatensor.equal_metadata_block(test_tensor_block_1, new_block)


def test_properties_order(test_tensor_map_1):
    """changing the order of the values of the properties should yield False"""
    new_blocks = []
    for key in test_tensor_map_1.keys:
        block = test_tensor_map_1[key].copy()
        properties = Labels(block.properties.names, block.properties.values[::-1])
        new_block = TensorBlock(
            values=block.values,
            samples=block.samples,
            properties=properties,
            components=block.components,
        )
        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=new_block.properties,
                ),
            )
        new_blocks.append(new_block)

    new_tensor = TensorMap(keys=test_tensor_map_1.keys, blocks=new_blocks)
    assert not metatensor.equal_metadata(test_tensor_map_1, new_tensor)


def test_properties_order_block(test_tensor_block_1):
    """changing the order of the values of the properties should yield False"""
    block = test_tensor_block_1.copy()
    properties = Labels(block.properties.names, block.properties.values[::-1])
    new_block = TensorBlock(
        values=block.values,
        samples=block.samples,
        properties=properties,
        components=block.components,
    )
    for parameter, gradient in block.gradients():
        new_block.add_gradient(
            parameter=parameter,
            gradient=TensorBlock(
                values=gradient.values,
                samples=gradient.samples,
                components=gradient.components,
                properties=new_block.properties,
            ),
        )

    assert not metatensor.equal_metadata_block(test_tensor_block_1, new_block)


def test_components_order(tensor_map):
    """changing the order of the values of the components should yield False"""
    new_blocks = []
    for key in tensor_map.keys:
        block = tensor_map[key].copy()
        components = [Labels(c.names, c.values[::-1]) for c in block.components]
        new_block = TensorBlock(
            values=block.values,
            samples=block.samples,
            properties=block.properties,
            components=components,
        )
        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values,
                    samples=gradient.samples,
                    components=components,
                    properties=new_block.properties,
                ),
            )
        new_blocks.append(new_block)

    new_tensor = TensorMap(keys=tensor_map.keys, blocks=new_blocks)
    assert not metatensor.equal_metadata(tensor_map, new_tensor)


def test_remove_last_sample(tensor_map):
    """removing the last sample should yield False"""
    new_blocks = []
    for key in tensor_map.keys:
        block = tensor_map[key].copy()
        new_block = TensorBlock(
            values=block.values[:-1],
            samples=Labels(block.samples.names, block.samples.values[:-1]),
            properties=block.properties,
            components=block.components,
        )
        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values[:-1],
                    samples=Labels(
                        gradient.samples.names,
                        gradient.samples.values[:-1],
                    ),
                    components=gradient.components,
                    properties=new_block.properties,
                ),
            )
        new_blocks.append(new_block)

    new_tensor = TensorMap(keys=tensor_map.keys, blocks=new_blocks)
    assert not metatensor.equal_metadata(tensor_map, new_tensor)


def test_remove_last_property(tensor_map):
    """removing the last property should yield False"""
    new_blocks = []
    for key in tensor_map.keys:
        block = tensor_map[key].copy()
        new_block = TensorBlock(
            values=block.values[..., :-1],
            samples=block.samples,
            properties=Labels(block.properties.names, block.properties.values[:-1]),
            components=block.components,
        )
        for parameter, gradient in block.gradients():
            new_block.add_gradient(
                parameter=parameter,
                gradient=TensorBlock(
                    values=gradient.values[..., :-1],
                    samples=gradient.samples,
                    components=gradient.components,
                    properties=new_block.properties,
                ),
            )
        new_blocks.append(new_block)

    new_tensor = TensorMap(keys=tensor_map.keys, blocks=new_blocks)
    assert not metatensor.equal_metadata(tensor_map, new_tensor)


def test_remove_last_component(tensor_map):
    """removing the last component should yield False"""
    new_blocks = []
    for key in tensor_map.keys:
        block = tensor_map[key].copy()

        can_remove = np.all([len(c) > 2 for c in block.components])

        if can_remove:
            components = [Labels(c.names, c.values[:-1]) for c in block.components]
            new_block = TensorBlock(
                values=block.values[:, :-1],
                samples=block.samples,
                properties=block.properties,
                components=components,
            )
            for parameter, gradient in block.gradients():
                new_block.add_gradient(
                    parameter=parameter,
                    gradient=TensorBlock(
                        values=gradient.values[:, :-1],
                        samples=gradient.samples,
                        components=components,
                        properties=new_block.properties,
                    ),
                )
        else:
            new_block = block.copy()
        new_blocks.append(new_block)

    new_tensor = TensorMap(keys=tensor_map.keys, blocks=new_blocks)
    assert not metatensor.equal_metadata(tensor_map, new_tensor)
