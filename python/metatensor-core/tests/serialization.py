import io
import os
import pickle

import numpy as np
import pytest

import metatensor
from metatensor import Labels, MetatensorError, TensorBlock, TensorMap

from . import utils


PICKLE_PROTOCOLS = (4, 5)


@pytest.fixture
def tensor():
    return utils.tensor()


@pytest.fixture
def labels():
    return utils.tensor().keys


@pytest.fixture
def tensor_zero_len_block():
    return utils.tensor_zero_len_block()


@pytest.mark.parametrize("use_numpy", (True, False))
@pytest.mark.parametrize("memory_buffer", (True, False))
@pytest.mark.parametrize("standalone_fn", (True, False))
def test_load(use_numpy, memory_buffer, standalone_fn):
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "data.npz",
    )

    if memory_buffer:
        with open(path, "rb") as fd:
            buffer = fd.read()

        assert isinstance(buffer, bytes)
    else:
        file = path

    if standalone_fn:
        if memory_buffer:
            tensor = metatensor.io.load_buffer(buffer, use_numpy=use_numpy)
        else:
            tensor = metatensor.load(file, use_numpy=use_numpy)
    else:
        if memory_buffer:
            tensor = TensorMap.load_buffer(buffer, use_numpy=use_numpy)
        else:
            tensor = TensorMap.load(file, use_numpy=use_numpy)

    assert isinstance(tensor, TensorMap)
    assert tensor.keys.names == [
        "o3_lambda",
        "o3_sigma",
        "center_type",
        "neighbor_type",
    ]
    assert len(tensor.keys) == 27

    block = tensor.block(o3_lambda=2, center_type=6, neighbor_type=1)
    assert block.samples.names == ["system", "atom"]
    assert block.values.shape == (9, 5, 3)

    gradient = block.gradient("positions")
    assert gradient.samples.names == ["sample", "system", "atom"]
    assert gradient.values.shape == (59, 3, 5, 3)


# using tmpdir as pytest-built-in fixture
# https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html#the-tmpdir-and-tmpdir-factory-fixtures
@pytest.mark.parametrize("use_numpy", (True, False))
@pytest.mark.parametrize("memory_buffer", (True, False))
@pytest.mark.parametrize("standalone_fn", (True, False))
def test_save(use_numpy, memory_buffer, standalone_fn, tmpdir, tensor):
    """Check that as saved file loads fine with numpy."""
    with tmpdir.as_cwd():
        if memory_buffer:
            if standalone_fn:
                buffer = metatensor.io.save_buffer(tensor, use_numpy=use_numpy)
            else:
                buffer = tensor.save_buffer(use_numpy=use_numpy)

            file = io.BytesIO(buffer)

        else:
            file = "serialize-test.npz"
            if standalone_fn:
                metatensor.save(file, tensor, use_numpy=use_numpy)
            else:
                tensor.save(file, use_numpy=use_numpy)

        data = np.load(file)

    assert len(data.keys()) == 29

    assert _npz_labels(data["keys"]) == tensor.keys
    for i, block in enumerate(tensor.blocks()):
        prefix = f"blocks/{i}"

        np.testing.assert_equal(data[f"{prefix}/values"], block.values)
        assert _npz_labels(data[f"{prefix}/samples"]) == block.samples
        assert _npz_labels(data[f"{prefix}/components/0"]) == block.components[0]
        assert _npz_labels(data[f"{prefix}/properties"]) == block.properties

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            prefix = f"blocks/{i}/gradients/{parameter}"

            np.testing.assert_equal(data[f"{prefix}/values"], gradient.values)
            assert _npz_labels(data[f"{prefix}/samples"]) == gradient.samples
            assert _npz_labels(data[f"{prefix}/components/0"]) == gradient.components[0]


@pytest.mark.parametrize("use_numpy_save", (True, False))
@pytest.mark.parametrize("use_numpy_load", (True, False))
def test_save_load_zero_length_block(
    use_numpy_save, use_numpy_load, tmpdir, tensor_zero_len_block
):
    """
    Tests that attempting to save and load a TensorMap with a zero-length axis block
    does not raise an error, when using combinations of use_numpy for save and
    load
    """
    file = "serialize-test-zero-len-block.npz"

    with tmpdir.as_cwd():
        metatensor.save(file, tensor_zero_len_block, use_numpy=use_numpy_save)
        metatensor.load(file, use_numpy=use_numpy_load)


def test_save_warning_errors(tmpdir, tensor):
    # does not have .npz ending and causes warning
    tmpfile = "serialize-test"

    with pytest.warns() as record:
        with tmpdir.as_cwd():
            metatensor.save(tmpfile, tensor)

    expected = f"adding '.npz' extension, the file will be saved at '{tmpfile}.npz'"
    assert str(record[0].message) == expected

    tmpfile = "serialize-test.npz"

    message = (
        "`data` must be either 'Labels' or 'TensorMap', "
        "not <class 'metatensor.block.TensorBlock'>"
    )
    with pytest.raises(TypeError, match=message):
        with tmpdir.as_cwd():
            metatensor.save(tmpfile, tensor.block(0))


@pytest.mark.parametrize("protocol", PICKLE_PROTOCOLS)
def test_pickle(protocol, tmpdir, tensor):
    """
    Checks that pickling and unpickling a tensor map results in the same tensor map
    """

    tmpfile = "serialize-test.pickle"

    with tmpdir.as_cwd():
        with open(tmpfile, "wb") as f:
            pickle.dump(tensor, f, protocol=protocol)

        with open(tmpfile, "rb") as f:
            tensor_loaded = pickle.load(f)

    np.testing.assert_equal(tensor.keys, tensor_loaded.keys)
    for key, block in tensor_loaded.items():
        ref_block = tensor.block(key)

        assert isinstance(block.values, np.ndarray)
        np.testing.assert_equal(block.values, ref_block.values)

        assert block.samples == ref_block.samples
        assert block.components == ref_block.components
        assert block.properties == ref_block.properties

        for parameter in block.gradients_list():
            gradient = block.gradient(parameter)
            ref_gradient = ref_block.gradient(parameter)

            np.testing.assert_equal(gradient.values, ref_gradient.values)

            assert gradient.samples == ref_gradient.samples
            assert gradient.components == ref_gradient.components


@pytest.mark.parametrize("use_numpy", (True, False))
def test_nested_gradients(tmpdir, use_numpy):
    block = TensorBlock(
        values=np.random.rand(3, 3),
        samples=Labels.range("s", 3),
        components=[],
        properties=Labels.range("p", 3),
    )

    grad = TensorBlock(
        values=np.random.rand(3, 3),
        samples=Labels.range("sample", 3),
        components=[],
        properties=Labels.range("p", 3),
    )

    grad_grad = TensorBlock(
        values=np.random.rand(3, 5, 3),
        samples=Labels.range("sample", 3),
        components=[Labels.range("c", 5)],
        properties=Labels.range("p", 3),
    )

    grad.add_gradient("grad-of-grad", grad_grad)
    block.add_gradient("grad", grad)
    tensor = TensorMap(Labels.single(), [block])

    tmpfile = "grad-grad-test.npz"

    with tmpdir.as_cwd():
        metatensor.save(tmpfile, tensor, use_numpy=use_numpy)

        # load back with numpy
        data = np.load(tmpfile)

        # load back with metatensor
        loaded = metatensor.load(tmpfile)

    assert _npz_labels(data["keys"]) == tensor.keys
    assert _npz_labels(data["keys"]) == loaded.keys

    for i, (key, block) in enumerate(tensor.items()):
        loaded_block = loaded.block(key)

        prefix = f"blocks/{i}"
        np.testing.assert_equal(data[f"{prefix}/values"], block.values)
        np.testing.assert_equal(block.values, loaded_block.values)

        assert _npz_labels(data[f"{prefix}/samples"]) == block.samples
        assert block.samples == loaded_block.samples

        assert _npz_labels(data[f"{prefix}/properties"]) == block.properties
        assert block.properties == loaded_block.properties

        assert block.gradients_list() == loaded_block.gradients_list()

        for parameter, gradient in block.gradients():
            loaded_gradient = loaded_block.gradient(parameter)
            grad_prefix = f"{prefix}/gradients/{parameter}"

            np.testing.assert_equal(data[f"{grad_prefix}/values"], gradient.values)
            np.testing.assert_equal(gradient.values, loaded_gradient.values)

            assert _npz_labels(data[f"{grad_prefix}/samples"]) == gradient.samples
            assert gradient.samples == loaded_gradient.samples

            assert gradient.components == loaded_gradient.components
            assert gradient.properties == loaded_gradient.properties

            assert gradient.gradients_list() == loaded_gradient.gradients_list()

            for parameter, grad_grad in gradient.gradients():
                loaded_grad_grad = loaded_gradient.gradient(parameter)
                grad_grad_prefix = f"{grad_prefix}/gradients/{parameter}"

                np.testing.assert_equal(
                    data[f"{grad_grad_prefix}/values"], grad_grad.values
                )
                np.testing.assert_equal(grad_grad.values, loaded_grad_grad.values)

                assert (
                    _npz_labels(data[f"{grad_grad_prefix}/samples"])
                    == grad_grad.samples
                )

                assert grad_grad.samples == loaded_grad_grad.samples

                assert len(grad_grad.components) == len(loaded_grad_grad.components)
                for a, b in zip(grad_grad.components, loaded_grad_grad.components):
                    assert a == b

                assert grad_grad.properties == loaded_grad_grad.properties


def _npz_labels(data):
    names = data.dtype.names
    return Labels(names=names, values=data.view(dtype=np.int32).reshape(-1, len(names)))


@pytest.mark.parametrize("memory_buffer", (True, False))
@pytest.mark.parametrize("standalone_fn", (True, False))
def test_load_labels(memory_buffer, standalone_fn):
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "metatensor-core",
        "tests",
        "keys.npy",
    )

    if memory_buffer:
        with open(path, "rb") as fd:
            buffer = fd.read()

        assert isinstance(buffer, bytes)
    else:
        file = path

    if standalone_fn:
        if memory_buffer:
            labels = metatensor.io.load_labels_buffer(buffer)
        else:
            labels = metatensor.load_labels(file)
    else:
        if memory_buffer:
            labels = Labels.load_buffer(buffer)
        else:
            labels = Labels.load(file)

    assert isinstance(labels, Labels)
    assert labels.names == [
        "o3_lambda",
        "o3_sigma",
        "center_type",
        "neighbor_type",
    ]
    assert len(labels) == 27


@pytest.mark.parametrize("memory_buffer", (True, False))
@pytest.mark.parametrize("standalone_fn", (True, False))
def test_save_labels(memory_buffer, standalone_fn, tmpdir, labels):
    """Check that as saved file loads fine with numpy."""
    with tmpdir.as_cwd():
        if memory_buffer:
            if standalone_fn:
                buffer = metatensor.io.save_buffer(labels)
            else:
                buffer = labels.save_buffer()

            file = io.BytesIO(buffer)
        else:
            file = "serialize-test.npy"
            if standalone_fn:
                metatensor.save(file, labels)
            else:
                labels.save(file)

        data = np.load(file)

    assert _npz_labels(data) == labels


@pytest.mark.parametrize("protocol", PICKLE_PROTOCOLS)
def test_pickle_labels(protocol, tmpdir, labels):
    """
    Checks that pickling and unpickling Labels results in the same Labels
    """

    tmpfile = "serialize-test.pickle"

    with tmpdir.as_cwd():
        with open(tmpfile, "wb") as f:
            pickle.dump(labels, f, protocol=protocol)

        with open(tmpfile, "rb") as f:
            labels_loaded = pickle.load(f)

    np.testing.assert_equal(labels, labels_loaded)


def test_wrong_load_error():
    data_root = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "metatensor-core", "tests"
    )

    message = (
        "serialization format error: unable to load a TensorMap from '.*', "
        "use `load_labels` to load Labels"
    )
    with pytest.raises(MetatensorError, match=message):
        metatensor.load(os.path.join(data_root, "keys.npy"))

    message = (
        "serialization format error: unable to load a TensorMap from buffer, "
        "use `load_labels_buffer` to load Labels"
    )
    with pytest.raises(MetatensorError, match=message):
        with open(os.path.join(data_root, "keys.npy"), "rb") as fd:
            buffer = fd.read()

        metatensor.load(io.BytesIO(buffer))

    message = (
        "serialization format error: unable to load Labels from '.*', "
        "use `load` to load TensorMap: start does not match magic string"
    )
    with pytest.raises(MetatensorError, match=message):
        metatensor.load_labels(os.path.join(data_root, "data.npz"))

    message = (
        "serialization format error: unable to load Labels from buffer, "
        "use `load_buffer` to load TensorMap: start does not match magic string"
    )
    with pytest.raises(MetatensorError, match=message):
        with open(os.path.join(data_root, "data.npz"), "rb") as fd:
            buffer = fd.read()

        metatensor.load_labels(io.BytesIO(buffer))
