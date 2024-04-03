import os

import numpy as np

import metatensor
from metatensor import Labels, TensorBlock, TensorMap


DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


def test_std_samples_block():
    tensor_se = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-spherical-expansion.npz"),
        use_numpy=True,
    )
    tensor_ps = metatensor.load(
        os.path.join(DATA_ROOT, "qm7-power-spectrum.npz"),
        use_numpy=True,
    )
    tensor_se = metatensor.remove_gradients(tensor_se)

    bl1 = tensor_ps[0]

    # check both passing a list and a single string for sample_names
    reduce_tensor_se = metatensor.std_over_samples(tensor_se, sample_names="atom")
    reduce_tensor_ps = metatensor.std_over_samples(tensor_ps, sample_names=["atom"])

    assert np.allclose(
        np.std(bl1.values[:4], axis=0),
        reduce_tensor_ps.block(0).values[0],
        rtol=1e-13,
    )

    assert np.allclose(
        np.std(bl1.values[4:10], axis=0),
        reduce_tensor_ps.block(0).values[1],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[22:26], axis=0),
        reduce_tensor_ps.block(0).values[5],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[38:46], axis=0),
        reduce_tensor_ps.block(0).values[8],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(bl1.values[46:], axis=0),
        reduce_tensor_ps.block(0).values[9],
        rtol=1e-13,
    )

    # Test the gradients
    gr1 = tensor_ps[0].gradient("positions")

    XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[0, 4, 8, 12])
    assert np.allclose(
        (
            np.mean(XdX, axis=0)
            - np.mean(bl1.values[:4], axis=0)
            * np.mean(gr1.values[[0, 4, 8, 12]], axis=0)
        )
        / np.std(bl1.values[:4], axis=0),
        reduce_tensor_ps.block(0).gradient("positions").values[0],
        rtol=1e-13,
    )

    XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[2, 6, 10, 14])
    assert np.allclose(
        (
            np.mean(XdX, axis=0)
            - np.mean(bl1.values[:4], axis=0)
            * np.mean(gr1.values[[2, 6, 10, 14]], axis=0)
        )
        / np.std(bl1.values[:4], axis=0),
        reduce_tensor_ps.block(0).gradient("positions").values[2],
    )

    XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[3, 7, 11, 15])
    assert np.allclose(
        (
            np.mean(XdX, axis=0)
            - np.mean(bl1.values[:4], axis=0)
            * np.mean(gr1.values[[3, 7, 11, 15]], axis=0)
        )
        / np.std(bl1.values[:4], axis=0),
        reduce_tensor_ps.block(0).gradient("positions").values[3],
    )

    XdX = get_XdX(block=tensor_ps[0], gradient=gr1, der_index=[96, 99, 102])
    idx = [
        i
        for i in range(len(bl1.samples))
        if bl1.samples[i][0] == bl1.samples[gr1.samples[96][0]][0]
    ]

    assert np.allclose(
        (
            np.mean(XdX, axis=0)
            - np.mean(bl1.values[idx], axis=0)
            * np.mean(gr1.values[[96, 99, 102]], axis=0)
        )
        / np.std(bl1.values[idx], axis=0),
        reduce_tensor_ps.block(0).gradient("positions").values[40],
        rtol=1e-13,
    )

    # The TensorBlock with key=(8,8,8) has nothing to be averaged over
    values = reduce_tensor_ps.block(
        center_type=8, neighbor_1_type=8, neighbor_2_type=8
    ).values
    assert np.allclose(
        np.zeros(values.shape),
        values,
    )

    for ii, bl2 in enumerate([tensor_se[0], tensor_se[1], tensor_se[2], tensor_se[3]]):
        assert np.allclose(
            np.std(bl2.values[:4], axis=0),
            reduce_tensor_se.block(ii).values[0],
            rtol=1e-13,
        )
        assert np.allclose(
            np.std(bl2.values[26:32], axis=0),
            reduce_tensor_se.block(ii).values[6],
            rtol=1e-13,
        )
        assert np.allclose(
            np.std(bl2.values[32:38], axis=0),
            reduce_tensor_se.block(ii).values[7],
            rtol=1e-13,
        )
        assert np.allclose(
            np.std(bl2.values[46:], axis=0),
            reduce_tensor_se.block(ii).values[9],
            rtol=1e-13,
        )


def test_reduction_block_two_samples():
    block_1 = TensorBlock(
        values=np.array(
            [
                [1, 2, 4],
                [3, 5, 6],
                [-1.3, 26.7, 4.54],
                [3.5, 5.3, 6.87],
                [6.1, 35.2, 44.5],
                [7.3, -7.65, 6.45],
                [11, 276.0, 4.09],
                [33, 55.5, -5.6],
            ]
        ),
        samples=Labels(
            ["s_1", "s_2", "s_3"],
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 0, 2],
                    [0, 1, 1],
                    [0, 1, 0],
                    [2, 1, 1],
                    [1, 1, 1],
                    [1, 0, 0],
                ],
            ),
        ),
        components=[],
        properties=Labels(["p"], np.array([[0], [1], [5]])),
    )

    keys = Labels(names=["key_1", "key_2"], values=np.array([[0, 0]]))
    X = TensorMap(keys, [block_1])

    reduce_X_12 = metatensor.std_over_samples(X, sample_names=["s_3"])
    reduce_X_23 = metatensor.std_over_samples(X, sample_names="s_1")
    reduce_X_2 = metatensor.std_over_samples(X, sample_names=["s_1", "s_3"])

    assert np.allclose(
        np.std(X.block(0).values[:3], axis=0),
        reduce_X_12.block(0).values[0],
        rtol=1e-13,
    )
    assert np.allclose(
        np.std(X.block(0).values[3:5], axis=0),
        reduce_X_12.block(0).values[1],
        rtol=1e-13,
    )
    assert np.all(np.array([0.0]) == reduce_X_12.block(0).values[4])
    assert np.all(np.array([0.0]) == reduce_X_12.block(0).values[3])
    assert np.all(np.array([0.0]) == reduce_X_12.block(0).values[2])

    assert np.all(
        np.std(X.block(0).values[[0, 7]], axis=0) == reduce_X_23.block(0).values[0]
    )
    assert np.allclose(
        np.std(X.block(0).values[[3, 5, 6]], axis=0),
        reduce_X_23.block(0).values[4],
        rtol=1e-13,
    )

    assert np.all(np.array([0.0]) == reduce_X_23.block(0).values[1])
    assert np.all(np.array([0.0]) == reduce_X_23.block(0).values[2])
    assert np.all(np.array([0.0]) == reduce_X_23.block(0).values[3])

    assert np.allclose(
        np.std(X.block(0).values[[0, 1, 2, 7]], axis=0),
        reduce_X_2.block(0).values[0],
        rtol=1e-13,
    )
    assert np.all(
        np.std(X.block(0).values[3:7], axis=0) == reduce_X_2.block(0).values[1]
    )

    # check metadata
    assert reduce_X_12.block(0).properties == X.block(0).properties
    assert reduce_X_23.block(0).properties == X.block(0).properties
    assert reduce_X_2.block(0).properties == X.block(0).properties

    samples_12 = Labels(
        names=["s_1", "s_2"],
        values=np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1]]),
    )
    samples_23 = Labels(
        names=["s_2", "s_3"],
        values=np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),
    )
    samples_2 = Labels(
        names=["s_2"],
        values=np.array([[0], [1]]),
    )
    assert reduce_X_12.block(0).samples == samples_12
    assert reduce_X_23.block(0).samples == samples_23
    assert reduce_X_2.block(0).samples == samples_2


def test_reduction_of_one_element():
    block_1 = TensorBlock(
        values=np.array([[1, 2, 4], [3, 5, 6], [-1.3, 26.7, 4.54]]),
        samples=Labels(["s_1", "s_2"], np.array([[0, 0], [1, 1], [2, 2]])),
        components=[],
        properties=Labels(["p"], np.array([[0], [1], [5]])),
    )

    block_1.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7.8]]),
            samples=Labels(["sample", "g"], np.array([[0, 0], [1, 1], [2, 2]])),
            components=[],
            properties=block_1.properties,
        ),
    )

    keys = Labels(names=["key_1"], values=np.array([[0]]))
    X = TensorMap(keys, [block_1])

    add_X = metatensor.sum_over_samples(X, sample_names=["s_1"])
    mean_X = metatensor.mean_over_samples(X, sample_names=["s_1"])
    var_X = metatensor.var_over_samples(X, sample_names=["s_1"])
    std_X = metatensor.std_over_samples(X, sample_names=["s_1"])

    # print(add_X[0])
    # print(X[0].values, add_X[0].values)
    assert np.all(X[0].values == add_X[0].values)
    assert np.all(X[0].values == mean_X[0].values)
    assert metatensor.equal(add_X, mean_X)
    assert metatensor.equal_metadata(add_X, var_X)
    assert metatensor.equal_metadata(mean_X, std_X)

    assert np.all(np.zeros((3, 3)) == std_X[0].values)
    assert metatensor.equal(var_X, std_X)

    # Gradients
    grad_sample_label = Labels(
        names=["sample", "g"],
        values=np.array([[0, 0], [1, 1], [2, 2]]),
    )

    assert std_X[0].gradient("g").samples == grad_sample_label
    assert np.all(X[0].gradient("g").values == add_X[0].gradient("g").values)
    assert np.all(X[0].gradient("g").values == mean_X[0].gradient("g").values)
    assert np.all(np.zeros((3, 3)) == std_X[0].gradient("g").values)
    assert np.all(np.zeros((3, 3)) == var_X[0].gradient("g").values)


def get_XdX(block, gradient, der_index):
    XdX = []
    for ig in der_index:
        idx = gradient.samples[ig][0]
        XdX.append(block.values[idx] * gradient.values[ig])
    return np.stack(XdX)
