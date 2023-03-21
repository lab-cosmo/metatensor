# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
from equistore import TensorMap, TensorBlock, Labels
import equistore
import numpy as np

def tensor_map():
    """
    Create a dummy tensor map to be used in tests. This is the same one as the
    tensor map used in `tensor.rs` tests.
    """
    block_1 = TensorBlock(
        values=np.full((3, 1, 1), 1.0),
        samples=Labels(["samples"], np.array([[0], [2], [4]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_1.add_gradient(
        "parameter",
        samples=Labels(
            ["sample", "parameter"], np.array([[0, -2], [2, 3]], dtype=np.int32)
        ),
        data=np.full((2, 1, 1), 11.0),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_2 = TensorBlock(
        values=np.full((3, 1, 3), 2.0),
        samples=Labels(["samples"], np.array([[0], [1], [3]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[3], [4], [5]], dtype=np.int32)),
    )
    block_2.add_gradient(
        "parameter",
        data=np.full((3, 1, 3), 12.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, -2], [0, 3], [2, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0]], dtype=np.int32))],
    )

    block_3 = TensorBlock(
        values=np.full((4, 3, 1), 3.0),
        samples=Labels(["samples"], np.array([[0], [3], [6], [8]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_3.add_gradient(
        "parameter",
        data=np.full((1, 3, 1), 13.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[1, -2]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
    )

    block_4 = TensorBlock(
        values=np.full((4, 3, 1), 4.0),
        samples=Labels(["samples"], np.array([[0], [1], [2], [5]], dtype=np.int32)),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
        properties=Labels(["properties"], np.array([[0]], dtype=np.int32)),
    )
    block_4.add_gradient(
        "parameter",
        data=np.full((2, 3, 1), 14.0),
        samples=Labels(
            ["sample", "parameter"],
            np.array([[0, 1], [3, 3]], dtype=np.int32),
        ),
        components=[Labels(["components"], np.array([[0], [1], [2]], dtype=np.int32))],
    )

    # TODO: different number of components?

    keys = Labels(
        names=["key_1", "key_2"],
        values=np.array([[0, 0], [1, 0], [2, 2], [2, 3]], dtype=np.int32),
    )

    return TensorMap(keys, [block_1, block_2, block_3, block_4])


class AddSuite:
    def setup(self):
        self.tm = tensor_map() # function from tests

    def mem_equistore_add(self):
        print("START", flush=True)
        return self.tm # crashes
        #return equistore.add(self.tm, self.tm)
