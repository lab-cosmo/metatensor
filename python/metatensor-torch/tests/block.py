from typing import List, Optional, Tuple

import pytest
import torch
from packaging import version
from torch import Tensor

from metatensor.torch import Labels, TensorBlock


def test_constructor():
    # keyword arguments
    block = TensorBlock(
        values=torch.full((3, 2), 11),
        samples=Labels(names=["s"], values=torch.tensor([[0], [2], [1]])),
        components=[],
        properties=Labels(names=["p"], values=torch.tensor([[1], [0]])),
    )

    assert torch.all(block.values == torch.full((3, 2), 11))
    assert block.samples.names == ["s"]
    assert block.components == []
    assert block.properties.names == ["p"]

    # positional arguments
    block = TensorBlock(
        torch.full((3, 2), 33),
        Labels(names=["s"], values=torch.tensor([[0], [2], [1]])),
        [],
        Labels(names=["p"], values=torch.tensor([[1], [0]])),
    )

    assert torch.all(block.values == torch.full((3, 2), 33))
    assert block.samples.names == ["s"]
    assert block.components == []
    assert block.properties.names == ["p"]


def test_repr():
    block = TensorBlock(
        values=torch.full((3, 2), 11),
        samples=Labels(names=["s"], values=torch.tensor([[0], [2], [1]])),
        components=[],
        properties=Labels(names=["p"], values=torch.tensor([[1], [0]])),
    )

    expected = """TensorBlock
    samples (3): ['s']
    components (): []
    properties (2): ['p']
    gradients: None
"""

    assert str(block) == expected

    block = TensorBlock(
        values=torch.full((3, 3, 1, 2, 5), 11),
        samples=Labels(
            names=["s_1", "s_2"], values=torch.tensor([[0, 0], [1, 1], [2, 2]])
        ),
        components=[
            Labels(names=["c_1"], values=torch.tensor([[0], [1], [2]])),
            Labels(names=["c_2"], values=torch.tensor([[0]])),
            Labels(names=["c_3"], values=torch.tensor([[0], [1]])),
        ],
        properties=Labels(
            names=["p_1", "p_2"],
            values=torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
        ),
    )

    block.add_gradient(
        "g",
        TensorBlock(
            values=torch.full((3, 3, 1, 2, 5), 11),
            samples=Labels(
                names=["sample", "g"], values=torch.tensor([[0, 0], [1, 1], [2, 2]])
            ),
            components=[
                Labels(names=["c_1"], values=torch.tensor([[0], [1], [2]])),
                Labels(names=["c_2"], values=torch.tensor([[0]])),
                Labels(names=["c_3"], values=torch.tensor([[0], [1]])),
            ],
            properties=Labels(
                names=["p_1", "p_2"],
                values=torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]),
            ),
        ),
    )

    expected = """TensorBlock
    samples (3): ['s_1', 's_2']
    components (3, 1, 2): ['c_1', 'c_2', 'c_3']
    properties (5): ['p_1', 'p_2']
    gradients: ['g']
"""
    assert str(block) == expected

    if version.parse(torch.__version__) >= version.parse("2.1"):
        # custom __repr__ definitions are only available since torch 2.1
        assert repr(block) == expected

    expected = """Gradient TensorBlock ('g')
    samples (3): ['sample', 'g']
    components (3, 1, 2): ['c_1', 'c_2', 'c_3']
    properties (5): ['p_1', 'p_2']
    gradients: None
"""
    assert str(block.gradient("g")) == expected

    if version.parse(torch.__version__) >= version.parse("2.1"):
        # custom __repr__ definitions are only available since torch 2.1
        assert repr(block.gradient("g")) == expected


def test_copy():
    values = torch.full((3, 2), 11)
    block = TensorBlock(
        values=values,
        samples=Labels(names=["s"], values=torch.tensor([[0], [2], [1]])),
        components=[],
        properties=Labels(names=["p"], values=torch.tensor([[1], [0]])),
    )

    assert values.data_ptr() == block.values.data_ptr()

    clone = block.copy()
    del block

    assert values.data_ptr() != clone.values.data_ptr()


def test_gradients():
    block = TensorBlock(
        values=torch.full((3, 2), 1),
        samples=Labels(names=["s"], values=torch.tensor([[0], [2], [1]])),
        components=[],
        properties=Labels(names=["p"], values=torch.tensor([[1], [0]])),
    )

    assert block.gradients_list() == []

    block.add_gradient(
        parameter="g",
        gradient=TensorBlock(
            values=torch.full((1, 3, 2), 11),
            samples=Labels(
                names=("sample", "g"),
                values=torch.tensor([[0, 1]]),
            ),
            components=[Labels(names=("c"), values=torch.tensor([[0], [1], [2]]))],
            properties=block.properties,
        ),
    )

    assert block.gradients_list() == ["g"]
    assert block.has_gradient("g") is True
    assert block.has_gradient(parameter="not-there") is False

    gradient = block.gradient("g")
    assert gradient.values.shape == (1, 3, 2)

    gradient = block.gradient(parameter="g")
    assert gradient.samples.names == ["sample", "g"]

    message = "can not find gradients with respect to 'not-there' in this block"
    with pytest.raises(RuntimeError, match=message):
        gradient = block.gradient("not-there")

    gradients = block.gradients()
    assert isinstance(gradients, list)
    assert len(gradients) == 1
    assert gradients[0][0] == "g"


def test_different_device():
    message = (
        "cannot create TensorBlock: values and samples must be on the same device, "
        "got meta and cpu"
    )
    with pytest.raises(ValueError, match=message):
        TensorBlock(
            values=torch.tensor([[3.0, 4.0]], device="meta"),
            samples=Labels.range("s", 1),
            components=[],
            properties=Labels.range("p", 2),
        )

    block = TensorBlock(
        values=torch.tensor([[3.0, 4.0]]),
        samples=Labels.range("s", 1),
        components=[],
        properties=Labels.range("p", 2),
    )

    devices = []
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        devices.append("mps")

    if torch.cuda.is_available():
        devices.append("cuda")

    for device in devices:
        message = "values and the new gradient must be on the same device, got cpu and"
        with pytest.raises(ValueError, match=message):
            block.add_gradient(
                "g",
                TensorBlock(
                    values=torch.tensor([[3.0, 4.0]], device=device),
                    samples=Labels.range("sample", 1).to(device),
                    components=[],
                    properties=Labels.range("p", 2).to(device),
                ),
            )


def test_different_dtype():
    block = TensorBlock(
        values=torch.tensor([[3.0, 4.0]]),
        samples=Labels.range("s", 1),
        components=[],
        properties=Labels.range("p", 2),
    )

    message = (
        "values and the new gradient must have the same dtype, "
        "got torch.float16 and torch.float32"
    )
    with pytest.raises(TypeError, match=message):
        block.add_gradient(
            "g",
            TensorBlock(
                values=torch.tensor([[3.0, 4.0]], dtype=torch.float16),
                samples=Labels.range("sample", 1),
                components=[],
                properties=Labels.range("p", 2),
            ),
        )


def test_to():
    block = TensorBlock(
        values=torch.tensor([[3.0, 4.0]]),
        samples=Labels.range("s", 1),
        components=[],
        properties=Labels.range("p", 2),
    )

    block.add_gradient(
        "g",
        TensorBlock(
            values=torch.tensor([[3.0, 4.0]]),
            samples=Labels.range("sample", 1),
            components=[],
            properties=Labels.range("p", 2),
        ),
    )

    assert block.device.type == torch.device("cpu").type
    if version.parse(torch.__version__) >= version.parse("2.1"):
        check_dtype(block, torch.float32)
        check_dtype(block.gradient("g"), torch.float32)

    converted = block.to(dtype=torch.float64)
    if version.parse(torch.__version__) >= version.parse("2.1"):
        check_dtype(converted, torch.float64)
        check_dtype(converted.gradient("g"), torch.float64)

    devices = ["meta", torch.device("meta")]
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        devices.append("mps")
        devices.append(torch.device("mps"))

    if torch.cuda.is_available():
        devices.append("cuda")
        devices.append("cuda:0")
        devices.append(torch.device("cuda"))

    for device in devices:
        moved = block.to(device=device)
        assert moved.device.type == torch.device(device).type
        assert moved.gradient("g").device.type == torch.device(device).type

    # this should run without error
    moved = block.to(arrays=None)
    moved = block.to(arrays="torch")

    message = "`arrays` must be None or 'torch', got 'numpy' instead"
    with pytest.raises(ValueError, match=message):
        moved = block.to(arrays="numpy")

    # check that the code handles both positional and keyword arguments
    device = "meta"
    moved = block.to(device, dtype=torch.float32)
    moved = block.to(torch.float32, device)
    moved = block.to(torch.float32, device=device)
    moved = block.to(device, torch.float32)

    message = "can not give a device twice in `TensorBlock.to`"
    with pytest.raises(ValueError, match=message):
        moved = block.to("meta", device="meta")

    message = "can not give a dtype twice in `TensorBlock.to`"
    with pytest.raises(ValueError, match=message):
        moved = block.to(torch.float32, dtype=torch.float32)

    message = "unexpected type in `TensorBlock.to`: Tensor"
    with pytest.raises(TypeError, match=message):
        moved = block.to(torch.tensor([0]))


# This function only works in script mode, because `block.dtype` is always an `int`, and
# `torch.dtype` is only an int in script mode.
@torch.jit.script
def check_dtype(block: TensorBlock, dtype: torch.dtype):
    assert block.dtype == dtype


# define a wrapper class to make sure the types TorchScript uses for of all
# C-defined functions matches what we expect
class TensorBlockWrap:
    def __init__(
        self,
        values: Tensor,
        samples: Labels,
        components: List[Labels],
        properties: Labels,
    ):
        self._c = TensorBlock(
            values=values,
            samples=samples,
            components=components,
            properties=properties,
        )

    def __str__(self) -> str:
        return self._c.__str__()

    def __repr__(self) -> str:
        return self._c.__repr__()

    def copy(self) -> TensorBlock:
        return self._c.copy()

    def values(self) -> Tensor:
        return self._c.values

    def samples(self) -> Labels:
        return self._c.samples

    def components(self) -> List[Labels]:
        return self._c.components

    def properties(self) -> Labels:
        return self._c.properties

    def add_gradient(self, parameter: str, gradient: TensorBlock):
        return self._c.add_gradient(parameter=parameter, gradient=gradient)

    def gradients_list(self) -> List[str]:
        return self._c.gradients_list()

    def has_gradient(self, parameter: str) -> bool:
        return self._c.has_gradient(parameter=parameter)

    def gradient(self, parameter: str) -> TensorBlock:
        return self._c.gradient(parameter=parameter)

    def gradients(self) -> List[Tuple[str, TensorBlock]]:
        return self._c.gradients()

    def device(self) -> torch.device:
        return self._c.device

    def dtype(self) -> torch.dtype:
        return self._c.dtype

    def to(
        self,
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
        arrays: Optional[str],
    ) -> TensorBlock:
        return self._c.to(dtype=dtype, device=device, arrays=arrays)


def test_script():
    class TestModule(torch.nn.Module):
        def forward(self, x: TensorBlockWrap) -> TensorBlockWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)
