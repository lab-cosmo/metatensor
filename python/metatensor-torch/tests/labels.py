from typing import Any, List, Optional, Tuple, Union

import pytest
import torch
from packaging import version
from torch import Tensor

from metatensor.torch import Labels, LabelsEntry


def test_constructor():
    # keyword arguments + name is a tuple
    labels = Labels(names=("a", "b"), values=torch.tensor([[0, 0], [0, 1]]))

    assert len(labels) == 2

    assert labels.names == ["a", "b"]
    assert torch.all(labels.values == torch.Tensor([[0, 0], [0, 1]]))

    # positional arguments + names is a list
    labels = Labels(["c"], torch.LongTensor([[0], [2], [-1]]))
    assert labels.names == ["c"]
    assert torch.all(labels.values == torch.Tensor([[0], [2], [-1]]))

    # single string for names
    labels = Labels(names="test", values=torch.tensor([[0], [1]]))

    assert labels.names == ["test"]
    assert torch.all(labels.values == torch.Tensor([[0], [1]]))

    labels = Labels.single()
    assert labels.names == ["_"]
    assert torch.all(labels.values == torch.Tensor([[0]]))

    labels = Labels.empty(["a", "bb", "ccc"])
    assert labels.names == ["a", "bb", "ccc"]
    assert labels.values.shape == (0, 3)

    labels = Labels.range("test", 33)
    assert labels.names == ["test"]
    assert torch.all(labels.values == torch.arange(33).reshape(-1, 1))


def test_constructor_errors():
    message = (
        "invalid Labels: the names must have an entry for each column of the array"
    )
    with pytest.raises(ValueError, match=message):
        _ = Labels(names="ab", values=torch.tensor([[0, 0], [0, 1]]))

    message = "names must be a tuple of strings, got element with type 'int' instead"
    with pytest.raises(TypeError, match=message):
        _ = Labels(names=(3, 4), values=torch.tensor([[0, 0], [0, 1]]))

    message = "names must be a list of strings, got element with type 'int' instead"
    with pytest.raises(TypeError, match=message):
        _ = Labels(names=[3, 4], values=torch.tensor([[0, 0], [0, 1]]))

    message = (
        "Expected a value of type 'Tensor' for argument 'values' but "
        + "instead found type 'tuple"
    )
    with pytest.raises(RuntimeError, match=message):
        _ = Labels(names="test", values=(3, 4))

    message = "Labels values must be an Tensor of 32-bit integers"
    with pytest.raises(ValueError, match=message):
        _ = Labels(names="test", values=torch.Tensor([[0, 0], [0, 1]]))

    message = "Labels values must be a 2D Tensor"
    with pytest.raises(ValueError, match=message):
        _ = Labels(names="test", values=torch.tensor([0, 1]))

    # check that normal validation from metatensor_core is passed through with
    # exceptions
    message = "invalid parameter: 'not an ident' is not a valid label name"
    with pytest.raises(RuntimeError, match=message):
        _ = Labels(names="not an ident", values=torch.tensor([[0]]))


def test_view():
    labels = Labels(names=("aaa", "bbb"), values=torch.tensor([[1, 2], [3, 4]]))

    assert not labels.is_view()

    view = labels.view("aaa")
    assert view.is_view()
    assert view.names == ["aaa"]
    assert torch.all(view.values == torch.tensor([[1], [3]]))

    view = labels.view("bbb")
    assert view.names == ["bbb"]
    assert torch.all(view.values == torch.tensor([[2], [4]]))

    view = labels.view(["bbb"])
    assert view.is_view()
    assert view.names == ["bbb"]
    assert torch.all(view.values == torch.tensor([[2], [4]]))

    view = labels.view(["bbb", "aaa"])
    assert view.is_view()
    assert view.names == ["bbb", "aaa"]
    assert torch.all(view.values == torch.tensor([[2, 1], [4, 3]]))

    view = labels.view(["aaa", "aaa", "aaa"])
    assert view.names == ["aaa", "aaa", "aaa"]
    assert torch.all(view.values == torch.tensor([[1, 1, 1], [3, 3, 3]]))

    message = "'ccc' not found in the dimensions of these Labels"
    with pytest.raises(ValueError, match=message):
        labels.view("ccc")

    message = "names must be a tuple of strings, got element with type 'int' instead"
    with pytest.raises(TypeError, match=message):
        labels.view((1, 2))

    view = labels.view("aaa")
    message = "can not call this function on Labels view, call to_owned first"
    with pytest.raises(ValueError, match=message):
        view.position([1])

    owned = view.to_owned()
    assert not owned.is_view()
    assert owned.position([1]) == 0
    assert owned.position([-1]) is None

    view = labels.view(["aaa", "aaa"])
    message = "invalid parameter: labels names must be unique, got 'aaa' multiple times"
    with pytest.raises(RuntimeError, match=message):
        view.to_owned()


def test_repr():
    labels = Labels(names=("aaa", "bbb"), values=torch.tensor([[1, 2], [3, 4]]))

    expected = "Labels(\n    aaa  bbb\n     1    2\n     3    4\n)"
    assert str(labels) == expected

    if version.parse(torch.__version__) >= version.parse("2.1"):
        # custom __repr__ definitions are only available since torch 2.1
        assert repr(labels) == expected

    expected = "LabelsEntry(aaa=1, bbb=2)"
    assert str(labels[0]) == expected
    # assert repr(labels[0]) == expected
    assert labels[0].print() == "(aaa=1, bbb=2)"

    labels = Labels(
        names=("aaa", "bbb"),
        values=torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]),
    )

    expected = """ aaa  bbb
  0    0
  1    1
  2    2
  3    3
  4    4
  5    5
  6    6"""
    assert labels.print(max_entries=-1) == expected

    expected = """ aaa  bbb
  0    0
   ...
  6    6"""
    assert labels.print(max_entries=0) == expected
    assert labels.print(max_entries=1) == expected
    assert labels.print(max_entries=2) == expected

    expected = """ aaa  bbb
  0    0
  1    1
  2    2
   ...
  5    5
  6    6"""
    assert labels.print(max_entries=5) == expected

    expected = """ aaa  bbb
     0    0
     1    1
      ...
     6    6"""
    assert labels.print(max_entries=3, indent=3) == expected

    labels = Labels(names=("aaa", "bbb"), values=torch.tensor([[0, 0], [0, 1]]))
    expected = "LabelsView(\n    bbb\n     0\n     1\n)"
    assert str(labels.view("bbb")) == expected

    labels = Labels(
        names=("aaa", "bbb"), values=torch.tensor([[111111111, 2], [3, 444444444]])
    )

    expected = """Labels(
       aaa        bbb
    111111111      2
        3      444444444
)"""
    assert str(labels) == expected


def test_indexing():
    labels = Labels(names=("a", "b"), values=torch.tensor([[1, 2], [3, 4]]))

    # indexing labels with integer
    entry = labels[0]
    assert entry.names == ["a", "b"]
    assert torch.all(entry.values == torch.tensor([1, 2]))

    entry = labels[-1]
    assert entry.names == ["a", "b"]
    assert torch.all(entry.values == torch.tensor([3, 4]))

    # indexing labels with string
    column = labels["a"]
    assert torch.all(column == torch.tensor([1, 3]))

    column = labels["b"]
    assert torch.all(column == torch.tensor([2, 4]))

    # indexing labels errors
    message = "out of range for tensor of size \\[2, 2\\] at dimension 0"
    with pytest.raises(IndexError, match=message):
        labels[3]

    with pytest.raises(IndexError, match=message):
        labels[-7]

    message = "Labels can only be indexed by int or str, got 'float' instead"
    with pytest.raises(TypeError, match=message):
        labels[3.4]

    message = "'cc' not found in the dimensions of these Labels"
    with pytest.raises(ValueError, match=message):
        labels["cc"]

    # indexing entry with integer
    entry = labels[0]
    assert entry[0] == 1
    assert entry[1] == 2
    assert tuple(entry) == (1, 2)

    # indexing entry with string
    assert entry["a"] == 1
    assert entry["b"] == 2

    # indexing entry errors
    message = (
        "LabelsEntry can only be indexed by int or str, got '\\(int, int\\)' instead"
    )
    with pytest.raises(TypeError, match=message):
        entry[1, 2]

    message = "LabelsEntry can only be indexed by int or str, got 'str\\[\\]' instead"
    with pytest.raises(TypeError, match=message):
        entry[["a", "b"]]


def test_iter():
    # we can iterate over Labels/LabelsEntry since they define both __len__ and
    # __getitem__
    labels = Labels(names=("a", "b"), values=torch.tensor([[0, 0], [0, 1]]))

    for i, values in enumerate(labels[1]):
        assert values == i

    for i, entry in enumerate(labels):
        assert tuple(entry) == (0, i)

    # expand labels entry to tuples during iteration
    for i, (a, b) in enumerate(labels):
        assert a == 0
        assert b == i


def test_eq():
    labels_1 = Labels(names=("a", "b"), values=torch.tensor([[0, 0], [0, 1]]))
    labels_2 = Labels(names=("a", "b"), values=torch.tensor([[0, 0], [0, 1]]))
    labels_3 = Labels(names=("a", "b"), values=torch.tensor([[0, 2], [0, 1]]))
    labels_4 = Labels(names=("a", "c"), values=torch.tensor([[0, 0], [0, 1]]))

    assert labels_1 == labels_2
    assert labels_1 != labels_3
    assert labels_1 != labels_4

    assert labels_1[0] == labels_1[0]
    assert labels_1[0] == labels_2[0]
    assert labels_1[0] != labels_3[0]
    assert labels_1[0] != labels_4[0]
    assert labels_1[1] == labels_3[1]


def test_to():
    devices = ["meta", torch.device("meta")]
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        devices.append("mps")
        devices.append(torch.device("mps"))

    if torch.cuda.is_available():
        devices.append("cuda")
        devices.append("cuda:0")
        devices.append(torch.device("cuda"))

    for device in devices:
        labels = Labels(names=("a", "b"), values=torch.tensor([[0, 0], [0, 1]]))
        assert labels.values.device.type == "cpu"
        assert labels.device.type == "cpu"

        labels = labels.to(device)
        assert labels.values.device.type == torch.device(device).type
        assert labels.device.type == torch.device(device).type


def test_position():
    labels = Labels(names=("a", "b"), values=torch.tensor([[0, 0], [0, 1]]))

    assert labels.position(torch.tensor([0, 0])) == 0
    assert labels.position(torch.tensor([1, 0])) is None
    assert torch.tensor([0, 0]) in labels
    assert torch.tensor([1, 0]) not in labels

    # list
    assert labels.position([0, 1]) == 1
    assert labels.position([1, 0]) is None
    assert [0, 1] in labels
    assert [1, 0] not in labels

    # tuple
    assert labels.position((0, 1)) == 1
    assert labels.position((1, 0)) is None
    assert (0, 1) in labels
    assert (1, 0) not in labels

    # LabelsEntry
    assert labels.position(labels[0]) == 0
    assert labels.position(labels[1]) == 1

    other_labels = Labels(names=("a", "b"), values=torch.tensor([[0, 1], [2, 3]]))
    assert labels.position(other_labels[0]) == 1
    assert labels.position(other_labels[1]) is None

    # make sure nothing breaks when using the same column multiple time in a view
    view = other_labels.view(["a", "a"])
    assert labels.position(view[0]) == 0

    message = (
        "parameter to Labels::positions must be a LabelsEntry, tensor, "
        "or list/tuple of integers"
    )
    with pytest.raises(TypeError, match=message):
        _ = labels.position(3)


def test_union():
    first = Labels(["aa", "bb"], torch.tensor([[0, 1], [1, 2]]))
    second = Labels(["aa", "bb"], torch.tensor([[2, 3], [1, 2], [4, 5]]))

    union = first.union(second)
    assert union.names == ["aa", "bb"]
    assert torch.all(union.values == torch.tensor([[0, 1], [1, 2], [2, 3], [4, 5]]))

    union_2, first_mapping, second_mapping = first.union_and_mapping(second)

    assert union == union_2
    assert torch.all(first_mapping == torch.LongTensor([0, 1]))
    assert torch.all(second_mapping == torch.LongTensor([2, 1, 3]))

    # check that union preserves devices
    first = first.to("meta")

    message = "device mismatch in union: got 'meta' and 'cpu'"
    with pytest.raises(ValueError, match=message):
        first.union(second)

    with pytest.raises(ValueError, match=message):
        first.union_and_mapping(second)

    second = second.to("meta")
    union = first.union(second)
    assert union.values.device == torch.device("meta")

    union, first_mapping, second_mapping = first.union_and_mapping(second)
    assert union.values.device == torch.device("meta")
    assert first_mapping.device == torch.device("meta")
    assert second_mapping.device == torch.device("meta")


def test_intersection():
    first = Labels(["aa", "bb"], torch.tensor([[0, 1], [1, 2]]))
    second = Labels(["aa", "bb"], torch.tensor([[2, 3], [1, 2], [4, 5]]))

    intersection = first.intersection(second)
    assert intersection.names == ["aa", "bb"]
    assert torch.all(intersection.values == torch.tensor([[1, 2]]))

    intersection_2, first_mapping, second_mapping = first.intersection_and_mapping(
        second
    )

    assert intersection == intersection_2
    assert torch.all(first_mapping == torch.LongTensor([-1, 0]))
    assert torch.all(second_mapping == torch.LongTensor([-1, 0, -1]))

    # check that intersection preserves devices
    first = first.to("meta")

    message = "device mismatch in intersection: got 'meta' and 'cpu'"
    with pytest.raises(ValueError, match=message):
        first.intersection(second)

    with pytest.raises(ValueError, match=message):
        first.intersection_and_mapping(second)

    second = second.to("meta")
    intersection = first.intersection(second)
    assert intersection.values.device == torch.device("meta")

    intersection, first_mapping, second_mapping = first.intersection_and_mapping(second)
    assert intersection.values.device == torch.device("meta")
    assert first_mapping.device == torch.device("meta")
    assert second_mapping.device == torch.device("meta")


def test_dimensions_manipulation():
    label = Labels("foo", torch.tensor([[42]]))

    # Labels.insert
    new_label = label.insert(0, name="bar", values=torch.tensor([10]))
    assert new_label.names == ["bar", "foo"]
    assert torch.all(new_label.values == torch.tensor([[10, 42]]))

    with pytest.raises(ValueError, match="`values` must be a 1D tensor"):
        label.insert(0, name="bar", values=torch.tensor([[10]]))

    # Labels.append
    new_label = label.append(name="bar", values=torch.tensor([10]))
    assert new_label.names == ["foo", "bar"]
    assert torch.all(new_label.values == torch.tensor([[42, 10]]))

    # Labels.remove
    removed_label = new_label.remove(name="bar")
    assert removed_label == label

    with pytest.raises(
        ValueError, match="'baz' not found in the dimensions of these Labels"
    ):
        new_label.remove(name="baz")

    # Labels.rename
    new_label = label.rename("foo", "bar")
    assert new_label.names == ["bar"]

    with pytest.raises(
        ValueError, match="'baz' not found in the dimensions of these Labels"
    ):
        new_label.rename("baz", "foo")

    # Labels.permute
    label = Labels(["foo", "bar", "baz"], torch.tensor([[42, 10, 3]]))

    new_label = label.permute([-1, 0, 1])
    assert new_label.names == ["baz", "foo", "bar"]
    torch.all(new_label.values == torch.tensor([[3, 42, 10]]))

    match = (
        r"the length of `dimensions_indexes` \(2\) does not match the number of "
        r"dimensions in the Labels \(3\)"
    )
    with pytest.raises(ValueError, match=match):
        label.permute([2, 0])

    match = r"out of range index 3 for labels dimensions \(3\)"
    with pytest.raises(IndexError, match=match):
        label.permute([0, 1, 3])


# define a wrapper class to make sure the types TorchScript uses for of all
# C-defined functions matches what we expect
class LabelsWrap:
    def __init__(self, names: List[str], values: Tensor):
        self._c = Labels(names, values)

    def __len___(self) -> int:
        return self._c.__len__()

    def __str__(self) -> str:
        return self._c.__str__()

    def __repr__(self) -> str:
        return self._c.__repr__()

    def __eq__(self, other: Labels) -> bool:
        return self._c.__eq__(other=other)

    def __ne__(self, other: Labels) -> bool:
        return self._c.__ne__(other=other)

    # we can not set the return type to Union[Labels, LabelsEntry] from C++
    # so it is set to Any
    def __getitem__(self, index: Union[int, str]) -> Any:
        return self._c.__getitem__(index=index)

    def contains(self, entry: Union[List[int], LabelsEntry]) -> bool:
        return self._c.__contains__(entry=entry)

    def names(self) -> List[str]:
        return self._c.names

    def values(self) -> Tensor:
        return self._c.values

    def device(self) -> torch.device:
        return self._c.device

    def to(self, device: torch.device) -> Labels:
        return self._c.to(device)

    def position(self, entry: Union[List[int], LabelsEntry]) -> Optional[int]:
        return self._c.position(entry=entry)

    def print_(self, max_entries: int, indent: int) -> str:
        return self._c.print(max_entries=max_entries, indent=indent)

    def to_owned(self) -> Labels:
        return self._c.to_owned()

    def entry(self, index: int) -> LabelsEntry:
        return self._c.entry(index=index)

    def column(self, dimension: str) -> torch.Tensor:
        return self._c.column(dimension=dimension)

    def view(self, names: Union[str, List[str]]) -> Labels:
        return self._c.view(names=names)

    def single(self) -> Labels:
        return Labels.single()

    def empty(self, names: Union[str, List[str]]) -> Labels:
        return Labels.empty(names)

    def range_(self, name: str, end: int) -> Labels:
        return Labels.range(name, end)

    def union(self, other: Labels) -> Labels:
        return self._c.union(other=other)

    def union_and_mapping(
        self, other: Labels
    ) -> Tuple[Labels, torch.Tensor, torch.Tensor]:
        return self._c.union_and_mapping(other=other)

    def intersection(self, other: Labels) -> Labels:
        return self._c.intersection(other=other)

    def intersection_and_mapping(
        self, other: Labels
    ) -> Tuple[Labels, torch.Tensor, torch.Tensor]:
        return self._c.intersection_and_mapping(other=other)

    def append(self, name: str, values: torch.Tensor) -> Labels:
        return self._c.append(name=name, values=values)

    def insert(self, index: int, name: str, values: torch.Tensor) -> Labels:
        return self._c.insert(index=index, name=name, values=values)

    def permute(self, dimensions_indexes: List[int]) -> Labels:
        return self._c.permute(dimensions_indexes=dimensions_indexes)

    def remove(self, name: str) -> Labels:
        return self._c.remove(name=name)

    def rename(self, old: str, new: str) -> Labels:
        return self._c.rename(old=old, new=new)


class LabelsEntryWrap:
    def __init__(self):
        labels = Labels(["a"], torch.tensor([]))
        entry = labels[0]
        assert isinstance(entry, LabelsEntry)
        self._c = entry

    def __str__(self) -> str:
        return self._c.__str__()

    def __repr__(self) -> str:
        return self._c.__repr__()

    def __len__(self) -> int:
        return self._c.__len__()

    def __getitem__(self, index: Union[int, str]) -> int:
        return self._c.__getitem__(index=index)

    def __eq__(self, other: LabelsEntry) -> bool:
        return self._c.__eq__(other=other)

    def __ne__(self, other: LabelsEntry) -> bool:
        return self._c.__ne__(other=other)

    def names(self) -> List[str]:
        return self._c.names

    def values(self) -> Tensor:
        return self._c.values

    def device(self) -> torch.device:
        return self._c.device

    def _print(self) -> str:
        return self._c.print()


def test_script():
    class TestModule(torch.nn.Module):
        def forward(self, x: LabelsWrap) -> LabelsWrap:
            return x

    module = TestModule()
    module = torch.jit.script(module)

    class TestModuleEntry(torch.nn.Module):
        def forward(self, x: LabelsEntryWrap) -> LabelsEntryWrap:
            return x

    module = TestModuleEntry()
    module = torch.jit.script(module)
