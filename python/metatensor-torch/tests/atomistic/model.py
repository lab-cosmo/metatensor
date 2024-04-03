import zipfile
from typing import Dict, List, Optional

import pytest
import torch

from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import (
    MetatensorAtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborsListOptions,
    System,
    check_atomistic_model,
)


class MinimalModel(torch.nn.Module):
    """The simplest possible metatensor atomistic model"""

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        if "dummy" in outputs:
            block = TensorBlock(
                values=torch.tensor([[0.0]]),
                samples=Labels("s", torch.tensor([[0]])),
                components=torch.jit.annotate(List[Labels], []),
                properties=Labels("p", torch.tensor([[0]])),
            )
            tensor = TensorMap(Labels("_", torch.tensor([[0]])), [block])
            return {
                "dummy": tensor,
            }
        else:
            return {}

    def requested_neighbors_lists(self) -> List[NeighborsListOptions]:
        return [
            NeighborsListOptions(cutoff=1.2, full_list=False),
            NeighborsListOptions(cutoff=4.3, full_list=True),
            NeighborsListOptions(cutoff=1.2, full_list=False),
        ]


@pytest.fixture
def model():
    model = MinimalModel()
    model.train(False)

    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 2, 3],
        interaction_range=4.3,
        outputs={
            "dummy": ModelOutput(
                quantity="",
                unit="",
                per_atom=False,
                explicit_gradients=[],
            ),
        },
        supported_devices=["cpu"],
        dtype="float64",
    )

    metadata = ModelMetadata()
    return MetatensorAtomisticModel(model, metadata, capabilities)


def test_export(model, tmpdir):
    with tmpdir.as_cwd():
        model.export("export.pt")

        with zipfile.ZipFile("export.pt") as file:
            assert "export/extra/metatensor-version" in file.namelist()
            assert "export/extra/torch-version" in file.namelist()

        check_atomistic_model("export.pt")


class ExampleModule(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self._name = name

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        return {}

    def requested_neighbors_lists(self) -> List[NeighborsListOptions]:
        return [NeighborsListOptions(1.0, False, self._name)]


class OtherModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        return {}

    def requested_neighbors_lists(self) -> List[NeighborsListOptions]:
        return [NeighborsListOptions(2.0, True, "other module")]


class FullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = ExampleModule("first module")
        self.second = ExampleModule("second module")
        self.other = OtherModule()

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        result = self.first(systems, outputs, selected_atoms)
        result.update(self.second(systems, outputs, selected_atoms))
        result.update(self.other(systems, outputs, selected_atoms))

        return result


def test_requested_neighbors_lists():
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        interaction_range=0.0,
        supported_devices=["cpu"],
        dtype="float64",
    )
    atomistic = MetatensorAtomisticModel(model, ModelMetadata(), capabilities)
    requests = atomistic.requested_neighbors_lists()

    assert len(requests) == 2

    assert requests[0].cutoff == 1.0
    assert not requests[0].full_list
    assert requests[0].requestors() == [
        "first module",
        "FullModel.first",
        "second module",
        "FullModel.second",
    ]

    assert requests[1].cutoff == 2.0
    assert requests[1].full_list
    assert requests[1].requestors() == [
        "other module",
        "FullModel.other",
    ]


def test_bad_capabilities():
    model = FullModel()
    model.train(False)

    capabilities = ModelCapabilities(
        supported_devices=["cpu"],
        dtype="float64",
    )
    message = (
        "`capabilities.interaction_range` was not set, "
        "but it is required to run simulations"
    )
    with pytest.raises(ValueError, match=message):
        MetatensorAtomisticModel(model, ModelMetadata(), capabilities)

    capabilities = ModelCapabilities(
        interaction_range=12,
        dtype="float64",
    )
    message = (
        "`capabilities.supported_devices` was not set, "
        "but it is required to run simulations"
    )
    with pytest.raises(ValueError, match=message):
        MetatensorAtomisticModel(model, ModelMetadata(), capabilities)

    capabilities = ModelCapabilities(
        interaction_range=float("nan"),
        supported_devices=["cpu"],
        dtype="float64",
    )
    message = (
        "`capabilities.interaction_range` should be a float between 0 and infinity"
    )
    with pytest.raises(ValueError, match=message):
        MetatensorAtomisticModel(model, ModelMetadata(), capabilities)

    capabilities = ModelCapabilities(
        interaction_range=12.0,
        supported_devices=["cpu"],
    )
    message = "`capabilities.dtype` was not set, but it is required to run simulations"
    with pytest.raises(ValueError, match=message):
        MetatensorAtomisticModel(model, ModelMetadata(), capabilities)
