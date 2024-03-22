# Changelog

All notable changes to metatensor-torch are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/lab-cosmo/metatensor/)

<!-- Possible sections for each package:

#### Added

#### Fixed

#### Changed

#### Removed
-->

### metatensor-torch C++

#### Added

- `ModelCapabilities::dtype`, used by the model to communicate the dtype it
  wants to use for inputs and outputs.

### metatensor-torch Python

#### Added

- `ModelCapabilities.dtype`, used by the model to communicate the dtype it
  wants to use for inputs and outputs.

- The `device` that should be used by a model inside the ASE's
  `MetatensorCalculator` can now be specified by the user.


## [Version 0.3.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-torch-v0.3.0) - 2024-03-01

### metatensor-torch C++

#### Added

- `ModelMetadata` to record metadata about a specific model such as it's name,
  authors, etc.
- Added `interaction_range` and `supported_devices` to `ModelCapabilities`

#### Changed

- `System::species` has been renamed to `System::types`.

### metatensor-torch Python

#### Added

- `ModelMetadata` to record metadata about a specific model such as it's name,
  authors, etc.
- Added `interaction_range` and `supported_devices` to `ModelCapabilities`

#### Changed

- `System.species` has been renamed to `System.types`.

## [Version 0.2.1](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-torch-v0.2.1) - 2024-01-26

### metatensor-torch C++

#### Added

- Offer serialization functionality as member functions (i.e. `TensorMap::load`)
  in addition to the existing free standing functions (i.e. `metatensor_torch::load`) (#453)
- In-memory serialization with `TensorMap::save_buffer`, `TensorMap::load_buffer`,
  and the respective free standing functions (#455)
- Serialization of Labels, with the same API as `TensorMap` (#455)


### metatensor-torch Python

#### Added

- Offer serialization functionality as member functions (i.e. `TensorMap.load`)
  in addition to the existing free standing functions (i.e. `metatensor.torch.load`) (#453)
- In-memory serialization with `TensorMap.save_buffer`, `TensorMap.load_buffer`,
  and the respective free standing functions (#455)
- Serialization of Labels, with the same API as `TensorMap` (#455)

## [Version 0.2.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-torch-v0.2.0) - 2024-01-08

### metatensor-torch C++

#### Added

- New classes specifically tailored for atomistic models (#405):
  - `System` defines the input of a model;
  - `NeighborsListOptions` allow a model to request a specific neighbors list;
  - `ModelRunOptions`, `ModelOutput` and `ModelCapabilities` allow to statically
    describe capabilities of a model, and request specific outputs from it.

- `TensorBlock::to`, `TensorMap::to`, and `System::to` to change the device or
  dtype of torch Tensor stored by metatensor
- `Labels::device`, `TensorBlock::device` and `TensorMap::device`; as well as
  `TensorMap::scalar_type`, and `TensorBlock::scalar_type` to query the current
  device and scalar type/dtype used by the data.
- `metatensor_torch::version` function, returning the version of the code as a
  string.

#### Fixed

- We now check that all tensors in a `TensorBlock`/`TensorMap` have the same
  dtype and device (#414)
- `keys_to_properties`, `keys_to_samples` and `components_to_properties` now
  keep the different Labels on the same device (#411)

### metatensor-torch Python

#### Added

- New classes specifically tailored for atomistic models (#405):
  - same classes as the C++ interfaces, in `metatensor.torch.atomistic`
  - `MetatensorAtomisticModel` as a way to wrap user-defined `torch.nn.Module`
    and export them in a unified way, handling unit conversions and metadata
    checks.
- [ASE](https://wiki.fysik.dtu.dk/ase/) calculator based on
  `MetatensorAtomisticModel` in `metatensor.torch.atomistic.ase_calculator`.
  This allow using arbitrary user-defined models to run simulations with ASE.

- `TensorBlock.to`, `TensorMap.to` and `System.to` to change the device or dtype
  of torch Tensor stored by metatensor
- `Labels.device`, `TensorBlock.device` and `TensorMap.device`; as well as
  `TensorMap.dtype`, and `TensorBlock.dtype` to query the current device and
  dtype used by the data.


## [Version 0.1.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-torch-v0.1.0) - 2023-10-11

### metatensor-torch C++

#### Added

- TorchScript bindings to all metatensor-core class: `Labels`, `LabelsEntry`,
  `TensorBlock`, and `TensorMap`;
- Implementation of `mts_array_t`/`metatensor::DataArrayBase` for `torch::Tensor`;

### metatensor-torch Python

#### Added

- Expose TorchScript classes to Python;
- Expose all functions from `metatensor-operations` as TorchScript compatible code;
