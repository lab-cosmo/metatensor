# Changelog

All notable changes to metatensor-core are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/lab-cosmo/metatensor/)

<!-- Possible sections for each package:

#### Added

#### Fixed

#### Changed

#### Removed
-->

### metatensor-core C++

### metatensor-core C

### metatensor-core Python

### metatensor-core Julia

#### Added

- the Julia bindings to metatensor-core in the Metatensor.jl package

## [Version 0.1.3](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-core-v0.1.2) - 2024-02-12

### Fixed

- Fixed the build with Cargo 1.65 (#505)
- Pin dependencies for metatensor-core, ensuring reproducibility of the build as
  new dependencies versions are published (#506)

## [Version 0.1.2](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-core-v0.1.2) - 2024-01-26

### metatensor-core C++

#### Added

- Offer serialization functionality as free standing functions (i.e. `metatensor::io::load`)
  in addition to the existing associated functions (i.e. `TensorMap::load`) (#453)
- Serialization of labels with `Labels::save`, `Labels::load`,
  `Labels::save_buffer`, `Labels::load_buffer` and the corresponding free
  functions. (#455)

#### Changed

- `TensorMap::save_string_buffer` has been replaced with the template version of
  `TensorMap::save_buffer`


### metatensor-core C

#### Added

- `mts_labels_save`, `mts_labels_load`, `mts_labels_save_buffer`, and
  `mts_labels_load_buffer` to handle serialization of `mts_labels_t`


### metatensor-core Python

#### Added

- `TensorMap.to` and `TensorBlock.to` to change the device, dtype, or backend
  (numpy or torch) of all arrays stored by metatensor
- `Labels.device`, `TensorBlock.device` and `TensorMap.device`; as well as
  `TensorMap.dtype`, and `TensorBlock.dtype` to query the current device and
  dtype used by the data.
- Offer serialization functionality as member functions (i.e. `TensorMap.load`)
  in addition to the existing free standing functions (i.e. `metatensor.load`) (#453)
- Serialization of labels with `Labels.save`, `Labels.load`,
  `Labels.save_buffer`, `Labels.load_buffer` and the corresponding free
  functions.

## [Version 0.1.1](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-core-v0.1.1) - 2024-01-05

### Fixed

- Fixed the build with Cargo 1.75 (#438)
- Allowed saving and loading empty TensorMap; i.e. TensorMap where one of the
  dimensions of the array has 0 elements (#419)

## [Version 0.1.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-core-v0.1.0) - 2023-10-11

### metatensor-core C

#### Added

- Initial implementation of all the core classes of metatensor: `mts_labels_t`,
  `mts_block_t`, `mts_tensormap_t`, `mts_array_t` and the corresponding
  functions;
- Serialization for `mts_tensormap_t` using a format derived from [numpy's
  npz](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html) format

### metatensor-core C++

#### Added

- C++ API for all the C data structures as C++ classes: `Labels`, `TensorBlock`,
  `TensorMap`;
- Wrapper around `mts_array_t` as an abstract base class `DataArrayBase`;
- Basic implementations of `DataArrayBase` in `SimpleDataArray` and
  `EmptyDataArray`;
- Basic n-dimensional array class `NDArray<T>`, intended to give a minimal API
  to use data stored in `TensorBlock` even if this data does not come from C++
  initially;

### metatensor-core Python

#### Added

- Python API for all the C data structures as Python classes: `Labels`,
  `LabelsEntry`, `TensorBlock`, `TensorMap`;
- Wrapper around `mts_array_t` as an abstract base class `metatensor.data.Array`;
- Implementation of `metatensor.data.Array` with `numpy.ndarray` and
  `torch.Tensor`;
