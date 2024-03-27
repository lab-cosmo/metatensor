# Changelog

All notable changes to metatensor-core are documented here, following the [keep
a changelog](https://keepachangelog.com/en/1.1.0/) format. This project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/lab-cosmo/metatensor/)

<!--
### metatensor-core C++

#### Added

#### Fixed

#### Changed

#### Removed


### metatensor-core C

#### Added

#### Fixed

#### Changed

#### Removed


### metatensor-core Python

#### Added

#### Fixed

#### Changed

#### Removed
-->

### metatensor-core Julia

#### Added

- the Julia bindings to metatensor-core in the Metatensor.jl package


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
