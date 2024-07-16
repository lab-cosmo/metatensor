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

#### Added

- `Labels::select` to sub-select entries in labels

### metatensor-core C

- `mts_labels_select` to sub-select entries in labels

### metatensor-core Python

- `Labels.select` to sub-select entries in labels

### metatensor-core Julia

#### Added

- the Julia bindings to metatensor-core in the Metatensor.jl package

## [Version 0.1.9](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-core-v0.1.9) - 2024-07-15

### metatensor-core Python

#### Added

- `TensorBlock.__len__` and `TensorBlock.shape`, which return the length and
  shape of the values in the block respectively
- We can now load (but not save) TensorMap stored in npz files using DEFLATE
  compression (#671)

#### Changed

- We now require Python >= 3.9

#### Fixed

- Fixed a memory leak affecting all data stored in TensorBlock (#683)

## [Version 0.1.8](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-core-v0.1.8) - 2024-05-13

### Fixed

- fix the build when using metatensor from the Rust bindings

## [Version 0.1.7](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-core-v0.1.7) - 2024-05-13

### metatensor-core C/C++

#### Added

- preprocessor macros containing the version number of metatensor:
  `METATENSOR_VERSION`, `METATENSOR_VERSION_MAJOR`, `METATENSOR_VERSION_MINOR`,
  and `METATENSOR_VERSION_PATCH`.

#### Changed

- installation configuration in CMake now uses the standard `GNUInstallDirs`

#### Fixed

- removed dependency on `bcrypt` on Windows
- the shared libraries is installed with execute permissions

## [Version 0.1.6](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-core-v0.1.6) - 2024-05-03

### metatensor-core C/C++

#### Fixed

- fixed compilation with Rust 1.78 (#605)
- fixed compilation on some Windows systems (#575)

## [Version 0.1.5](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-core-v0.1.5) - 2024-04-09

### metatensor-core C++

#### Fixed

- fixed compilation with cmake 3.29.1 (#573)

### metatensor-core Python

### Changed

-  allow positional arguments in `TensorMap.to`/`TensorBlock.to` (#556)


## [Version 0.1.4](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-core-v0.1.4) - 2024-03-01

### Fixed

- fixed compilation on macOS (#525)
- added checks for the size of the new values in `Labels.insert` (#519)

## [Version 0.1.3](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-core-v0.1.3) - 2024-02-12

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
