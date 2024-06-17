# Changelog

All notable changes to metatensor-operations are documented here, following the
[keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/lab-cosmo/metatensor/)

<!-- Possible sections

### Added

### Fixed

### Changed

### Removed
-->

## [Version 0.2.2](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-operations-v0.2.2) - 2024-06-19

### Fixed

- Fixed a bug in `metatensor.torch.sort` where the labels where not properly
  sorted (#647)
- Fix `metatensor.abs` when used with complex values (#553)


## [Version 0.2.1](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-operations-v0.2.1) - 2024-03-01

### Changed

- Use `torch.jit.script` by default on all operations when using the TorchScript
  backend (i.e. `metatensor.torch`) (#504)

## [Version 0.2.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-operations-v0.2.0) - 2024-02-07

### Changed

- `join()` operation now includes the `sort_samples` argument to control the
  sorting of samples in the output joined TensorMap. This defaults to False, opposite
  to the previous default behaviour.

### Added

- `detach()` operation to detach all values in a TensorMap/TensorBlock from any
  computational graph
- `requires_grad()` operation to set the `requires_grad` of all values in a
  TensorMap/TensorBlock when storing data in torch Tensors.

### Removed

- the `to` operation was removed. Similar functionality is now offered by
  `TensorMap.to`, `TensorBlock.to`, and the operations `detach()` and
  `requires_grad()`.

## [Version 0.1.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-operations-v0.1.0) - 2023-10-11

### Added

- Creation operations: `empty_like()`, `ones_like()`, `zeros_like()`,
  `random_like()`, `block_from_array()`;
- Linear algebra: `dot()`, `lstsq()`, `solve()`;
- Logic function: `allclose()`, `equal()`, `equal_metadata()`;
- Manipulation operations: `drop_blocks()`, `join()`, `manipulate dimension`,
  `one_hot()`, `remove_gradients()`, `samples reduction`, `slice()`, `split()`,
  `to()`;
- Mathematical functions: `abs()`, `add()`, `divide()`, `multiply()`, `pow()`,
  `subtract()`;
- Set operations: `unique_metadata()`, `sort()`;
