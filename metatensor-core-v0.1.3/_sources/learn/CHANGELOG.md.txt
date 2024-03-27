# Changelog

All notable changes to metatensor-learn are documented here, following the
[keep a changelog](https://keepachangelog.com/en/1.1.0/) format. This project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased](https://github.com/lab-cosmo/metatensor/)

<!-- Possible sections

### Added

### Fixed

### Changed

### Removed
-->

## [Version 0.2.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-learn-v0.2.0) - 2024-02-07

### Changed

- Pluralization removed for special kwarg `sample_ids` of `IndexedDataset` ->
  `sample_id`, and provided collate functions `group` and `group_and_join`
  updated accordingly.

### Fixed

- Removal of usage of Labels.range in nn modules to support torch.jit.save (#410)

## [Version 0.1.0](https://github.com/lab-cosmo/metatensor/releases/tag/metatensor-learn-v0.1.0) - 2024-01-26

### Added

- `ModuleMap` and `Linear` modules, following torch.nn.ModuleDict and
  torch.nn.Linear in PyTorch but adapted for `TensorMap`'s (#427)
- `Dataset` and `DataLoader` facilities, following the corresponding classes in
  PyTorch (#428)
