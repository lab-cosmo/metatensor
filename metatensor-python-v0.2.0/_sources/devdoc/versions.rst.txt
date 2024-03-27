.. _devdoc-versions:

Version number management
=========================

As explained in :ref:`devdoc-architecture`, the metatensor git repository
contains multiple packages --- each with its own version number --- for multiple
languages --- each with their own way of managing versions numbers. This page
explains how we handle versions and how to update version number for a release.

The core idea is that the version number stored in source file will always be
the **last released version**, and when handling development versions we
dynamically change this version number if necessary.

The overall sources of truth for version numbers are:

- the ``version`` field in ``metatensor-core/Cargo.toml`` for
  ``metatensor-core`` C, C++ API and Python API.
- the ``version`` field in ``metatensor/Cargo.toml`` for the ``metatensor`` Rust
  crate.
- the ``VERSION`` file in ``metatensor-torch/`` for ``metatensor-core`` C++ and
  Python API.
- the ``METATENSOR_OPERATIONS_VERSION`` variable in
  ``python/metatensor-operations/setup.py`` for the ``metatensor-operations``
  Python package.
- the ``METATENSOR_VERSION`` variable in ``setup.py`` for the ``metatensor``
  Python package.

When doing a build from a git checkout, we use
``scripts/n-commits-since-last-tag.py`` to determine the number of commits since
the last tag matching the current package. If there where no commits since the
last tag, the version number from the list above is kept as-is. If there where
commits, the version number if updated, and a ``dev<X>`` suffix is added to
indicate this is a development version.

Doing a release
---------------

When doing a release of any of the packages, the first step is to determine the
new version number using `semantic versioning`_ rules. Then, you should update
the version number at the right location from the list above. Finally, you
should add a tag on the corresponding commit, following this naming convention
(replacing ``<x.y.z>`` with the new version number).

- ``metatensor-core-v<x.y.z>`` for releases of ``metatensor-core``;
- ``metatensor-torch-v<x.y.z>`` for releases of ``metatensor-torch``;
- ``metatensor-python-v<x.y.z>`` for releases of the Python ``metatensor``
  package;
- ``metatensor-operations-v<x.y.z>`` for releases of the Python
  ``metatensor-operations`` package;
- ``metatensor-rust-v<x.y.z>`` for releases of the Rust ``metatensor`` crate;

When releasing multiple packages at the same time, the corresponding commit will
have multiple tags.

.. _semantic versioning: https://semver.org/
