# metatensor-sys crate

This crate contains the Rust declaration matching the C API of metatensor. It
also builds and links to the metatensor shared library.

## Features

This crate offers one feature: `static` which uses a static build of metatensor
instead of a shared one. Unless you know that your code will never interact with
another metatensor-based codebase, it is better to not use this feature.
