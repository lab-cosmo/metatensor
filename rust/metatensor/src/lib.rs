//! # Metatensor
//!
//! Metatensor is a library providing a specialized data storage format for
//! atomistic machine learning (think ``numpy`` or ``torch.Tensor``, but also
//! carrying extra metadata for atomistic systems).
//!
//! The core of the format is implemented in `metatensor-core`, and exported as a
//! C API. This C API is then re-exported to Rust in this crate. By default,
//! metatensor-core is distributed as a shared library that you'll need to
//! install separately on end user machines.
//!
//! ## Features
//!
//! You can enable the `static` feature in Cargo.toml to use a static build of
//! the C API, removing the need to carry around the metatensor-core shared
//! library.
//!
//! ```toml
//! [dependencies]
//! metatensor = {version = "...", features = ["static"]}
//! ```

#![warn(clippy::all, clippy::pedantic)]

// disable some style lints
#![allow(clippy::needless_return, clippy::must_use_candidate, clippy::comparison_chain)]
#![allow(clippy::redundant_field_names, clippy::redundant_closure_for_method_calls, clippy::redundant_else)]
#![allow(clippy::unreadable_literal, clippy::option_if_let_else, clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc, clippy::missing_safety_doc)]
#![allow(clippy::similar_names, clippy::borrow_as_ptr, clippy::uninlined_format_args)]

pub use metatensor_sys as c_api;

pub mod errors;
pub use self::errors::Error;

mod data;
pub use self::data::{ArrayRef, ArrayRefMut};
pub use self::data::{Array, EmptyArray};

mod labels;
pub use self::labels::{Labels, LabelsBuilder, LabelValue};
pub use self::labels::{LabelsIter, LabelsFixedSizeIter};

#[cfg(feature = "rayon")]
pub use self::labels::LabelsParIter;

mod block;
pub use self::block::{TensorBlock, TensorBlockRef, TensorBlockRefMut};
pub use self::block::{TensorBlockData, TensorBlockDataMut};
pub use self::block::{GradientsIter, GradientsMutIter};
pub use self::block::LazyMetadata;

mod tensor;
pub use self::tensor::TensorMap;
pub use self::tensor::{TensorMapIter, TensorMapIterMut};
#[cfg(feature = "rayon")]
pub use self::tensor::{TensorMapParIter, TensorMapParIterMut};

pub mod io;
