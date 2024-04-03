// TensorBlock can be manipulated in three forms: as standalone block owning
// it's data, or as a reference (or mutable reference) to a block inside a
// `TensorMap`. The three forms are represented by a pointer to `mts_block_t`
// in the C API, but we wrap them into three different types for Rust.

mod block_ref;
use crate::c_api::mts_block_t;

pub use self::block_ref::{TensorBlockRef, TensorBlockData, GradientsIter};

mod block_mut;
pub use self::block_mut::{TensorBlockRefMut, TensorBlockDataMut, GradientsMutIter};

mod owned;
pub use self::owned::TensorBlock;



/// Lazily accessed metadata inside a `TensorBlock`
///
/// This struct provides immutable access to an object of type `T`, that is
/// conceptually a field of a `TensorBlock`. The object is initialized on the
/// first access, potentially saving some computation/allocations if the object
/// is not needed.
pub struct LazyMetadata<T> {
    block: *const mts_block_t,
    init: fn(*const mts_block_t) -> T,
    metadata: once_cell::sync::OnceCell<T>,
}

impl<T: std::fmt::Debug> std::fmt::Debug for LazyMetadata<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lazy")
            .field("block", &self.block)
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl<T> std::ops::Deref for LazyMetadata<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.metadata.get_or_init(|| (self.init)(self.block))
    }
}

impl<T> LazyMetadata<T> {
    /// Create a new `LazyMetadata` with the given initialization function
    pub fn new(init: fn(*const mts_block_t) -> T, block: *const mts_block_t) -> LazyMetadata<T> {
        LazyMetadata {
            block: block,
            init: init,
            metadata: once_cell::sync::OnceCell::new(),
        }
    }
}
