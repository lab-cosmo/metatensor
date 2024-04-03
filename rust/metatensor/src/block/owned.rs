use crate::c_api::mts_block_t;
use crate::errors::check_status;
use crate::{Array, ArrayRef, Labels, Error};

use super::{TensorBlockRef, TensorBlockRefMut};

/// A single block, containing both values & optionally gradients of these
/// values w.r.t. any relevant quantity.
#[derive(Debug)]
#[repr(transparent)]
pub struct TensorBlock {
    ptr: *mut mts_block_t,
}

// SAFETY: TensorBlock can be freed from any thread
unsafe impl Send for TensorBlock {}
// SAFETY: Sync is fine since there is no internal mutability in TensorBlock
unsafe impl Sync for TensorBlock {}

impl std::ops::Drop for TensorBlock {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        unsafe {
            crate::c_api::mts_block_free(self.as_mut_ptr());
        }
    }
}

impl TensorBlock {
    /// Create a new `TensorBlock` from a raw pointer.
    ///
    /// This function takes ownership of the pointer, and will call
    /// `mts_block_free` on it when the `TensorBlock` goes out of scope.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and point to a owned block, not a reference
    /// to a block from inside a [`TensorMap`](crate::TensorMap).
    pub(crate) unsafe fn from_raw(ptr: *mut mts_block_t) -> TensorBlock {
        assert!(!ptr.is_null(), "pointer to mts_block_t should not be NULL");

        TensorBlock {
            ptr: ptr,
        }
    }

    /// Get the underlying raw pointer
    pub(super) fn as_ptr(&self) -> *const mts_block_t {
        self.ptr
    }

    /// Get the underlying (mutable) raw pointer
    pub(super) fn as_mut_ptr(&mut self) -> *mut mts_block_t {
        self.ptr
    }

    /// Get a non mutable reference to this block
    #[inline]
    pub fn as_ref(&self) -> TensorBlockRef<'_> {
        unsafe {
            TensorBlockRef::from_raw(self.as_ptr())
        }
    }

    /// Get a non mutable reference to this block
    #[inline]
    pub fn as_ref_mut(&mut self) -> TensorBlockRefMut<'_> {
        unsafe {
            TensorBlockRefMut::from_raw(self.as_mut_ptr())
        }
    }

    /// Get the array for the values in this block
    #[inline]
    pub fn values(&self) -> ArrayRef<'_> {
        return self.as_ref().values();
    }

    /// Get the samples for this block
    #[inline]
    pub fn samples(&self) -> Labels {
        return self.as_ref().samples();
    }

    /// Get the components for this block
    #[inline]
    pub fn components(&self) -> Vec<Labels> {
        return self.as_ref().components();
    }

    /// Get the properties for this block
    #[inline]
    pub fn properties(&self) -> Labels {
        return self.as_ref().properties();
    }

    /// Create a new [`TensorBlock`] containing the given data, described by the
    /// `samples`, `components`, and `properties` labels. The block is
    /// initialized without any gradients.
    #[inline]
    pub fn new(
        data: impl Array,
        samples: &Labels,
        components: &[Labels],
        properties: &Labels
    ) -> Result<TensorBlock, Error> {
        let mut c_components = Vec::new();
        for component in components {
            c_components.push(component.as_mts_labels_t());
        }

        let ptr = unsafe {
            crate::c_api::mts_block(
                (Box::new(data) as Box<dyn Array>).into(),
                samples.as_mts_labels_t(),
                c_components.as_ptr(),
                c_components.len(),
                properties.as_mts_labels_t(),
            )
        };

        crate::errors::check_ptr(ptr)?;

        return Ok(unsafe { TensorBlock::from_raw(ptr) });
    }

    /// Add a gradient with respect to `parameter` to this block.
    ///
    /// The property of the gradient should match the ones of this block. The
    /// components of the gradients must contain at least the same entries as
    /// the value components, and can prepend other components.
    #[allow(clippy::needless_pass_by_value)]
    #[inline]
    pub fn add_gradient(
        &mut self,
        parameter: &str,
        mut gradient: TensorBlock
    ) -> Result<(), Error> {
        let mut parameter = parameter.to_owned().into_bytes();
        parameter.push(b'\0');


        let gradient_ptr = gradient.as_ref_mut().as_mut_ptr();
        // we give ownership of the gradient to `self`, so we should not free
        // them again from here
        std::mem::forget(gradient);

        unsafe {
            check_status(crate::c_api::mts_block_add_gradient(
                self.as_ref_mut().as_mut_ptr(),
                parameter.as_ptr().cast(),
                gradient_ptr,
            ))?;
        }

        return Ok(());
    }
}


#[cfg(test)]
mod tests {
    use crate::c_api::mts_block_t;
    use super::*;

    #[test]
    fn check_repr() {
        // we are casting `*mut TensorBlock` to `*mut mts_block_t` in TensorMap::new,
        // this is only legal because TensorBlock == *mut mts_block_t
        assert_eq!(std::mem::size_of::<TensorBlock>(), std::mem::size_of::<*mut mts_block_t>());
        assert_eq!(std::mem::align_of::<TensorBlock>(), std::mem::align_of::<*mut mts_block_t>());
    }
}
