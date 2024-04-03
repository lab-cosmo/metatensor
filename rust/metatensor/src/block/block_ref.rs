use std::ffi::{CStr, CString};
use std::iter::FusedIterator;

use crate::c_api::{mts_block_t, mts_array_t, mts_labels_t};
use crate::c_api::MTS_INVALID_PARAMETER_ERROR;

use crate::errors::check_status;
use crate::{ArrayRef, Labels, Error};

use super::{TensorBlock, LazyMetadata};

/// Reference to a [`TensorBlock`]
#[derive(Debug, Clone, Copy)]
pub struct TensorBlockRef<'a> {
    ptr: *const mts_block_t,
    marker: std::marker::PhantomData<&'a mts_block_t>,
}

// SAFETY: Send is fine since TensorBlockRef does not implement Drop
unsafe impl<'a> Send for TensorBlockRef<'a> {}
// SAFETY: Sync is fine since there is no internal mutability in TensorBlockRef
unsafe impl<'a> Sync for TensorBlockRef<'a> {}

/// All the basic data in a `TensorBlockRef` as a struct with separate fields.
///
/// This can be useful when you need to borrow different fields on this struct
/// separately. They are separate in the underlying metatensor-core, but since we
/// go through the C API to access them, we need to re-expose them as separate
/// fields for the rust compiler to be able to understand that.
///
/// The metadata is initialized lazily on first access, to not pay the cost of
/// allocation/reference count increase if some metadata is not used.
#[derive(Debug)]
pub struct TensorBlockData<'a> {
    pub values: ArrayRef<'a>,
    pub samples: LazyMetadata<Labels>,
    pub components: LazyMetadata<Vec<Labels>>,
    pub properties: LazyMetadata<Labels>,
}

impl<'a> TensorBlockRef<'a> {
    /// Create a new `TensorBlockRef` from the given raw `mts_block_t`
    ///
    /// This is a **VERY** unsafe function, creating a lifetime out of thin air.
    /// Make sure the lifetime is actually constrained by the lifetime of the
    /// owner of this `mts_block_t`.
    pub(crate) unsafe fn from_raw(ptr: *const mts_block_t) -> TensorBlockRef<'a> {
        assert!(!ptr.is_null(), "pointer to mts_block_t should not be NULL");

        TensorBlockRef {
            ptr: ptr,
            marker: std::marker::PhantomData,
        }
    }

    /// Get the underlying raw pointer
    pub(super) fn as_ptr(&self) -> *const mts_block_t {
        self.ptr
    }
}

/// Get a gradient from this block
fn block_gradient(block: *const mts_block_t, parameter: &CStr) -> Option<*const mts_block_t> {
    let mut gradient_block = std::ptr::null_mut();
    let status = unsafe { crate::c_api::mts_block_gradient(
            // the cast to mut pointer is fine since we are only returning a
            // non-mut mts_block_t below
            block.cast_mut(),
            parameter.as_ptr(),
            &mut gradient_block
        )
    };

    match crate::errors::check_status(status) {
        Ok(()) => Some(gradient_block.cast_const()),
        Err(error) => {
            if error.code == Some(MTS_INVALID_PARAMETER_ERROR) {
                // there is no array for this gradient
                None
            } else {
                panic!("failed to get the gradient from a block: {:?}", error)
            }
        }
    }
}

pub(super) fn get_samples(ptr: *const mts_block_t) -> Labels {
    unsafe {
        TensorBlockRef::from_raw(ptr).samples()
    }
}

pub(super) fn get_components(ptr: *const mts_block_t) -> Vec<Labels> {
    unsafe {
        TensorBlockRef::from_raw(ptr).components()
    }
}

pub(super) fn get_properties(ptr: *const mts_block_t) -> Labels {
    unsafe {
        TensorBlockRef::from_raw(ptr).properties()
    }
}

impl<'a> TensorBlockRef<'a> {
    /// Get all the data and metadata inside this `TensorBlockRef` as a
    /// struct with separate fields, to allow borrowing them separately.
    #[inline]
    pub fn data(&'a self) -> TensorBlockData<'a> {
        TensorBlockData {
            values: self.values(),
            samples: LazyMetadata::new(get_samples, self.as_ptr()),
            components: LazyMetadata::new(get_components, self.as_ptr()),
            properties: LazyMetadata::new(get_properties, self.as_ptr()),
        }
    }

    /// Get the array for the values in this block
    #[inline]
    pub fn values(&self) -> ArrayRef<'a> {
        let mut array = mts_array_t::null();
        unsafe {
            crate::errors::check_status(crate::c_api::mts_block_data(
                self.as_ptr().cast_mut(),
                &mut array
            )).expect("failed to get the array for a block");
        };

        // SAFETY: we can return an `ArrayRef` with lifetime `'a` (instead of
        // `'self`) (which allows to get multiple references to the BasicBlock
        // simultaneously), because there is no way to also get a mutable
        // reference to the block at the same time (since we are already holding
        // a const reference to the block itself).
        unsafe { ArrayRef::from_raw(array) }
    }

    #[inline]
    fn labels(&self, dimension: usize) -> Labels {
        let mut labels = mts_labels_t::null();
        unsafe {
            check_status(crate::c_api::mts_block_labels(
                self.as_ptr(),
                dimension,
                &mut labels,
            )).expect("failed to get labels");
        }
        return unsafe { Labels::from_raw(labels) };
    }

    /// Get the samples for this block
    #[inline]
    pub fn samples(&self) -> Labels {
        return self.labels(0);
    }

    /// Get the components for this block
    #[inline]
    pub fn components(&self) -> Vec<Labels> {
        let values = self.values();
        let shape = values.as_raw().shape().expect("failed to get the data shape");

        let mut result = Vec::new();
        for i in 1..(shape.len() - 1) {
            result.push(self.labels(i));
        }
        return result;
    }

    /// Get the properties for this block
    #[inline]
    pub fn properties(&self) -> Labels {
        let values = self.values();
        let shape = values.as_raw().shape().expect("failed to get the data shape");

        return self.labels(shape.len() - 1);
    }

    /// Get the full list of gradients in this block

    // SAFETY: we can return strings with the `'a` lifetime (instead of
    // `'self`), because there is no way to also get a mutable reference
    // to the gradient parameters at the same time.
    #[inline]
    pub fn gradient_list(&self) -> Vec<&'a str> {
        let mut parameters_ptr = std::ptr::null();
        let mut parameters_count = 0;
        unsafe {
            check_status(crate::c_api::mts_block_gradients_list(
                self.as_ptr(),
                &mut parameters_ptr,
                &mut parameters_count
            )).expect("failed to get gradient list");
        }

        if parameters_count == 0 {
            return Vec::new();
        } else {
            assert!(!parameters_ptr.is_null());
            unsafe {
                let parameters = std::slice::from_raw_parts(parameters_ptr, parameters_count);
                return parameters.iter()
                    .map(|&ptr| CStr::from_ptr(ptr).to_str().unwrap())
                    .collect();
            }
        }
    }

    /// Get the data and metadata for the gradient with respect to the given
    /// parameter in this block, if it exists.

    // SAFETY: we can return a TensorBlockRef with lifetime `'a` (instead of
    // `'self`) for the same reasons as in the `values` function.
    #[inline]
    pub fn gradient(&self, parameter: &str) -> Option<TensorBlockRef<'a>> {
        let parameter = CString::new(parameter).expect("invalid C string");

        block_gradient(self.as_ptr(), &parameter)
            .map(|gradient_block| {
                // SAFETY: the lifetime of the block is the same as
                // the lifetime of self, both are constrained to the
                // root TensorMap/TensorBlock
                unsafe { TensorBlockRef::from_raw(gradient_block) }
        })
    }

    /// Clone this block, cloning all the data and metadata contained inside.
    ///
    /// This can fail if the external data held inside an `mts_array_t` can not
    /// be cloned.
    #[inline]
    pub fn try_clone(&self) -> Result<TensorBlock, Error> {
        let ptr = unsafe {
            crate::c_api::mts_block_copy(self.as_ptr())
        };
        crate::errors::check_ptr(ptr)?;

        return Ok(unsafe { TensorBlock::from_raw(ptr) });
    }

    /// Get an iterator over parameter/[`TensorBlockRef`] pairs for all gradients in
    /// this block
    #[inline]
    pub fn gradients(&self) -> GradientsIter<'_> {
        GradientsIter {
            parameters: self.gradient_list().into_iter(),
            block: self.as_ptr(),
        }
    }
}

/// Iterator over parameter/[`TensorBlockRef`] pairs for all gradients in a
/// [`TensorBlockRef`]
pub struct GradientsIter<'a> {
    parameters: std::vec::IntoIter<&'a str>,
    block: *const mts_block_t,
}

impl<'a> Iterator for GradientsIter<'a> {
    type Item = (&'a str, TensorBlockRef<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.parameters.next().map(|parameter| {
            let parameter_c = CString::new(parameter).expect("invalid C string");
            let block = block_gradient(self.block, &parameter_c).expect("missing gradient");

            // SAFETY: the lifetime of the block is the same as the lifetime of
            // the GradientsIter, both are constrained to the root
            // TensorMap/TensorBlock
            let block = unsafe { TensorBlockRef::from_raw(block) };
            return (parameter, block);
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a> ExactSizeIterator for GradientsIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.parameters.len()
    }
}

impl<'a> FusedIterator for GradientsIter<'a> {}

#[cfg(test)]
mod tests {
    // TODO: check gradient/gradient iter code
}
