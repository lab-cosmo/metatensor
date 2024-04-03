use std::ffi::{CString, CStr};
use std::iter::FusedIterator;

use crate::c_api::{mts_block_t, mts_array_t, MTS_INVALID_PARAMETER_ERROR};
use crate::{ArrayRef, ArrayRefMut, Labels};

use super::{TensorBlockRef, LazyMetadata};
use super::block_ref::{get_samples, get_components, get_properties};

/// Mutable reference to a [`TensorBlock`](crate::TensorBlock)
#[derive(Debug)]
pub struct TensorBlockRefMut<'a> {
    ptr: *mut mts_block_t,
    marker: std::marker::PhantomData<&'a mut mts_block_t>,
}

// SAFETY: Send is fine since TensorBlockRefMut does not implement Drop
unsafe impl<'a> Send for TensorBlockRefMut<'a> {}
// SAFETY: Sync is fine since there is no internal mutability in TensorBlockRefMut
// (all mutations still require a `&mut TensorBlockRefMut`)
unsafe impl<'a> Sync for TensorBlockRefMut<'a> {}

/// All the basic data in a `TensorBlockRefMut` as a struct with separate fields.
///
/// This can be useful when you need to borrow different fields on this struct
/// separately. They are separate in the underlying metatensor-core, but since we
/// go through the C API to access them, we need to re-expose them as separate
/// fields for the rust compiler to be able to understand that.
///
/// The metadata is initialized lazily on first access, to not pay the cost of
/// allocation/reference count increase if some metadata is not used.
#[derive(Debug)]
pub struct TensorBlockDataMut<'a> {
    pub values: ArrayRefMut<'a>,
    pub samples: LazyMetadata<Labels>,
    pub components: LazyMetadata<Vec<Labels>>,
    pub properties: LazyMetadata<Labels>,
}

/// Get a gradient from this block
fn block_gradient(block: *mut mts_block_t, parameter: &CStr) -> Option<*mut mts_block_t> {
    let mut gradient_block = std::ptr::null_mut();
    let status = unsafe { crate::c_api::mts_block_gradient(
            block,
            parameter.as_ptr(),
            &mut gradient_block
        )
    };

    match crate::errors::check_status(status) {
        Ok(()) => Some(gradient_block),
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

impl<'a> TensorBlockRefMut<'a> {
    /// Create a new `TensorBlockRefMut` from the given raw `mts_block_t`
    ///
    /// This is a **VERY** unsafe function, creating a lifetime out of thin air,
    /// and allowing mutable access to the `mts_block_t`. Make sure the lifetime
    /// is actually constrained by the lifetime of the owner of this
    /// `mts_block_t`; and that the owner is mutably borrowed by this
    /// `TensorBlockRefMut`.
    pub(crate) unsafe fn from_raw(ptr: *mut mts_block_t) -> TensorBlockRefMut<'a> {
        assert!(!ptr.is_null(), "pointer to mts_block_t should not be NULL");

        TensorBlockRefMut {
            ptr: ptr,
            marker: std::marker::PhantomData,
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

    /// Get all the data and metadata inside this `TensorBlockRefMut` as a
    /// struct with separate fields, to allow borrowing them separately.
    #[inline]
    pub fn data_mut(&mut self) -> TensorBlockDataMut<'_> {
        let samples = LazyMetadata::new(get_samples, self.as_ptr());
        let components = LazyMetadata::new(get_components, self.as_ptr());
        let properties = LazyMetadata::new(get_properties, self.as_ptr());

        TensorBlockDataMut {
            // SAFETY: we are returning an `ArrayRefMut` mutably borrowing from `self`
            values: self.values_mut(),
            samples: samples,
            components: components,
            properties: properties,
        }
    }

    /// Get a mutable reference to the values in this block
    #[inline]
    pub fn values_mut(&mut self) -> ArrayRefMut<'_> {
        let mut array = mts_array_t::null();
        unsafe {
            crate::errors::check_status(crate::c_api::mts_block_data(
                self.as_mut_ptr(),
                &mut array
            )).expect("failed to get the array for a block");
        };

        // SAFETY: we are returning an `ArrayRefMut` mutably borrowing from `self`
        unsafe { ArrayRefMut::new(array) }
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

    /// Get a mutable reference to the data and metadata for the gradient with
    /// respect to the given parameter in this block, if it exists.
    #[inline]
    pub fn gradient_mut(&mut self, parameter: &str) -> Option<TensorBlockRefMut<'_>> {
        let parameter = CString::new(parameter).expect("invalid C string");

        block_gradient(self.as_mut_ptr(), &parameter)
            .map(|gradient_block| {
                // SAFETY: we are returning an `TensorBlockRefMut` mutably
                // borrowing from `self`
                unsafe { TensorBlockRefMut::from_raw(gradient_block) }
            })
    }

    /// Get an iterator over parameter/[`TensorBlockRefMut`] pairs for all gradients
    /// in this block
    #[inline]
    pub fn gradients_mut(&mut self) -> GradientsMutIter<'_> {
        let block_ptr = self.as_mut_ptr();
        GradientsMutIter {
            parameters: self.as_ref().gradient_list().into_iter(),
            block: block_ptr,
        }
    }
}

/// Iterator over parameter/[`TensorBlockRefMut`] pairs for all gradients in a
/// [`TensorBlockRefMut`]
pub struct GradientsMutIter<'a> {
    parameters: std::vec::IntoIter<&'a str>,
    block: *mut mts_block_t,
}

impl<'a> Iterator for GradientsMutIter<'a> {
    type Item = (&'a str, TensorBlockRefMut<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.parameters.next().map(|parameter| {
            let parameter_c = CString::new(parameter).expect("invalid C string");
            let block = block_gradient(self.block, &parameter_c).expect("missing gradient");

            // SAFETY: all blocks are disjoint, and we are only returning a
            // mutable reference to each once. The reference lifetime is
            // constrained by the lifetime of the parent TensorBlockRefMut
            let block = unsafe { TensorBlockRefMut::from_raw(block) };
            return (parameter, block);
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.parameters.size_hint()
    }
}

impl<'a> ExactSizeIterator for GradientsMutIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.parameters.len()
    }
}

impl<'a> FusedIterator for GradientsMutIter<'a> {}

#[cfg(test)]
mod tests {
    // TODO
}
