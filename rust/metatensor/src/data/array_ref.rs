use crate::c_api::mts_array_t;
use crate::data::origin::get_data_origin;

use super::Array;

/// Reference to a data array in metatensor-core
///
/// The data array can come from any origin, this struct provides facilities to
/// access data that was created through the [`Array`] trait, and in particular
/// as `ndarray::ArrayD` instances.
#[derive(Debug, Clone, Copy)]
pub struct ArrayRef<'a> {
    array: mts_array_t,
    /// ArrayRef should behave like &'a mts_array_t
    marker: std::marker::PhantomData<&'a mts_array_t>,
}

impl<'a> ArrayRef<'a> {
    /// Create a new `ArrayRef` from the given raw `mts_array_t`
    ///
    /// This is a **VERY** unsafe function, creating a lifetime out of thin air.
    /// Make sure the lifetime is actually constrained by the lifetime of the
    /// owner of this `mts_array_t`.
    pub unsafe fn from_raw(array: mts_array_t) -> ArrayRef<'a> {
        ArrayRef {
            array,
            marker: std::marker::PhantomData,
        }
    }

    /// Get the underlying array as an `&dyn Any` instance.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn as_any(&self) -> &dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Get a reference to the underlying array as an `&dyn Any` instance,
    /// re-using the same lifetime as the `ArrayRef`.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn to_any(self) -> &'a dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Get the data in this `ArrayRef` as a `ndarray::ArrayD`. This function
    /// will panic if the data in this `mts_array_t` is not a `ndarray::ArrayD`.
    #[inline]
    pub fn as_array(&self) -> &ndarray::ArrayD<f64> {
        self.as_any().downcast_ref().expect("this is not a ndarray::ArrayD")
    }

    /// Transform this `ArrayRef` into a reference to an `ndarray::ArrayD`,
    /// keeping the lifetime of the `ArrayRef`.
    ///
    /// This function will panic if the data in this `mts_array_t` is not a
    /// `ndarray::ArrayD`.
    #[inline]
    pub fn to_array(self) -> &'a ndarray::ArrayD<f64> {
        self.to_any().downcast_ref().expect("this is not a ndarray::ArrayD")
    }

    /// Get the raw underlying `mts_array_t`
    pub fn as_raw(&self) -> &mts_array_t {
        &self.array
    }
}

/// Mutable reference to a data array in metatensor-core
///
/// The data array can come from any origin, this struct provides facilities to
/// access data that was created through the [`Array`] trait, and in particular
/// as `ndarray::ArrayD` instances.
#[derive(Debug)]
pub struct ArrayRefMut<'a> {
    array: mts_array_t,
    /// ArrayRef should behave like &'a mut mts_array_t
    marker: std::marker::PhantomData<&'a mut mts_array_t>,
}

impl<'a> ArrayRefMut<'a> {
    /// Create a new `ArrayRefMut` from the given raw `mts_array_t`
    ///
    /// This is a **VERY** unsafe function, creating a lifetime out of thin air,
    /// and allowing mutable access to the `mts_array_t`. Make sure the lifetime
    /// is actually constrained by the lifetime of the owner of this
    /// `mts_array_t`; and that the owner is mutably borrowed by this
    /// `ArrayRefMut`.
    #[inline]
    pub unsafe fn new(array: mts_array_t) -> ArrayRefMut<'a> {
        ArrayRefMut {
            array,
            marker: std::marker::PhantomData,
        }
    }

    /// Get the underlying array as an `&dyn Any` instance.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn as_any(&self) -> &dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Get the underlying array as an `&dyn Any` instance,
    /// re-using the same lifetime as the `ArrayRefMut`.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn to_any(&self) -> &'a dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any();
        }
    }

    /// Get the underlying array as an `&mut dyn Any` instance.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any_mut();
        }
    }

    /// Get the underlying array as an `&mut dyn Any` instance, re-using the
    /// same lifetime as the `ArrayRefMut`.
    ///
    /// This function panics if the array was not created though this crate and
    /// the [`Array`] trait.
    #[inline]
    pub fn to_any_mut(self) -> &'a mut dyn std::any::Any {
        let origin = self.array.origin().unwrap_or(0);
        assert_eq!(
            origin, *super::array::RUST_DATA_ORIGIN,
            "this array was not created as a rust Array (origin is '{}')",
            get_data_origin(origin).unwrap_or_else(|_| "unknown".into())
        );

        let array = self.array.ptr.cast::<Box<dyn Array>>();
        unsafe {
            return (*array).as_any_mut();
        }
    }

    /// Get the data in this `ArrayRef` as a `ndarray::ArrayD`. This function
    /// will panic if the data in this `mts_array_t` is not a `ndarray::ArrayD`.
    #[inline]
    pub fn as_array(&self) -> &ndarray::ArrayD<f64> {
        self.as_any().downcast_ref().expect("this is not a ndarray::ArrayD")
    }

    /// Transform this `ArrayRefMut` into a reference to an `ndarray::ArrayD`,
    /// keeping the lifetime of the `ArrayRefMut`.
    ///
    /// This function will panic if the data in this `mts_array_t` is not a
    /// `ndarray::ArrayD`.
    #[inline]
    pub fn to_array(&self) -> &ndarray::ArrayD<f64> {
        self.to_any().downcast_ref().expect("this is not a ndarray::ArrayD")
    }

    /// Get the data in this `ArrayRef` as a mutable reference to an
    /// `ndarray::ArrayD`. This function will panic if the data in this
    /// `mts_array_t` is not a `ndarray::ArrayD`.
    #[inline]
    pub fn as_array_mut(&mut self) -> &mut ndarray::ArrayD<f64> {
        self.as_any_mut().downcast_mut().expect("this is not a ndarray::ArrayD")
    }

    /// Transform this `ArrayRefMut` into a mutable reference to an
    /// `ndarray::ArrayD`, keeping the lifetime of the `ArrayRefMut`.
    ///
    /// This function will panic if the data in this `mts_array_t` is not a
    /// `ndarray::ArrayD`.
    #[inline]
    pub fn to_array_mut(self) -> &'a mut ndarray::ArrayD<f64> {
        self.to_any_mut().downcast_mut().expect("this is not a ndarray::ArrayD")
    }

    /// Get the raw underlying `mts_array_t`
    pub fn as_raw(&self) -> &mts_array_t {
        &self.array
    }

    /// Get a mutable reference to the raw underlying `mts_array_t`
    pub fn as_raw_mut(&mut self) -> &mut mts_array_t {
        &mut self.array
    }
}
