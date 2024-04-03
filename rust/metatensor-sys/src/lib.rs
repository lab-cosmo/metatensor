#![allow(clippy::needless_return)]

mod c_api;

pub use c_api::*;


impl mts_labels_t {
    /// Create an `mts_labels_t` with all members set to null pointers/zero
    pub fn null() -> mts_labels_t {
        mts_labels_t {
            internal_ptr_: std::ptr::null_mut(),
            names: std::ptr::null(),
            values: std::ptr::null(),
            size: 0,
            count: 0,
        }
    }
}

/// Error type used in metatensor
#[derive(Debug, Clone)]
pub struct Error {
    /// Error code from the metatensor-core C API
    pub code: Option<mts_status_t>,
    /// Error message associated with the code
    pub message: String
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(code) = self.code {
            write!(f, " (code {})", code)?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

// Box<dyn Any + Send + 'static> is the error type in std::panic::catch_unwind
impl From<Box<dyn std::any::Any + Send + 'static>> for Error {
    fn from(error: Box<dyn std::any::Any + Send + 'static>) -> Error {
        let message = if let Some(message) = error.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = error.downcast_ref::<&str>() {
            (*message).to_owned()
        } else {
            panic!("panic message is not a string, something is very wrong")
        };

        Error {
            code: None, message,
        }
    }
}

/// Check the status code returned by arbitrary functions inside an
/// `mts_array_t`
fn check_status_external(status: mts_status_t, function: &str) -> Result<(), Error> {
    if status == MTS_SUCCESS {
        return Ok(());
    } else if status > 0 {
        let message = unsafe {
            std::ffi::CStr::from_ptr(mts_last_error())
        };
        let message = message.to_str().expect("invalid UTF8");

        return Err(Error { code: Some(status), message: message.to_owned() });
    } else {
        return Err(Error { code: Some(status), message: format!("calling {} failed", function) });
    }
}

impl mts_array_t {
    /// Create an `mts_array_t` with all members set to null pointers/None
    pub fn null() -> mts_array_t {
        mts_array_t {
            ptr: std::ptr::null_mut(),
            origin: None,
            data: None,
            shape: None,
            reshape: None,
            swap_axes: None,
            create: None,
            copy: None,
            destroy: None,
            move_samples_from: None,
        }
    }

    /// call `mts_array_t.origin` with a more convenient API
    pub fn origin(&self) -> Result<mts_data_origin_t, Error> {
        let function = self.origin.expect("mts_array_t.origin function is NULL");

        let mut origin = 0;
        unsafe {
            check_status_external(
                function(self.ptr, &mut origin),
                "mts_array_t.origin",
            )?;
        }

        return Ok(origin);
    }

    /// call `mts_array_t.shape` with a more convenient API
    #[allow(clippy::cast_possible_truncation)]
    pub fn shape(&self) -> Result<&[usize], Error> {
        let function = self.shape.expect("mts_array_t.shape function is NULL");

        let mut shape = std::ptr::null();
        let mut shape_count: usize = 0;

        unsafe {
            check_status_external(
                function(self.ptr, &mut shape, &mut shape_count),
                "mts_array_t.shape"
            )?;
        }

        assert!(shape_count > 0);
        let shape = unsafe {
            std::slice::from_raw_parts(shape, shape_count)
        };

        return Ok(shape);
    }

    /// call `mts_array_t.data` with a more convenient API
    pub fn data(&mut self) -> Result<&mut [f64], Error> {
        let shape = self.shape()?;
        let mut len = 1;
        for s in shape {
            len *= s;
        }

        let function = self.data.expect("mts_array_t.data function is NULL");

        let mut data_ptr = std::ptr::null_mut();
        let data = unsafe {
            check_status_external(
                function(self.ptr, &mut data_ptr),
                "mts_array_t.data"
            )?;

            if len == 0 {
                &mut []
            } else {
                assert!(!data_ptr.is_null());
                std::slice::from_raw_parts_mut(data_ptr, len)
            }
        };

        return Ok(data);
    }

    /// call `mts_array_t.reshape` with a more convenient API
    pub fn reshape(&mut self, shape: &[usize]) -> Result<(), Error> {
        let function = self.reshape.expect("mts_array_t.reshape function is NULL");

        unsafe {
            check_status_external(
                function(self.ptr, shape.as_ptr(), shape.len()),
                "mts_array_t.reshape",
            )?;
        }

        return Ok(());
    }

    /// call `mts_array_t.swap_axes` with a more convenient API
    pub fn swap_axes(&mut self, axis_1: usize, axis_2: usize) -> Result<(), Error> {
        let function = self.swap_axes.expect("mts_array_t.swap_axes function is NULL");

        unsafe {
            check_status_external(
                function(self.ptr, axis_1, axis_2),
                "mts_array_t.swap_axes",
            )?;
        }

        return Ok(());
    }

    /// call `mts_array_t.create` with a more convenient API
    pub fn create(&self, shape: &[usize]) -> Result<mts_array_t, Error> {
        let function = self.create.expect("mts_array_t.create function is NULL");

        let mut data_storage = mts_array_t {
            ptr: std::ptr::null_mut(),
            origin: None,
            data: None,
            shape: None,
            reshape: None,
            swap_axes: None,
            create: None,
            copy: None,
            destroy: None,
            move_samples_from: None
        };
        unsafe {
            check_status_external(
                function(self.ptr, shape.as_ptr(), shape.len(), &mut data_storage),
                "mts_array_t.create",
            )?;
        }

        return Ok(data_storage);
    }

    /// call `mts_array_t.move_samples_from` with a more convenient API
    pub fn move_samples_from(
        &mut self,
        input: &mts_array_t,
        samples: &[mts_sample_mapping_t],
        properties: std::ops::Range<usize>,
    ) -> Result<(), Error> {
        let function = self.move_samples_from.expect("mts_array_t.move_samples_from function is NULL");

        unsafe {
            check_status_external(
                function(
                    self.ptr,
                    input.ptr,
                    samples.as_ptr(),
                    samples.len(),
                    properties.start,
                    properties.end,
                ),
                "mts_array_t.move_samples_from",
            )?;
        }

        return Ok(());
    }
}

/// Path where the metatensor shared library has been built
pub fn c_api_install_dir() -> &'static str {
    env!("OUT_DIR")
}
