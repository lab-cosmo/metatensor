use std::ffi::CString;

use crate::c_api::mts_labels_t;
use crate::errors::check_status;
use crate::{Labels, Error};

use super::realloc_vec;

/// Save labels to the file at the given path.
///
/// If the file already exists, it is overwritten
pub fn load_labels(path: impl AsRef<std::path::Path>) -> Result<Labels, Error> {
    let path = path.as_ref().as_os_str().to_str().expect("this path is not valid UTF8");
    let path = CString::new(path).expect("this path contains a NULL byte");

    let mut labels = mts_labels_t::null();

    unsafe {
        check_status(crate::c_api::mts_labels_load(path.as_ptr(), &mut labels))?;
    }

    return Ok(unsafe { Labels::from_raw(labels) });
}

/// Load a serialized `TensorMap` from a `buffer`.
pub fn load_labels_buffer(buffer: &[u8]) -> Result<Labels, Error> {
    let mut labels = mts_labels_t::null();

    unsafe {
        check_status(crate::c_api::mts_labels_load_buffer(
            buffer.as_ptr(),
            buffer.len(),
            &mut labels
        ))?;
    }

    return Ok(unsafe { Labels::from_raw(labels) });
}

/// Save the given Labels to a file.
pub fn save_labels(path: impl AsRef<std::path::Path>, labels: &Labels) -> Result<(), Error> {
    let path = path.as_ref().as_os_str().to_str().expect("this path is not valid UTF8");
    let path = CString::new(path).expect("this path contains a NULL byte");

    unsafe {
        check_status(crate::c_api::mts_labels_save(path.as_ptr(), labels.raw))
    }
}


/// Save the given `labels` to an in-memory `buffer`.
///
/// This function will grow the buffer as required to fit the labels.
pub fn save_labels_buffer(labels: &Labels, buffer: &mut Vec<u8>) -> Result<(), Error> {
    let mut buffer_ptr = buffer.as_mut_ptr();
    let mut buffer_count = buffer.len();

    unsafe {
        check_status(crate::c_api::mts_labels_save_buffer(
            &mut buffer_ptr,
            &mut buffer_count,
            (buffer as *mut Vec<u8>).cast(),
            Some(realloc_vec),
            labels.raw,
        ))?;
    }

    buffer.resize(buffer_count, 0);

    Ok(())
}
