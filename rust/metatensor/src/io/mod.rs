//! Input/Output facilities for storing [`crate::TensorMap`] and
//! [`crate::Labels`] on disk

use std::os::raw::c_void;
use crate::c_api::MTS_SUCCESS;

mod tensor;
pub use self::tensor::{load, save, load_buffer, save_buffer};

mod labels;
pub use self::labels::{load_labels, load_labels_buffer, save_labels, save_labels_buffer};


/// Implementation of realloc for `Vec<u8>`, used in `save_buffer`
unsafe extern fn realloc_vec(user_data: *mut c_void, _ptr: *mut u8, new_size: usize) -> *mut u8 {
    let mut result = std::ptr::null_mut();
    let unwind_wrapper = std::panic::AssertUnwindSafe(&mut result);

    let status = crate::errors::catch_unwind(move || {
        let vector = &mut *user_data.cast::<Vec<u8>>();
        vector.resize(new_size, 0);

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *(unwind_wrapper.0) = vector.as_mut_ptr();
    });

    if status != MTS_SUCCESS {
        return std::ptr::null_mut();
    }

    return result;
}
