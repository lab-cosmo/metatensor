use crate::c_api::mts_data_origin_t;

use crate::errors::{check_status, Error};

/// Register a new data origin with the given name
pub(super) fn register_data_origin(name: String) -> Result<mts_data_origin_t, Error> {
    let mut origin = 0 as mts_data_origin_t;

    let mut name = name.into_bytes();
    name.push(b'\0');

    unsafe {
        check_status(crate::c_api::mts_register_data_origin(
            name.as_ptr().cast(),
            &mut origin
        ))?;
    }

    return Ok(origin);
}

/// Get the name associated with a data origin
pub(super) fn get_data_origin(origin: mts_data_origin_t) -> Result<String, Error> {
    use std::ffi::CStr;

    let mut buffer: Vec<u8> = vec![0; 32];
    loop {
        let status = unsafe {
            crate::c_api::mts_get_data_origin(origin, buffer.as_mut_ptr().cast(), buffer.len())
        };

        if status == crate::c_api::MTS_BUFFER_SIZE_ERROR {
            buffer.resize(2 * buffer.len(), 0);
        } else {
            check_status(status)?;
            break;
        }
    }

    let first_null = buffer.iter().position(|&c| c == 0).expect("should contain a NULL byte");
    buffer.resize(first_null + 1, 0);

    let string = CStr::from_bytes_with_nul(&buffer).expect("should have a single NULL byte");
    return Ok(string.to_str().expect("should be UTF8").to_owned());
}
