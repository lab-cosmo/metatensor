use std::os::raw::{c_char, c_void};
use std::ffi::CStr;
use std::sync::Arc;

use crate::{LabelValue, Labels, LabelsBuilder, Error};
use super::status::{mts_status_t, catch_unwind};

/// A set of labels used to carry metadata associated with a tensor map.
///
/// This is similar to a list of `count` named tuples, but stored as a 2D array
/// of shape `(count, size)`, with a set of names associated with the columns of
/// this array (often called *dimensions*). Each row/entry in this array is
/// unique, and they are often (but not always) sorted in lexicographic order.
///
/// `mts_labels_t` with a non-NULL `internal_ptr_` correspond to a
/// reference-counted Rust data structure, which allow for fast lookup inside
/// the labels with `mts_labels_positions`.

// An `mts_labels_t` can either correspond to a Rust `Arc<Labels>` (`labels_ptr`
// is non-NULL, and corresponds to the pointer `Arc::into_raw` gives); or to a
// set of Labels created from C, and containing pointer to C-allocated data.
#[repr(C)]
#[allow(non_camel_case_types)]
pub struct mts_labels_t {
    /// internal: pointer to the rust `Labels` struct if any, null otherwise
    pub internal_ptr_: *mut c_void,

    /// Names of the dimensions composing this set of labels. There are `size`
    /// elements in this array, each being a NULL terminated UTF-8 string.
    pub names: *const *const c_char,
    /// Pointer to the first element of a 2D row-major array of 32-bit signed
    /// integer containing the values taken by the different dimensions in
    /// `names`. Each row has `size` elements, and there are `count` rows in
    /// total.
    pub values: *const i32,
    /// Number of dimensions/size of a single entry in the set of labels
    pub size: usize,
    /// Number entries in the set of labels
    pub count: usize,
}

impl mts_labels_t {
    /// Check if these mts_labels_t are associated with a Rust `Arc<Labels>` or
    /// not
    pub fn is_rust(&self) -> bool {
        !self.internal_ptr_.is_null()
    }
}

/// Create a new `mts_labels_t` from a Rust `Arc<Labels>`
pub unsafe fn rust_to_mts_labels(labels: Arc<Labels>) -> mts_labels_t {
    let size = labels.size();
    let count = labels.count();

    let values = if labels.count() == 0 || labels.size() == 0 {
        std::ptr::null()
    } else {
        (&labels[0][0] as *const LabelValue).cast()
    };

    let names = if labels.size() == 0 {
        std::ptr::null()
    } else {
        labels.c_names().as_ptr().cast()
    };

    let internal_ptr_ = Arc::into_raw(labels).cast::<c_void>().cast_mut();

    mts_labels_t {
        internal_ptr_,
        names,
        values,
        size,
        count
    }
}

/// Convert from `mts_label_t` back to a Rust `Arc<Labels>`, potentially
/// constructing new `Labels` if they don't exist yet.
pub unsafe fn mts_labels_to_rust(labels: &mts_labels_t) -> Result<Arc<Labels>, Error> {
    // if the labels have already been constructed on the rust side,
    // increase the reference count of the arc & return that
    if labels.is_rust() {
        let labels = Arc::from_raw(labels.internal_ptr_.cast());
        let cloned = Arc::clone(&labels);

        // keep the original arc alive
        std::mem::forget(labels);

        return Ok(cloned);
    }

    // otherwise, create new labels from the data
    return create_rust_labels(labels);
}

/// Create a new set of rust Labels from `mts_labels_t`, copying the data into
/// Rust managed memory.
unsafe fn create_rust_labels(labels: &mts_labels_t) -> Result<Arc<Labels>, Error> {
    assert!(!labels.is_rust());

    if labels.size == 0 {
        if labels.count > 0 {
            return Err(Error::InvalidParameter("can not have labels.count > 0 if labels.size is 0".into()));
        }

        let builder = LabelsBuilder::new(vec![]).expect("invalid builder");
        return Ok(Arc::new(builder.finish()));
    }

    if labels.names.is_null() {
        return Err(Error::InvalidParameter("labels.names can not be NULL in mts_labels_t".into()))
    }

    if labels.values.is_null() && labels.count > 0 {
        return Err(Error::InvalidParameter("labels.values is NULL but labels.count is >0 in mts_labels_t".into()))
    }

    let mut names = Vec::new();
    for i in 0..labels.size {
        let name = CStr::from_ptr(*(labels.names.add(i)));
        let name = name.to_str().expect("invalid UTF8 name");
        if !crate::labels::is_valid_label_name(name) {
            return Err(Error::InvalidParameter(format!(
                "'{}' is not a valid label name", name
            )));
        }
        names.push(name);
    }

    let mut builder = LabelsBuilder::new(names)?;
    builder.reserve(labels.count);

    if labels.count != 0 && labels.size != 0 {
        assert!(!labels.values.is_null());
        let slice = std::slice::from_raw_parts(labels.values.cast::<LabelValue>(), labels.count * labels.size);
        for chunk in slice.chunks_exact(labels.size) {
            builder.add(chunk)?;
        }
    }

    return Ok(Arc::new(builder.finish()));
}


/// Get the position of the entry defined by the `values` array in the given set
/// of `labels`. This operation is only available if the labels correspond to a
/// set of Rust Labels (i.e. `labels.internal_ptr_` is not NULL).
///
/// @param labels set of labels with an associated Rust data structure
/// @param values array containing the label to lookup
/// @param values_count size of the values array
/// @param result position of the values in the labels or -1 if the values
///               were not found
///
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
#[allow(clippy::cast_possible_wrap)]
pub unsafe extern fn mts_labels_position(
    labels: mts_labels_t,
    values: *const i32,
    values_count: usize,
    result: *mut i64
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(values, result);
        if !labels.is_rust() {
            return Err(Error::InvalidParameter(
                "these labels do not support calling mts_labels_position, \
                call mts_labels_create first".into()
            ));
        }

        let labels = &(*labels.internal_ptr_.cast::<Labels>());
        if values_count != labels.size() {
            return Err(Error::InvalidParameter(format!(
                "expected label of size {} in mts_labels_position, got size {}",
                (*labels).size(), values_count
            )));
        }

        assert!(values_count != 0);
        let label = std::slice::from_raw_parts(values.cast(), values_count);
        *result = labels.position(label).map_or(-1, |p| p as i64);

        Ok(())
    })
}


/// Finish the creation of `mts_labels_t` by associating it to Rust-owned
/// labels.
///
/// This allows using the `mts_labels_positions` and `mts_labels_clone`
/// functions on the `mts_labels_t`.
///
/// This function allocates memory which must be released `mts_labels_free` when
/// you don't need it anymore.
///
/// @param labels new set of labels containing pointers to user-managed memory
///        on input, and pointers to Rust-managed memory on output.
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_labels_create(
    labels: *mut mts_labels_t,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(labels);

        if (*labels).is_rust() {
            return Err(Error::InvalidParameter(
                "these labels already correspond to rust labels".into()
            ));
        }

        let rust_labels = create_rust_labels(&*labels)?;
        *labels = rust_to_mts_labels(rust_labels);

        Ok(())
    })
}

/// Update the registered user data in `labels`
///
/// This function changes the registered user data in the Rust Labels to be
/// `user_data`; and store the corresponding `user_data_delete` function to be
/// called once the labels go out of scope.
///
/// Any existing user data will be released (by calling the provided
/// `user_data_delete` function) before overwriting with the new data.
///
/// @param labels set of labels where we want to add user data
/// @param user_data pointer to the data
/// @param user_data_delete function pointer that will be used (if not NULL)
///                         to free the memory associated with `data` when the
///                         `labels` are freed.
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_labels_set_user_data(
    labels: mts_labels_t,
    user_data: *mut c_void,
    user_data_delete: Option<unsafe extern fn(*mut c_void)>
) -> mts_status_t {
    catch_unwind(|| {
        if !labels.is_rust() {
            return Err(Error::InvalidParameter(
                "these labels do not support calling mts_labels_set_user_data, \
                call mts_labels_create first".into()
            ));
        }

        let rust_labels = &*labels.internal_ptr_.cast::<Labels>();
        rust_labels.set_user_data(user_data, user_data_delete);

        Ok(())
    })
}

/// Get the registered user data in `labels` in `*user_data`.
///
/// If no data has been registered, `*user_data` will be NULL.
///
/// @param labels set of labels containing user data
/// @param user_data this will be set to the pointer than was registered with
///                  these `labels`
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_labels_user_data(
    labels: mts_labels_t,
    user_data: *mut *mut c_void,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(user_data);

        if !labels.is_rust() {
            return Err(Error::InvalidParameter(
                "these labels do not support calling mts_labels_get_user_data, \
                call mts_labels_create first".into()
            ));
        }

        let rust_labels = &*labels.internal_ptr_.cast::<Labels>();
        *user_data = rust_labels.user_data();

        Ok(())
    })
}

/// Make a copy of `labels` inside `clone`.
///
/// Since `mts_labels_t` are immutable, the copy is actually just a reference
/// count increase, and as such should not be an expensive operation.
///
/// `mts_labels_free` must be used with `clone` to decrease the reference count
/// and release the memory when you don't need it anymore.
///
/// @param labels set of labels with an associated Rust data structure
/// @param clone empty labels, on output will contain a copy of `labels`
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_labels_clone(
    labels: mts_labels_t,
    clone: *mut mts_labels_t,
) -> mts_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(clone);
    catch_unwind(|| {
        if !labels.is_rust() {
            return Err(Error::InvalidParameter(
                "these labels do not support calling mts_labels_clone, \
                call mts_labels_create first".into()
            ));
        }

        if (*clone).is_rust() {
            return Err(Error::InvalidParameter(
                "output labels already contain some data".into()
            ));
        }

        let rust_labels = Arc::from_raw(labels.internal_ptr_.cast::<Labels>());

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = rust_to_mts_labels(Arc::clone(&rust_labels));

        // keep the original arc alive
        std::mem::forget(rust_labels);

        Ok(())
    })
}

/// common checks and transformations for the set operations
unsafe fn labels_set_common<'a>(
    operation: &str,
    first: &mts_labels_t,
    second: &mts_labels_t,
    first_mapping: *mut i64,
    first_mapping_count: usize,
    second_mapping: *mut i64,
    second_mapping_count: usize,
) -> Result<(&'a mut [i64], &'a mut [i64]), Error> {
    if !first.is_rust() {
        return Err(Error::InvalidParameter(format!(
            "the `first` labels do not support {}, call mts_labels_create first",
            operation
        )));
    }

    if !second.is_rust() {
        return Err(Error::InvalidParameter(format!(
            "the `second` labels do not support {}, call mts_labels_create first",
            operation
        )));
    }

    let first_mapping = if first_mapping.is_null() {
        &mut []
    } else {
        if first_mapping_count != first.count {
            return Err(Error::InvalidParameter(format!(
                "`first_mapping_count` ({}) must match the number of elements \
                in `first` ({}) but doesn't",
                first_mapping_count,
                first.count,
            )));
        }
        std::slice::from_raw_parts_mut(first_mapping, first_mapping_count)
    };

    let second_mapping = if second_mapping.is_null() {
        &mut []
    } else {
        if second_mapping_count != second.count {
            return Err(Error::InvalidParameter(format!(
                "`second_mapping_count` ({}) must match the number of elements \
                in `second` ({}) but doesn't",
                second_mapping_count,
                second.count,
            )));
        }
        std::slice::from_raw_parts_mut(second_mapping, second_mapping_count)
    };

    return Ok((first_mapping, second_mapping));
}

/// Take the union of two `mts_labels_t`.
///
/// If requested, this function can also give the positions in the union where
/// each entry of the input `mts_labels_t` ended up.
///
/// This function allocates memory for `result` which must be released
/// `mts_labels_free` when you don't need it anymore.
///
/// @param first first set of labels
/// @param second second set of labels
/// @param result empty labels, on output will contain the union of `first` and
///        `second`
/// @param first_mapping if you want the mapping from the positions of entries
///        in `first` to the positions in `result`, this should be a pointer
///        to an array containing `first.count` elements, to be filled by this
///        function. Otherwise it should be a `NULL` pointer.
/// @param first_mapping_count number of elements in the `first_mapping` array
/// @param second_mapping if you want the mapping from the positions of entries
///        in `second` to the positions in `result`, this should be a pointer
///        to an array containing `second.count` elements, to be filled by this
///        function. Otherwise it should be a `NULL` pointer.
/// @param second_mapping_count number of elements in the `second_mapping` array
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_labels_union(
    first: mts_labels_t,
    second: mts_labels_t,
    result: *mut mts_labels_t,
    first_mapping: *mut i64,
    first_mapping_count: usize,
    second_mapping: *mut i64,
    second_mapping_count: usize,
) -> mts_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(result);
    catch_unwind(|| {
        let (first_mapping, second_mapping) = labels_set_common(
            "union",
            &first,
            &second,
            first_mapping,
            first_mapping_count,
            second_mapping,
            second_mapping_count
        )?;

        let first = &*first.internal_ptr_.cast::<Labels>();
        let second = &*second.internal_ptr_.cast::<Labels>();

        let result_rust = first.union(
            second,
            first_mapping,
            second_mapping,
        )?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = rust_to_mts_labels(Arc::new(result_rust));

        Ok(())
    })
}

/// Take the intersection of two `mts_labels_t`.
///
/// If requested, this function can also give the positions in the intersection
/// where each entry of the input `mts_labels_t` ended up.
///
/// This function allocates memory for `result` which must be released
/// `mts_labels_free` when you don't need it anymore.
///
/// @param first first set of labels
/// @param second second set of labels
/// @param result empty labels, on output will contain the union of `first` and
///        `second`
/// @param first_mapping if you want the mapping from the positions of entries
///        in `first` to the positions in `result`, this should be a pointer to
///        an array containing `first.count` elements, to be filled by this
///        function. Otherwise it should be a `NULL` pointer. If an entry in
///        `first` is not used in `result`, the mapping will be set to -1.
/// @param first_mapping_count number of elements in the `first_mapping` array
/// @param second_mapping if you want the mapping from the positions of entries
///        in `second` to the positions in `result`, this should be a pointer
///        to an array containing `second.count` elements, to be filled by this
///        function. Otherwise it should be a `NULL` pointer. If an entry in
///        `first` is not used in `result`, the mapping will be set to -1.
/// @param second_mapping_count number of elements in the `second_mapping` array
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_labels_intersection(
    first: mts_labels_t,
    second: mts_labels_t,
    result: *mut mts_labels_t,
    first_mapping: *mut i64,
    first_mapping_count: usize,
    second_mapping: *mut i64,
    second_mapping_count: usize,
) -> mts_status_t {
    let unwind_wrapper = std::panic::AssertUnwindSafe(result);
    catch_unwind(|| {
        let (first_mapping, second_mapping) = labels_set_common(
            "intersection",
            &first,
            &second,
            first_mapping,
            first_mapping_count,
            second_mapping,
            second_mapping_count
        )?;

        let first = &*first.internal_ptr_.cast::<Labels>();
        let second = &*second.internal_ptr_.cast::<Labels>();

        let result_rust = first.intersection(
            second,
            first_mapping,
            second_mapping,
        )?;

        // force the closure to capture the full unwind_wrapper, not just
        // unwind_wrapper.0
        let _ = &unwind_wrapper;
        *unwind_wrapper.0 = rust_to_mts_labels(Arc::new(result_rust));

        Ok(())
    })
}

/// Select entries in the `labels` that match the `selection`.
///
/// The selection's names must be a subset of the name of the `labels` names.
///
/// All entries in the `labels` that match one of the entry in the `selection`
/// for all the selection's dimension will be picked. Any entry in the
/// `selection` but not in the `labels` will be ignored.
///
/// @param labels Labels on which to run the selection
/// @param selection definition of the selection criteria. Multiple entries are
///        interpreted as a logical `or` operation.
/// @param selected on input, a pointer to an array with space for
///        `*selected_count` entries. On output, the first `*selected_count`
///        values will contain the index in `labels` of selected entries.
/// @param selected_count on input, size of the `selected` array. On output,
///        this will contain the number of selected entries.
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_labels_select(
    labels: mts_labels_t,
    selection: mts_labels_t,
    selected: *mut i64,
    selected_count: *mut usize,
) -> mts_status_t {
    catch_unwind(|| {
        check_pointers_non_null!(selected, selected_count);

        if !labels.is_rust() {
            return Err(Error::InvalidParameter(
                "these `labels` do not support mts_labels_select, call mts_labels_create first".into()
            ));
        }

        if !selection.is_rust() {
            return Err(Error::InvalidParameter(
                "the `selection` do not support mts_labels_select, call mts_labels_create first".into()
            ));
        }

        if *selected_count != labels.count {
            return Err(Error::InvalidParameter(format!(
                "`selected_count` ({}) must match the number of elements \
                in `labels` ({}) but doesn't",
                *selected_count,
                labels.count,
            )));
        }

        let labels = &*labels.internal_ptr_.cast::<Labels>();
        let selection = &*selection.internal_ptr_.cast::<Labels>();
        let selected = std::slice::from_raw_parts_mut(selected, *selected_count);

        *selected_count = labels.select(selection, selected)?;

        Ok(())
    })
}

/// Decrease the reference count of `labels`, and release the corresponding
/// memory once the reference count reaches 0.
///
/// @param labels set of labels with an associated Rust data structure
/// @returns The status code of this operation. If the status is not
///          `MTS_SUCCESS`, you can use `mts_last_error()` to get the full
///          error message.
#[no_mangle]
pub unsafe extern fn mts_labels_free(
    labels: *mut mts_labels_t,
) -> mts_status_t {
    catch_unwind(|| {
        if labels.is_null() {
            return Ok(());
        }

        if !(*labels).is_rust() {
            return Ok(());
        }

        std::mem::drop(Arc::from_raw((*labels).internal_ptr_.cast::<Labels>()));

        (*labels).internal_ptr_ = std::ptr::null_mut();

        Ok(())
    })
}
