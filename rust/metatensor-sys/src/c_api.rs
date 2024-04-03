#![allow(warnings)]
//! Rust definition corresponding to metatensor-core C-API.
//!
//! This module is exported for advanced users of the metatensor crate, but
//! should not be needed by most.

#[cfg_attr(feature="static", link(name="metatensor", kind = "static", modifiers = "-whole-archive"))]
#[cfg_attr(all(not(feature="static"), not(target_os="windows")), link(name="metatensor", kind = "dylib"))]
#[cfg_attr(all(not(feature="static"), target_os="windows"), link(name="metatensor.dll", kind = "dylib"))]
extern "C" {}

pub const MTS_SUCCESS: i32 = 0;
pub const MTS_INVALID_PARAMETER_ERROR: i32 = 1;
pub const MTS_IO_ERROR: i32 = 2;
pub const MTS_SERIALIZATION_ERROR: i32 = 3;
pub const MTS_BUFFER_SIZE_ERROR: i32 = 254;
pub const MTS_INTERNAL_ERROR: i32 = 255;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_block_t {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_tensormap_t {
    _unused: [u8; 0],
}
pub type mts_status_t = i32;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_labels_t {
    pub internal_ptr_: *mut ::std::os::raw::c_void,
    pub names: *const *const ::std::os::raw::c_char,
    pub values: *const i32,
    pub size: usize,
    pub count: usize,
}
#[test]
fn bindgen_test_layout_mts_labels_t() {
    const UNINIT: ::std::mem::MaybeUninit<mts_labels_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<mts_labels_t>(),
        40usize,
        concat!("Size of: ", stringify!(mts_labels_t))
    );
    assert_eq!(
        ::std::mem::align_of::<mts_labels_t>(),
        8usize,
        concat!("Alignment of ", stringify!(mts_labels_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).internal_ptr_) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_labels_t),
            "::",
            stringify!(internal_ptr_)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).names) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_labels_t),
            "::",
            stringify!(names)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).values) as usize - ptr as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_labels_t),
            "::",
            stringify!(values)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).size) as usize - ptr as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_labels_t),
            "::",
            stringify!(size)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).count) as usize - ptr as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_labels_t),
            "::",
            stringify!(count)
        )
    );
}
pub type mts_data_origin_t = u64;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_sample_mapping_t {
    pub input: usize,
    pub output: usize,
}
#[test]
fn bindgen_test_layout_mts_sample_mapping_t() {
    const UNINIT: ::std::mem::MaybeUninit<mts_sample_mapping_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<mts_sample_mapping_t>(),
        16usize,
        concat!("Size of: ", stringify!(mts_sample_mapping_t))
    );
    assert_eq!(
        ::std::mem::align_of::<mts_sample_mapping_t>(),
        8usize,
        concat!("Alignment of ", stringify!(mts_sample_mapping_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).input) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_sample_mapping_t),
            "::",
            stringify!(input)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).output) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_sample_mapping_t),
            "::",
            stringify!(output)
        )
    );
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct mts_array_t {
    pub ptr: *mut ::std::os::raw::c_void,
    pub origin: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            origin: *mut mts_data_origin_t,
        ) -> mts_status_t,
    >,
    pub data: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            data: *mut *mut f64,
        ) -> mts_status_t,
    >,
    pub shape: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            shape: *mut *const usize,
            shape_count: *mut usize,
        ) -> mts_status_t,
    >,
    pub reshape: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            shape: *const usize,
            shape_count: usize,
        ) -> mts_status_t,
    >,
    pub swap_axes: ::std::option::Option<
        unsafe extern "C" fn(
            array: *mut ::std::os::raw::c_void,
            axis_1: usize,
            axis_2: usize,
        ) -> mts_status_t,
    >,
    pub create: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            shape: *const usize,
            shape_count: usize,
            new_array: *mut mts_array_t,
        ) -> mts_status_t,
    >,
    pub copy: ::std::option::Option<
        unsafe extern "C" fn(
            array: *const ::std::os::raw::c_void,
            new_array: *mut mts_array_t,
        ) -> mts_status_t,
    >,
    pub destroy: ::std::option::Option<unsafe extern "C" fn(array: *mut ::std::os::raw::c_void)>,
    pub move_samples_from: ::std::option::Option<
        unsafe extern "C" fn(
            output: *mut ::std::os::raw::c_void,
            input: *const ::std::os::raw::c_void,
            samples: *const mts_sample_mapping_t,
            samples_count: usize,
            property_start: usize,
            property_end: usize,
        ) -> mts_status_t,
    >,
}
#[test]
fn bindgen_test_layout_mts_array_t() {
    const UNINIT: ::std::mem::MaybeUninit<mts_array_t> = ::std::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::std::mem::size_of::<mts_array_t>(),
        80usize,
        concat!("Size of: ", stringify!(mts_array_t))
    );
    assert_eq!(
        ::std::mem::align_of::<mts_array_t>(),
        8usize,
        concat!("Alignment of ", stringify!(mts_array_t))
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).ptr) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_array_t),
            "::",
            stringify!(ptr)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).origin) as usize - ptr as usize },
        8usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_array_t),
            "::",
            stringify!(origin)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).data) as usize - ptr as usize },
        16usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_array_t),
            "::",
            stringify!(data)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).shape) as usize - ptr as usize },
        24usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_array_t),
            "::",
            stringify!(shape)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).reshape) as usize - ptr as usize },
        32usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_array_t),
            "::",
            stringify!(reshape)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).swap_axes) as usize - ptr as usize },
        40usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_array_t),
            "::",
            stringify!(swap_axes)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).create) as usize - ptr as usize },
        48usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_array_t),
            "::",
            stringify!(create)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).copy) as usize - ptr as usize },
        56usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_array_t),
            "::",
            stringify!(copy)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).destroy) as usize - ptr as usize },
        64usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_array_t),
            "::",
            stringify!(destroy)
        )
    );
    assert_eq!(
        unsafe { ::std::ptr::addr_of!((*ptr).move_samples_from) as usize - ptr as usize },
        72usize,
        concat!(
            "Offset of field: ",
            stringify!(mts_array_t),
            "::",
            stringify!(move_samples_from)
        )
    );
}
pub type mts_realloc_buffer_t = ::std::option::Option<
    unsafe extern "C" fn(
        user_data: *mut ::std::os::raw::c_void,
        ptr: *mut u8,
        new_size: usize,
    ) -> *mut u8,
>;
pub type mts_create_array_callback_t = ::std::option::Option<
    unsafe extern "C" fn(
        shape: *const usize,
        shape_count: usize,
        array: *mut mts_array_t,
    ) -> mts_status_t,
>;
extern "C" {
    pub fn mts_disable_panic_printing();
    pub fn mts_version() -> *const ::std::os::raw::c_char;
    pub fn mts_last_error() -> *const ::std::os::raw::c_char;
    #[must_use]
    pub fn mts_labels_position(
        labels: mts_labels_t,
        values: *const i32,
        values_count: usize,
        result: *mut i64,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_create(labels: *mut mts_labels_t) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_set_user_data(
        labels: mts_labels_t,
        user_data: *mut ::std::os::raw::c_void,
        user_data_delete: ::std::option::Option<
            unsafe extern "C" fn(arg1: *mut ::std::os::raw::c_void),
        >,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_user_data(
        labels: mts_labels_t,
        user_data: *mut *mut ::std::os::raw::c_void,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_clone(labels: mts_labels_t, clone: *mut mts_labels_t) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_union(
        first: mts_labels_t,
        second: mts_labels_t,
        result: *mut mts_labels_t,
        first_mapping: *mut i64,
        first_mapping_count: usize,
        second_mapping: *mut i64,
        second_mapping_count: usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_intersection(
        first: mts_labels_t,
        second: mts_labels_t,
        result: *mut mts_labels_t,
        first_mapping: *mut i64,
        first_mapping_count: usize,
        second_mapping: *mut i64,
        second_mapping_count: usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_free(labels: *mut mts_labels_t) -> mts_status_t;
    #[must_use]
    pub fn mts_register_data_origin(
        name: *const ::std::os::raw::c_char,
        origin: *mut mts_data_origin_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_get_data_origin(
        origin: mts_data_origin_t,
        buffer: *mut ::std::os::raw::c_char,
        buffer_size: usize,
    ) -> mts_status_t;
    pub fn mts_block(
        data: mts_array_t,
        samples: mts_labels_t,
        components: *const mts_labels_t,
        components_count: usize,
        properties: mts_labels_t,
    ) -> *mut mts_block_t;
    #[must_use]
    pub fn mts_block_free(block: *mut mts_block_t) -> mts_status_t;
    pub fn mts_block_copy(block: *const mts_block_t) -> *mut mts_block_t;
    #[must_use]
    pub fn mts_block_labels(
        block: *const mts_block_t,
        axis: usize,
        labels: *mut mts_labels_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_block_gradient(
        block: *mut mts_block_t,
        parameter: *const ::std::os::raw::c_char,
        gradient: *mut *mut mts_block_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_block_data(block: *mut mts_block_t, data: *mut mts_array_t) -> mts_status_t;
    #[must_use]
    pub fn mts_block_add_gradient(
        block: *mut mts_block_t,
        parameter: *const ::std::os::raw::c_char,
        gradient: *mut mts_block_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_block_gradients_list(
        block: *const mts_block_t,
        parameters: *mut *const *const ::std::os::raw::c_char,
        parameters_count: *mut usize,
    ) -> mts_status_t;
    pub fn mts_tensormap(
        keys: mts_labels_t,
        blocks: *mut *mut mts_block_t,
        blocks_count: usize,
    ) -> *mut mts_tensormap_t;
    #[must_use]
    pub fn mts_tensormap_free(tensor: *mut mts_tensormap_t) -> mts_status_t;
    pub fn mts_tensormap_copy(tensor: *const mts_tensormap_t) -> *mut mts_tensormap_t;
    #[must_use]
    pub fn mts_tensormap_keys(
        tensor: *const mts_tensormap_t,
        keys: *mut mts_labels_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_tensormap_block_by_id(
        tensor: *mut mts_tensormap_t,
        block: *mut *mut mts_block_t,
        index: usize,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_tensormap_blocks_matching(
        tensor: *const mts_tensormap_t,
        block_indexes: *mut usize,
        count: *mut usize,
        selection: mts_labels_t,
    ) -> mts_status_t;
    pub fn mts_tensormap_keys_to_properties(
        tensor: *const mts_tensormap_t,
        keys_to_move: mts_labels_t,
        sort_samples: bool,
    ) -> *mut mts_tensormap_t;
    pub fn mts_tensormap_components_to_properties(
        tensor: *mut mts_tensormap_t,
        dimensions: *const *const ::std::os::raw::c_char,
        dimensions_count: usize,
    ) -> *mut mts_tensormap_t;
    pub fn mts_tensormap_keys_to_samples(
        tensor: *const mts_tensormap_t,
        keys_to_move: mts_labels_t,
        sort_samples: bool,
    ) -> *mut mts_tensormap_t;
    #[must_use]
    pub fn mts_labels_load(
        path: *const ::std::os::raw::c_char,
        labels: *mut mts_labels_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_load_buffer(
        buffer: *const u8,
        buffer_count: usize,
        labels: *mut mts_labels_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_save(
        path: *const ::std::os::raw::c_char,
        labels: mts_labels_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_labels_save_buffer(
        buffer: *mut *mut u8,
        buffer_count: *mut usize,
        realloc_user_data: *mut ::std::os::raw::c_void,
        realloc: mts_realloc_buffer_t,
        labels: mts_labels_t,
    ) -> mts_status_t;
    pub fn mts_tensormap_load(
        path: *const ::std::os::raw::c_char,
        create_array: mts_create_array_callback_t,
    ) -> *mut mts_tensormap_t;
    pub fn mts_tensormap_load_buffer(
        buffer: *const u8,
        buffer_count: usize,
        create_array: mts_create_array_callback_t,
    ) -> *mut mts_tensormap_t;
    #[must_use]
    pub fn mts_tensormap_save(
        path: *const ::std::os::raw::c_char,
        tensor: *const mts_tensormap_t,
    ) -> mts_status_t;
    #[must_use]
    pub fn mts_tensormap_save_buffer(
        buffer: *mut *mut u8,
        buffer_count: *mut usize,
        realloc_user_data: *mut ::std::os::raw::c_void,
        realloc: mts_realloc_buffer_t,
        tensor: *const mts_tensormap_t,
    ) -> mts_status_t;
}
