# This file defines the CMake build system for the C and C++ API of metatensor.
#
# This API is implemented in Rust, in the metatensor-core crate, but Rust users
# of the API should use the metatensor crate instead, wrapping metatensor-core in
# an easier to use, idiomatic Rust API.
cmake_minimum_required(VERSION 3.16)

# Is metatensor the main project configured by the user? Or is this being used
# as a submodule/subdirectory?
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(METATENSOR_MAIN_PROJECT ON)
else()
    set(METATENSOR_MAIN_PROJECT OFF)
endif()

if(${METATENSOR_MAIN_PROJECT} AND NOT "${CACHED_LAST_CMAKE_VERSION}" VERSION_EQUAL ${CMAKE_VERSION})
    # We use CACHED_LAST_CMAKE_VERSION to only print the cmake version
    # once in the configuration log
    set(CACHED_LAST_CMAKE_VERSION ${CMAKE_VERSION} CACHE INTERNAL "Last version of cmake used to configure")
    message(STATUS "Running CMake version ${CMAKE_VERSION}")
endif()

if (POLICY CMP0077)
    # use variables to set OPTIONS
    cmake_policy(SET CMP0077 NEW)
endif()

file(STRINGS "Cargo.toml" CARGO_TOML_CONTENT)
foreach(line ${CARGO_TOML_CONTENT})
    string(REGEX REPLACE "^version = \"(.*)\"" "\\1" METATENSOR_VERSION ${line})
    if (NOT ${CMAKE_MATCH_COUNT} EQUAL 0)
        # stop on the first regex match, this should be the right version
        break()
    endif()
endforeach()

include(cmake/dev-versions.cmake)
create_development_version("${METATENSOR_VERSION}" "metatensor-core-v" METATENSOR_FULL_VERSION)
message(STATUS "Building metatensor-core v${METATENSOR_FULL_VERSION}")

# strip any -dev/-rc suffix on the version since project(VERSION) does not support it
string(REGEX REPLACE "([0-9]*)\\.([0-9]*)\\.([0-9]*).*" "\\1.\\2.\\3" METATENSOR_VERSION ${METATENSOR_FULL_VERSION})
project(metatensor
    VERSION ${METATENSOR_VERSION}
    LANGUAGES C CXX # we need to declare a language to access CMAKE_SIZEOF_VOID_P later
)
set(PROJECT_VERSION ${METATENSOR_FULL_VERSION})


# We follow the standard CMake convention of using BUILD_SHARED_LIBS to provide
# either a shared or static library as a default target. But since cargo always
# builds both versions by default, we also install both versions by default.
# `METATENSOR_INSTALL_BOTH_STATIC_SHARED=OFF` allow to disable this behavior, and
# only install the file corresponding to `BUILD_SHARED_LIBS=ON/OFF`.
#
# BUILD_SHARED_LIBS controls the `metatensor` cmake target, making it an alias of
# either `metatensor::static` or `metatensor::shared`. This is mainly relevant
# when using metatensor from another cmake project, either as a submodule or from
# an installed library (see cmake/metatensor-config.cmake)
option(BUILD_SHARED_LIBS "Use a shared library by default instead of a static one" ON)
option(METATENSOR_INSTALL_BOTH_STATIC_SHARED "Install both shared and static libraries" ON)

set(BIN_INSTALL_DIR "bin" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install binaries/DLL")
set(LIB_INSTALL_DIR "lib" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install libraries")
set(INCLUDE_INSTALL_DIR "include" CACHE PATH "Path relative to CMAKE_INSTALL_PREFIX where to install headers")
set(RUST_BUILD_TARGET "" CACHE STRING "Cross-compilation target for rust code. Leave empty to build for the host")
set(EXTRA_RUST_FLAGS "" CACHE STRING "Flags used to build rust code")

set(CMAKE_MACOSX_RPATH ON)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${LIB_INSTALL_DIR}")

if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
    message(STATUS "Setting build type to 'release' as none was specified.")
    set(CMAKE_BUILD_TYPE "release"
        CACHE STRING
        "Choose the type of build, options are: debug or release"
    FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
endif()

if(${METATENSOR_MAIN_PROJECT} AND NOT "${CACHED_LAST_CMAKE_BUILD_TYPE}" STREQUAL "${CMAKE_BUILD_TYPE}")
    set(CACHED_LAST_CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE INTERNAL "Last build type used in configuration")
    message(STATUS "Building metatensor in ${CMAKE_BUILD_TYPE} mode")
endif()

find_program(CARGO_EXE "cargo" DOC "path to cargo (Rust build system)")
if (NOT CARGO_EXE)
    message(FATAL_ERROR
        "could not find cargo, please make sure the Rust compiler is installed \
        (see https://www.rust-lang.org/tools/install) or set CARGO_EXE"
    )
endif()

execute_process(
    COMMAND ${CARGO_EXE} "--version" "--verbose"
    RESULT_VARIABLE CARGO_STATUS
    OUTPUT_VARIABLE CARGO_VERSION_RAW
)

if(CARGO_STATUS AND NOT CARGO_STATUS EQUAL 0)
    message(FATAL_ERROR
        "could not run cargo, please make sure the Rust compiler is installed \
        (see https://www.rust-lang.org/tools/install)"
    )
endif()

set(REQUIRED_RUST_VERSION "1.53.0")
if (CARGO_VERSION_RAW MATCHES "cargo ([0-9]+\\.[0-9]+\\.[0-9]+).*")
    set(CARGO_VERSION "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR "failed to determine cargo version, output was: ${CARGO_VERSION_RAW}")
endif()

if (${CARGO_VERSION} VERSION_LESS ${REQUIRED_RUST_VERSION})
    message(FATAL_ERROR
        "your Rust installation is too old (you have version ${CARGO_VERSION}), \
        at least ${REQUIRED_RUST_VERSION} is required"
    )
else()
    if(NOT "${CACHED_LAST_CARGO_VERSION}" STREQUAL ${CARGO_VERSION})
        set(CACHED_LAST_CARGO_VERSION ${CARGO_VERSION} CACHE INTERNAL "Last version of cargo used in configuration")
        message(STATUS "Using cargo version ${CARGO_VERSION} at ${CARGO_EXE}")
        set(CARGO_VERSION_CHANGED TRUE)
    endif()
endif()

# ============================================================================ #
# determine Cargo flags

set(CARGO_BUILD_ARG "")

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock)
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--locked")
endif()

# TODO: support multiple configuration generators (MSVC, ...)
string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)
if ("${BUILD_TYPE}" STREQUAL "debug")
    set(CARGO_BUILD_TYPE "debug")
elseif("${BUILD_TYPE}" STREQUAL "release")
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--release")
    set(CARGO_BUILD_TYPE "release")
elseif("${BUILD_TYPE}" STREQUAL "relwithdebinfo")
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--release")
    set(CARGO_BUILD_TYPE "release")
else()
    message(FATAL_ERROR "unsuported build type: ${CMAKE_BUILD_TYPE}")
endif()

set(CARGO_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR}/target)
set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--target-dir=${CARGO_TARGET_DIR}")

# Handle cross compilation with RUST_BUILD_TARGET
if ("${RUST_BUILD_TARGET}" STREQUAL "")
    if (CARGO_VERSION_RAW MATCHES "host: ([a-zA-Z0-9_\\-]*)\n")
        set(RUST_HOST_TARGET "${CMAKE_MATCH_1}")
    else()
        message(FATAL_ERROR "failed to determine host target, output was: ${CARGO_VERSION_RAW}")
    endif()

    if (${METATENSOR_MAIN_PROJECT})
        message(STATUS "Compiling to host (${RUST_HOST_TARGET})")
    endif()

    set(CARGO_OUTPUT_DIR "${CARGO_TARGET_DIR}/${CARGO_BUILD_TYPE}")
    set(RUST_BUILD_TARGET ${RUST_HOST_TARGET})
else()
    if (${METATENSOR_MAIN_PROJECT})
        message(STATUS "Cross-compiling to ${RUST_BUILD_TARGET}")
    endif()

    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--target=${RUST_BUILD_TARGET}")
    set(CARGO_OUTPUT_DIR "${CARGO_TARGET_DIR}/${RUST_BUILD_TARGET}/${CARGO_BUILD_TYPE}")
endif()


# Get the list of libraries linked by default by cargo/rustc to add when linking
# to metatensor::static
if (CARGO_VERSION_CHANGED)
    include(cmake/tempdir.cmake)
    get_tempdir(TMPDIR)

    # Adapted from https://github.com/corrosion-rs/corrosion/blob/dc1e4e5/cmake/FindRust.cmake
    execute_process(
        COMMAND "${CARGO_EXE}" new --lib _cargo_required_libs
        WORKING_DIRECTORY "${TMPDIR}"
        RESULT_VARIABLE cargo_new_result
        ERROR_QUIET
    )

    if (cargo_new_result)
        message(FATAL_ERROR "could not create empty project to find default static libs: ${cargo_new_result}")
    endif()

    file(APPEND "${TMPDIR}/_cargo_required_libs/Cargo.toml" "[lib]\ncrate-type=[\"staticlib\"]")

    execute_process(
        COMMAND ${CARGO_EXE} rustc --color never --target=${RUST_BUILD_TARGET} -- --print=native-static-libs
        WORKING_DIRECTORY "${TMPDIR}/_cargo_required_libs"
        RESULT_VARIABLE cargo_build_result
        ERROR_VARIABLE cargo_build_error_message
    )

    # clean up the files
    file(REMOVE_RECURSE "${TMPDIR}")

    if(cargo_build_result)
        message(FATAL_ERROR "could extract default static libs: ${cargo_build_result}")
    endif()

    # The pattern starts with `native-static-libs:` and goes to the end of the line.
    if(cargo_build_error_message MATCHES "native-static-libs: ([^\r\n]+)\r?\n")
        string(REPLACE " " ";" "libs_list" "${CMAKE_MATCH_1}")
        set(stripped_lib_list "")
        foreach(lib ${libs_list})
            # Strip leading `-l` (unix) and potential .lib suffix (windows)
            string(REGEX REPLACE "^-l" "" "stripped_lib" "${lib}")
            string(REGEX REPLACE "\.lib$" "" "stripped_lib" "${stripped_lib}")
            list(APPEND stripped_lib_list "${stripped_lib}")
        endforeach()

        # Special case `msvcrt` to link with the debug version in Debug mode.
        list(TRANSFORM stripped_lib_list REPLACE "^msvcrt$" "\$<\$<CONFIG:Debug>:msvcrtd>")

        if (APPLE)
            # Prevent warnings about duplicated `System` in linked libraries
            # from Apple's `ld`
            list(REMOVE_ITEM stripped_lib_list "System")
        endif()

        list(REMOVE_DUPLICATES stripped_lib_list)
        set(CARGO_DEFAULT_LIBRARIES "${stripped_lib_list}" CACHE INTERNAL "list of implicitly linked libraries")

        if (${METATENSOR_MAIN_PROJECT})
            message(STATUS "Cargo default link libraries are: ${CARGO_DEFAULT_LIBRARIES}")
        endif()
    else()
        message(FATAL_ERROR "could not find default static libs: `native-static-libs` not found in: `${cargo_build_error_message}`")
    endif()
endif()

file(GLOB_RECURSE ALL_RUST_SOURCES
    ${PROJECT_SOURCE_DIR}/Cargo.toml
    ${PROJECT_SOURCE_DIR}/src/**.rs
)

add_library(metatensor::shared SHARED IMPORTED GLOBAL)
set(METATENSOR_SHARED_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}metatensor${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(METATENSOR_IMPLIB_LOCATION "${METATENSOR_SHARED_LOCATION}.lib")

if (MINGW)
    # `rustc` does not follow the usual naming scheme for DLL with mingw (it
    # would typically be 'libmetatensor.dll')
    set(METATENSOR_SHARED_LOCATION "${CARGO_OUTPUT_DIR}/metatensor.dll")
    set(METATENSOR_IMPLIB_LOCATION "${CARGO_OUTPUT_DIR}/libmetatensor.dll.a")
endif()

add_library(metatensor::static STATIC IMPORTED GLOBAL)
set(METATENSOR_STATIC_LOCATION "${CARGO_OUTPUT_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}metatensor${CMAKE_STATIC_LIBRARY_SUFFIX}")

get_filename_component(METATENSOR_SHARED_LIB_NAME ${METATENSOR_SHARED_LOCATION} NAME)
get_filename_component(METATENSOR_IMPLIB_NAME     ${METATENSOR_IMPLIB_LOCATION} NAME)
get_filename_component(METATENSOR_STATIC_LIB_NAME ${METATENSOR_STATIC_LOCATION} NAME)

# We need to add some metadata to the shared library to enable linking to it
# without using an absolute path.
if (UNIX)
    if (APPLE)
        # set the install name to `@rpath/libmetatensor.dylib`
        set(CARGO_RUSTC_ARGS "-Clink-arg=-Wl,-install_name,@rpath/${METATENSOR_SHARED_LIB_NAME}")
        set_target_properties(metatensor::shared PROPERTIES
            IMPORTED_SONAME @rpath/${METATENSOR_SHARED_LIB_NAME}
        )
    else() # LINUX
        # set the SONAME to libmetatensor.so
        set(CARGO_RUSTC_ARGS "-Clink-arg=-Wl,-soname,${METATENSOR_SHARED_LIB_NAME}")
        set_target_properties(metatensor::shared PROPERTIES
            IMPORTED_SONAME ${METATENSOR_SHARED_LIB_NAME}
        )
    endif()
else()
    set(CARGO_RUSTC_ARGS "")
endif()

if (NOT "${EXTRA_RUST_FLAGS}" STREQUAL "")
    set(CARGO_RUSTC_ARGS "${CARGO_RUSTC_ARGS};${EXTRA_RUST_FLAGS}")
endif()

if (METATENSOR_INSTALL_BOTH_STATIC_SHARED)
    set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--crate-type=cdylib;--crate-type=staticlib")
    set(FILES_CREATED_BY_CARGO "${METATENSOR_SHARED_LIB_NAME} and ${METATENSOR_STATIC_LIB_NAME}")
else()
    if (BUILD_SHARED_LIBS)
        set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--crate-type=cdylib")
        set(FILES_CREATED_BY_CARGO "${METATENSOR_SHARED_LIB_NAME}")
    else()
        set(CARGO_BUILD_ARG "${CARGO_BUILD_ARG};--crate-type=staticlib")
        set(FILES_CREATED_BY_CARGO "${METATENSOR_STATIC_LIB_NAME}")
    endif()
endif()

# Set environement variables for cargo build
set(CARGO_ENV "METATENSOR_FULL_VERSION=${METATENSOR_FULL_VERSION}")
if (NOT "${CMAKE_OSX_DEPLOYMENT_TARGET}" STREQUAL "")
    list(APPEND CARGO_ENV "MACOSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif()
if (NOT "$ENV{RUSTC_WRAPPER}" STREQUAL "")
    list(APPEND CARGO_ENV "RUSTC_WRAPPER=$ENV{RUSTC_WRAPPER}")
endif()

add_custom_target(cargo-build-metatensor ALL
    COMMAND ${CMAKE_COMMAND} -E env ${CARGO_ENV}
        cargo rustc ${CARGO_BUILD_ARG} -- ${CARGO_RUSTC_ARGS}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    DEPENDS ${ALL_RUST_SOURCES}
    COMMENT "Building ${FILES_CREATED_BY_CARGO} with cargo"
    BYPRODUCTS ${METATENSOR_STATIC_LOCATION} ${METATENSOR_SHARED_LOCATION} ${METATENSOR_IMPLIB_LOCATION}
)

add_dependencies(metatensor::shared cargo-build-metatensor)
add_dependencies(metatensor::static cargo-build-metatensor)
set(METATENSOR_HEADERS
    "${PROJECT_SOURCE_DIR}/include/metatensor.h"
    "${PROJECT_SOURCE_DIR}/include/metatensor.hpp"
)
set(METATENSOR_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/)

set_target_properties(metatensor::shared PROPERTIES
    IMPORTED_LOCATION ${METATENSOR_SHARED_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${METATENSOR_INCLUDE_DIR}
    BUILD_VERSION "${METATENSOR_FULL_VERSION}"
)
target_compile_features(metatensor::shared INTERFACE cxx_std_11)

if (WIN32)
    set_target_properties(metatensor::shared PROPERTIES
        IMPORTED_IMPLIB ${METATENSOR_IMPLIB_LOCATION}
    )
endif()

set_target_properties(metatensor::static PROPERTIES
    IMPORTED_LOCATION ${METATENSOR_STATIC_LOCATION}
    INTERFACE_INCLUDE_DIRECTORIES ${METATENSOR_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES "${CARGO_DEFAULT_LIBRARIES}"
    BUILD_VERSION "${METATENSOR_FULL_VERSION}"
)
target_compile_features(metatensor::static INTERFACE cxx_std_11)


if (BUILD_SHARED_LIBS)
    add_library(metatensor ALIAS metatensor::shared)
else()
    add_library(metatensor ALIAS metatensor::static)
endif()

#------------------------------------------------------------------------------#
# Installation configuration
#------------------------------------------------------------------------------#
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/metatensor-config.in.cmake
    ${PROJECT_BINARY_DIR}/metatensor-config.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/metatensor
)
write_basic_package_version_file(
    metatensor-config-version.cmake
    VERSION ${METATENSOR_FULL_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(FILES ${METATENSOR_HEADERS} DESTINATION ${INCLUDE_INSTALL_DIR})

if (METATENSOR_INSTALL_BOTH_STATIC_SHARED OR BUILD_SHARED_LIBS)
    if (WIN32)
        # DLL files should go in <prefix>/bin
        install(FILES ${METATENSOR_SHARED_LOCATION} DESTINATION ${BIN_INSTALL_DIR})
        # .lib files should go in <prefix>/lib
        install(FILES ${METATENSOR_IMPLIB_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
    else()
        install(FILES ${METATENSOR_SHARED_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
    endif()
endif()

if (METATENSOR_INSTALL_BOTH_STATIC_SHARED OR NOT BUILD_SHARED_LIBS)
    install(FILES ${METATENSOR_STATIC_LOCATION} DESTINATION ${LIB_INSTALL_DIR})
endif()

install(FILES
    ${PROJECT_BINARY_DIR}/metatensor-config-version.cmake
    ${PROJECT_BINARY_DIR}/metatensor-config.cmake
    DESTINATION ${LIB_INSTALL_DIR}/cmake/metatensor
)
