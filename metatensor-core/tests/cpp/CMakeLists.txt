cmake_minimum_required(VERSION 3.16)
project(metatensor-tests)

if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    if("${CMAKE_BUILD_TYPE}" STREQUAL "" AND "${CMAKE_CONFIGURATION_TYPES}" STREQUAL "")
        message(STATUS "Setting build type to 'release' as none was specified.")
        set(CMAKE_BUILD_TYPE "release"
            CACHE STRING
            "Choose the type of build, options are: debug or release"
        FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS release debug)
    endif()
endif()

add_subdirectory(../../ metatensor)
get_target_property(METATENSOR_IMPORTED_LOCATION metatensor::shared IMPORTED_LOCATION)
get_filename_component(METATENSOR_DIR ${METATENSOR_IMPORTED_LOCATION} DIRECTORY)

add_subdirectory(external)

find_program(VALGRIND valgrind)
if (VALGRIND)
    if (NOT "$ENV{METATENSOR_DISABLE_VALGRIND}" EQUAL "1")
        message(STATUS "Running tests using valgrind")
        set(TEST_COMMAND
            "${VALGRIND}" "--tool=memcheck" "--dsymutil=yes" "--error-exitcode=125"
            "--leak-check=full" "--show-leak-kinds=definite,indirect,possible" "--track-origins=yes"
            "--gen-suppressions=all"
        )
    endif()
else()
    set(TEST_COMMAND "")
endif()


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "-Weverything")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-weak-vtables")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-float-equal")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-prototypes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow-uncaptured-local")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unsafe-buffer-usage")
endif()


enable_testing()

file(GLOB ALL_TESTS *.cpp)
foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(${_name_} ${_file_})
    target_link_libraries(${_name_} metatensor catch)

    set_target_properties(${_name_} PROPERTIES
        # Ensure that the binaries find the right shared library.
        #
        # Without this, when configuring with cmake before the library is built,
        # cmake does not find the library on the filesystem and does not add the
        # RPATH to executables linking to it
        BUILD_RPATH ${METATENSOR_DIR}
        NO_SYSTEM_FROM_IMPORTED ON
    )

    # Give path to tests data files to the tests through defines
    target_compile_definitions(${_name_} PRIVATE "-DTEST_DATA_NPZ_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../data.npz\"")
    target_compile_definitions(${_name_} PRIVATE "-DTEST_KEYS_NPY_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/../keys.npy\"")

    add_test(
        NAME ${_name_}
        COMMAND ${TEST_COMMAND} $<TARGET_FILE:${_name_}>
    )

    if(WIN32)
        # We need to set the path to allow access to metatensor.dll
        # this does a similar job to the BUILD_RPATH above
        STRING(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
        set_tests_properties(${_name_} PROPERTIES
            ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:metatensor>"
        )
    endif()
endforeach()
