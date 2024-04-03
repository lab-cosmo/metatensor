# re-use catch from metatensor-core C++ tests
add_subdirectory(../../metatensor-core/tests/cpp/external external)

# make sure we compile catch with the flags that torch requires. In particular,
# torch sets -D_GLIBCXX_USE_CXX11_ABI=0 on Linux, which changes some of the
# symbols in catch
target_link_libraries(catch torch)

find_program(VALGRIND valgrind)
if (VALGRIND)
    if (NOT "$ENV{METATENSOR_DISABLE_VALGRIND}" EQUAL "1")
        message(STATUS "Running tests using valgrind")
        set(TEST_COMMAND
            "${VALGRIND}" "--tool=memcheck" "--dsymutil=yes" "--error-exitcode=125"
            "--leak-check=full" "--show-leak-kinds=definite,indirect,possible" "--track-origins=yes"
            "--gen-suppressions=all" "--suppressions=${CMAKE_CURRENT_SOURCE_DIR}/valgrind.supp"
        )
    endif()
else()
    set(TEST_COMMAND "")
endif()


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++98-compat")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-weak-vtables")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-float-equal")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-missing-prototypes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow-uncaptured-local")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-padded")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-exit-time-destructors")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-global-constructors")
endif()

file(GLOB ALL_TESTS *.cpp)
foreach(_file_ ${ALL_TESTS})
    get_filename_component(_name_ ${_file_} NAME_WE)
    add_executable(torch-${_name_} ${_file_})
    target_link_libraries(torch-${_name_} metatensor_torch catch)
    target_compile_definitions(torch-${_name_} PRIVATE "-DDATA_NPZ=\"${CMAKE_CURRENT_SOURCE_DIR}/../../metatensor-core/tests/data.npz\"")

    add_test(
        NAME torch-${_name_}
        COMMAND ${TEST_COMMAND} $<TARGET_FILE:torch-${_name_}>
    )

    if(WIN32)
        # We need to set the path to allow access to torch.dll
        # (and any other DLL)
        STRING(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
        set_tests_properties(torch-${_name_} PROPERTIES
            ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:torch>\;$<TARGET_FILE_DIR:metatensor_torch>\;$<TARGET_FILE_DIR:metatensor::shared>"
        )
    endif()
endforeach()
