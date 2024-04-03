cmake_minimum_required(VERSION 3.16)

message(STATUS "Running with CMake version ${CMAKE_VERSION}")

project(metatensor-torch-test-cmake-project CXX)

option(USE_CMAKE_SUBDIRECTORY OFF)

if (USE_CMAKE_SUBDIRECTORY)
    message(STATUS "Using metatensor-torch with add_subdirectory")
    set(BUILD_METATENSOR_TORCH ON)
    # build metatensor_torch as part of this project
    add_subdirectory(../../../ metatensor)
    find_package(Torch)

    # load metatensor and metatensor-torch from the build path
    set(CMAKE_BUILD_RPATH "$<TARGET_FILE_DIR:torch>;$<TARGET_FILE_DIR:metatensor::shared>;$<TARGET_FILE_DIR:metatensor_torch>")
else()
    message(STATUS "Using metatensor-torch with find_package")
    # If building a dev version, we also need to update the REQUIRED_METATENSOR_VERSION
    # in the same way we update the metatensor-torch version
    include(../../cmake/dev-versions.cmake)
    set(REQUIRED_METATENSOR_TORCH_VERSION "0.3.0")
    create_development_version("${REQUIRED_METATENSOR_TORCH_VERSION}" "metatensor-torch-v" METATENSOR_TORCH_FULL_VERSION)
    string(REGEX REPLACE "([0-9]*)\\.([0-9]*).*" "\\1.\\2" REQUIRED_METATENSOR_TORCH_VERSION ${METATENSOR_TORCH_FULL_VERSION})

    # find metatensor_torch with find_package
    find_package(metatensor_torch ${REQUIRED_METATENSOR_TORCH_VERSION} CONFIG REQUIRED)
endif()

add_executable(torch-main src/main.cpp)
target_link_libraries(torch-main metatensor_torch)

enable_testing()
add_test(NAME torch-main COMMAND torch-main)

if(WIN32)
    # We need to set the path to allow access to metatensor.dll
    STRING(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
    set_tests_properties(torch-main PROPERTIES
        ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:torch>\;$<TARGET_FILE_DIR:metatensor::shared>\;$<TARGET_FILE_DIR:metatensor_torch>"
    )
endif()
