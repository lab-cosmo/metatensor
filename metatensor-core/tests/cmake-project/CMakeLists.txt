cmake_minimum_required(VERSION 3.16)

message(STATUS "Running with CMake version ${CMAKE_VERSION}")

project(metatensor-test-cmake-project C CXX)

option(USE_CMAKE_SUBDIRECTORY OFF)

if (USE_CMAKE_SUBDIRECTORY)
    message(STATUS "Using metatensor with add_subdirectory")
    # build metatensor as part of this project
    add_subdirectory(../../../ metatensor)

    # load metatensor from the build path
    set(CMAKE_BUILD_RPATH "$<TARGET_FILE_DIR:metatensor::shared>")
else()
    message(STATUS "Using metatensor with find_package")
    # If building a dev version, we also need to update the REQUIRED_METATENSOR_VERSION
    # in the same way we update the metatensor-torch version
    include(../../cmake/dev-versions.cmake)
    set(REQUIRED_METATENSOR_VERSION "0.1.0")
    create_development_version("${REQUIRED_METATENSOR_VERSION}" "metatensor-core-v" METATENSOR_CORE_FULL_VERSION)
    string(REGEX REPLACE "([0-9]*)\\.([0-9]*).*" "\\1.\\2" REQUIRED_METATENSOR_VERSION ${METATENSOR_CORE_FULL_VERSION})

    find_package(metatensor ${REQUIRED_METATENSOR_VERSION} REQUIRED)
endif()

add_executable(c-main src/main.c)
target_link_libraries(c-main metatensor::shared)

add_executable(c-main-static src/main.c)
target_link_libraries(c-main-static metatensor::static)

add_executable(cxx-main src/main.cpp)
target_link_libraries(cxx-main metatensor::shared)

add_executable(cxx-main-static src/main.cpp)
target_link_libraries(cxx-main-static metatensor::static)

enable_testing()
add_test(NAME c-main COMMAND c-main)
add_test(NAME c-main-static COMMAND c-main-static)
add_test(NAME cxx-main COMMAND cxx-main)
add_test(NAME cxx-main-static COMMAND cxx-main-static)

if(WIN32)
    # We need to set the path to allow access to metatensor.dll
    STRING(REPLACE ";" "\\;" PATH_STRING "$ENV{PATH}")
    set_tests_properties(c-main PROPERTIES
        ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:metatensor::shared>"
    )

    set_tests_properties(cxx-main PROPERTIES
        ENVIRONMENT "PATH=${PATH_STRING}\;$<TARGET_FILE_DIR:metatensor::shared>"
    )
endif()
