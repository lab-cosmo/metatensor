cmake_minimum_required(VERSION 3.16)

project(metatensor-test-cmake-project C CXX)

find_package(metatensor 0.1 REQUIRED)

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