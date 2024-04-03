include(CMakeFindDependencyMacro)

# use the same version for metatensor-core as the main CMakeLists.txt
set(REQUIRED_METATENSOR_VERSION @REQUIRED_METATENSOR_VERSION@)
find_package(metatensor ${REQUIRED_METATENSOR_VERSION} CONFIG REQUIRED)

# We can only load metatensor_torch with the same minor version of Torch that
# was used to compile it (and is stored in BUILD_TORCH_VERSION)
set(BUILD_TORCH_VERSION @Torch_VERSION@)
set(BUILD_TORCH_MAJOR @Torch_VERSION_MAJOR@)
set(BUILD_TORCH_MINOR @Torch_VERSION_MINOR@)

find_package(Torch ${BUILD_TORCH_VERSION} REQUIRED)

if (NOT "${BUILD_TORCH_MAJOR}" STREQUAL "${Torch_VERSION_MAJOR}")
    message(FATAL_ERROR "found incompatible torch version: metatensor-torch was built against v${BUILD_TORCH_VERSION} but we found v${Torch_VERSION}")
endif()

if (NOT "${BUILD_TORCH_MINOR}" STREQUAL "${Torch_VERSION_MINOR}")
    message(FATAL_ERROR "found incompatible torch version: metatensor-torch was built against v${BUILD_TORCH_VERSION} but we found v${Torch_VERSION}")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/metatensor_torch-targets.cmake)
