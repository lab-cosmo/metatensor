set(PACKAGE_VERSION "@PROJECT_VERSION@")

if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
  set(PACKAGE_VERSION_EXACT TRUE)
endif()

# Assume true until shown otherwise
set(PACKAGE_VERSION_COMPATIBLE TRUE)

if(PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
endif()

if(@PROJECT_VERSION_MAJOR@ EQUAL 0)
    if(NOT PACKAGE_FIND_VERSION_MINOR EQUAL "@PROJECT_VERSION_MINOR@")
        set(PACKAGE_VERSION_COMPATIBLE FALSE)
    endif()
else()
    if(NOT PACKAGE_FIND_VERSION_MAJOR EQUAL "@PROJECT_VERSION_MAJOR@")
        set(PACKAGE_VERSION_COMPATIBLE FALSE)
    endif()
endif()


# if the installed or the using project don't have CMAKE_SIZEOF_VOID_P set, ignore it:
if(NOT "${CMAKE_SIZEOF_VOID_P}" STREQUAL "" AND NOT "@CMAKE_SIZEOF_VOID_P@" STREQUAL "")
  # check that the installed version has the same 32/64bit-ness
  # as the one which is currently searching:
  if(NOT CMAKE_SIZEOF_VOID_P STREQUAL "@CMAKE_SIZEOF_VOID_P@")
    math(EXPR installedBits "@CMAKE_SIZEOF_VOID_P@ * 8")
    set(PACKAGE_VERSION "${PACKAGE_VERSION} (${installedBits}bit)")
    set(PACKAGE_VERSION_UNSUITABLE TRUE)
  endif()
endif()
