cmake_minimum_required(VERSION 3.24) # Dependency providers introduced in CMake 3.24

option(HOMEBREW_ALLOW_FETCHCONTENT "Allow FetchContent to be used in Homebrew builds" OFF)

if (HOMEBREW_ALLOW_FETCHCONTENT)
    return()
endif()

macro(trap_fetchcontent_provider method depName)
    message(FATAL_ERROR "Refusing to populate dependency '${depName}' with FetchContent while building in Homebrew, please use a formula dependency or add a resource to the formula.")
endmacro()

cmake_language(
    SET_DEPENDENCY_PROVIDER trap_fetchcontent_provider
    SUPPORTED_METHODS FETCHCONTENT_MAKEAVAILABLE_SERIAL
)
