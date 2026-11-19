#===========================================================================================
#                                   odecraft test suite
#===========================================================================================
#
# Part of odecraft's own build, from the repository root, in a build directory of its own with the
# library's default options:
#
#     cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Release -DODECRAFT_BUILD_TESTS=ON
#     cmake --build build-tests -j 4
#     ctest --test-dir build-tests --output-on-failure -j 4
#
# Two executables, registered test case by test case with CTest:
#   odecraft_tests          the header-only interface, namespace ode
#   odecraft_crafted_tests  the pre-compiled interface, namespace ode::crafted. It includes only
#                           the Crafted headers, so it also proves that the library ships every
#                           instantiation its interface promises: a missing one fails to link.
#
# Options:
#   ODECRAFT_TESTS_SANITIZE  AddressSanitizer + UndefinedBehaviorSanitizer
#   ODECRAFT_TESTS_WERROR    warnings are errors

# 3.25 for FetchContent's SYSTEM, below
cmake_minimum_required(VERSION 3.25)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "The tests build as part of odecraft, from the repository root: "
                        "cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Release -DODECRAFT_BUILD_TESTS=ON")
endif()

# The tests check that the solvers detect non-finite states, which this option compiles out
if(ODECRAFT_NO_NAN_CHECK)
    message(FATAL_ERROR "odecraft tests: ODECRAFT_NO_NAN_CHECK is ON, which removes the checks the tests "
                        "verify. Build them in a directory of their own with the default options: "
                        "cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Release -DODECRAFT_BUILD_TESTS=ON")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(WARNING "odecraft tests: no CMAKE_BUILD_TYPE, so they compile unoptimized and run slowly. "
                    "Configure with -DCMAKE_BUILD_TYPE=Release.")
endif()

option(ODECRAFT_TESTS_SANITIZE "Build the tests with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(ODECRAFT_TESTS_WERROR "Treat compiler warnings in the tests as errors" OFF)

#-------------------------------------------------------------------------------------------
# Catch2: an installed one if there is one, otherwise this exact release.

include(FetchContent)
FetchContent_Declare(Catch2
    URL      https://github.com/catchorg/Catch2/archive/refs/tags/v3.16.1.tar.gz
    URL_HASH SHA256=d16bd1b8b2364918bd472cb216394acb93bde6784890df0d91bccfb3f28d777d
    SYSTEM
    FIND_PACKAGE_ARGS 3
)
FetchContent_MakeAvailable(Catch2)

# Catch.cmake ships in extras/ of the sources, and next to the package config once installed
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras" "${Catch2_DIR}")
include(Catch)

# The multiprecision tests use mpfr::mpreal, which needs MPFR and GMP
find_library(MPFR_LIBRARY mpfr REQUIRED)
find_library(GMP_LIBRARY gmp REQUIRED)

#-------------------------------------------------------------------------------------------
# What every test executable shares: the helpers in support/, Catch2's main and the flags.

add_library(odecraft_test_common INTERFACE)
target_include_directories(odecraft_test_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/support)
target_link_libraries(odecraft_test_common INTERFACE Catch2::Catch2WithMain ${MPFR_LIBRARY} ${GMP_LIBRARY})
target_compile_features(odecraft_test_common INTERFACE cxx_std_20)
target_compile_options(odecraft_test_common INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wno-unused-parameter
    -fopenmp-simd
    $<$<BOOL:${ODECRAFT_TESTS_WERROR}>:-Werror>
)

if(ODECRAFT_TESTS_SANITIZE)
    target_compile_options(odecraft_test_common INTERFACE -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
    target_link_options(odecraft_test_common INTERFACE -fsanitize=address,undefined)
endif()

# Each test case becomes a CTest test of its own, listed when ctest runs (not at build time)
function(odecraft_register_tests target prefix)
    catch_discover_tests(${target}
        TEST_PREFIX "${prefix}: "
        DISCOVERY_MODE PRE_TEST
        PROPERTIES LABELS "${prefix}" TIMEOUT 900
    )
endfunction()

#-------------------------------------------------------------------------------------------
# The header-only interface.
#
# Its tests run over five scalar types, and instantiating every stepper for all five in one
# translation unit takes several GB of memory. So the sources whose test cases all run over
# ODECRAFT_TEST_SCALARS are compiled once per scalar type, each time with
# ODECRAFT_TEST_SCALAR_INDEX set to that type: five small translation units per source, built in
# parallel, registering the same test cases between them. Test cases that run over other types,
# or none, go in the sources compiled once.

set(ODECRAFT_TESTS_PER_SCALAR
    odecraft/test_steppers.cpp
    odecraft/test_state_descriptors.cpp
    odecraft/test_solver_lifecycle.cpp
    odecraft/test_solver_failure.cpp
    odecraft/test_jacobians.cpp
    odecraft/test_runtime_events.cpp
    odecraft/test_static_events.cpp
    odecraft/test_dense_output.cpp
    odecraft/test_factories.cpp
    odecraft/test_ode_driver.cpp
    odecraft/test_chaos.cpp
    odecraft/test_chaos_jacobians.cpp
)
set(ODECRAFT_TESTS_ONCE
    support/precision_listener.cpp
    odecraft/test_toolkit.cpp
    odecraft/test_univariate_interpolation.cpp
    odecraft/test_precision.cpp
    odecraft/test_catalogs.cpp
    odecraft/test_nd_interpolation.cpp
    odecraft/test_vector_fields.cpp
)

add_executable(odecraft_tests ${ODECRAFT_TESTS_ONCE})
target_link_libraries(odecraft_tests PRIVATE odecraft::odecraft odecraft_test_common)
# Assertions stay on whatever the build type: they check the library's own invariants.
target_compile_options(odecraft_tests PRIVATE -UNDEBUG)

# In the order of ODECRAFT_TEST_SCALARS in support/scalars.hpp
set(ODECRAFT_TEST_SCALAR_NAMES float double long_double mpreal lazy_mpreal)
foreach(index RANGE 4)
    list(GET ODECRAFT_TEST_SCALAR_NAMES ${index} name)
    add_library(odecraft_tests_${name} OBJECT ${ODECRAFT_TESTS_PER_SCALAR})
    target_compile_definitions(odecraft_tests_${name} PRIVATE ODECRAFT_TEST_SCALAR_INDEX=${index})
    target_compile_options(odecraft_tests_${name} PRIVATE -UNDEBUG)
    target_link_libraries(odecraft_tests_${name} PRIVATE odecraft::odecraft odecraft_test_common)
    target_link_libraries(odecraft_tests PRIVATE odecraft_tests_${name})
endforeach()

odecraft_register_tests(odecraft_tests odecraft)

#-------------------------------------------------------------------------------------------
# The pre-compiled interface, built against the same NDEBUG setting as the library it links.

if(TARGET odecraft::crafted)
    add_executable(odecraft_crafted_tests
        crafted/precision_listener.cpp
        crafted/test_crafted_vocabulary.cpp
        crafted/test_crafted_solvers.cpp
        crafted/test_crafted_events.cpp
        crafted/test_crafted_drivers.cpp
        crafted/test_crafted_chaos.cpp
        crafted/test_crafted_nd_interpolation.cpp
    )
    target_link_libraries(odecraft_crafted_tests PRIVATE odecraft::crafted odecraft_test_common)
    odecraft_register_tests(odecraft_crafted_tests crafted)
else()
    message(STATUS "odecraft tests: ode::crafted is not built (ODECRAFT_BUILD_CRAFTED=OFF), skipping its tests")
endif()
