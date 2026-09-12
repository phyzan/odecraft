#===========================================================================================
#                           odecraft tests (standalone project)
#===========================================================================================
# Configure and build from this directory:
#
#     cmake -S . -B build && cmake --build build -j
#     ./build/odecraft_tests
#
# Add -DDEBUG=ON for an -O0 build with asserts and AddressSanitizer.
#
# Each src/*.cpp (besides main.cpp) implements a void test_<name>() entry point declared in
# the matching include/<name>.hpp, and src/main.cpp includes those headers and calls each in
# turn. All of src/*.cpp are compiled and linked into one odecraft_tests executable.

cmake_minimum_required(VERSION 3.15...3.26)

project(odecraft_tests LANGUAGES CXX)

# clangd and most IDEs read build/compile_commands.json. See the note in the top-level
# CMakeLists for why this tests the empty value rather than definedness.
if(CMAKE_EXPORT_COMPILE_COMMANDS STREQUAL "")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# lazy_scalar.cpp runs the solvers on mpfr::mpreal, so it needs lazy::mpfr, which only exists
# when lazy is configured with this option. lazy is configured from inside odecraft's own
# CMakeLists, so this has to be set before pulling that in below.
set(LAZY_ENABLE_MPREAL ON CACHE BOOL "" FORCE)

# odecraft is a subproject here, where the crafted library defaults to OFF. The tests cover it,
# so ask for it explicitly.
set(ODECRAFT_BUILD_CRAFTED ON CACHE BOOL "" FORCE)

# The library itself lives one level up. Building it here keeps this project self-contained:
# there is nothing to install or configure first.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/odecraft)

#-------------------------------------------------------------------------------------------

file(GLOB ODECRAFT_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

# compiled_interface.cpp exercises ode::crafted and has nothing to link against without it.
if(NOT TARGET odecraft::crafted)
    list(REMOVE_ITEM ODECRAFT_TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/compiled_interface.cpp)
endif()

add_executable(odecraft_tests ${ODECRAFT_TEST_SOURCES})

target_include_directories(odecraft_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(odecraft_tests PRIVATE odecraft::odecraft lazy::mpfr)

if(TARGET odecraft::crafted)
    target_link_libraries(odecraft_tests PRIVATE odecraft::crafted)
    target_compile_definitions(odecraft_tests PRIVATE ODECRAFT_HAS_CRAFTED)
endif()

set(ODECRAFT_TEST_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wno-sign-conversion
    -fopenmp
)
set(ODECRAFT_TEST_LINK_OPTIONS -fopenmp)

# -finline-limit is a GCC-only spelling; clang and MSVC reject/ignore it.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND ODECRAFT_TEST_COMPILE_OPTIONS -finline-limit=10000)
endif()

# DEBUG is the option odecraft itself defines; honour it here too so one flag covers both.
set(ODECRAFT_TEST_DEBUG_BUILD OFF)
if(DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ODECRAFT_TEST_DEBUG_BUILD ON)
endif()

if(ODECRAFT_TEST_DEBUG_BUILD)
    list(APPEND ODECRAFT_TEST_COMPILE_OPTIONS -O0 -g3 -ggdb3 -fsanitize=address -fno-omit-frame-pointer -UNDEBUG -Wconversion)
    list(APPEND ODECRAFT_TEST_LINK_OPTIONS -fsanitize=address)
    message(STATUS "odecraft_tests: debug build (asserts enabled, ASan on)")
else()
    list(APPEND ODECRAFT_TEST_COMPILE_OPTIONS -Wno-unused-parameter -O3 -DNDEBUG -g0 -fno-math-errno -fno-trapping-math -ffunction-sections -fdata-sections)
    list(APPEND ODECRAFT_TEST_LINK_OPTIONS -Wl,--gc-sections -Wl,--as-needed)

    include(CheckIPOSupported)
    check_ipo_supported(RESULT ODECRAFT_IPO_SUPPORTED OUTPUT ODECRAFT_IPO_ERROR LANGUAGES CXX)
    if(ODECRAFT_IPO_SUPPORTED)
        set_target_properties(odecraft_tests PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(STATUS "IPO/LTO not enabled: ${ODECRAFT_IPO_ERROR}")
    endif()
    message(STATUS "odecraft_tests: release build (asserts disabled via NDEBUG)")
endif()

target_compile_options(odecraft_tests PRIVATE ${ODECRAFT_TEST_COMPILE_OPTIONS})
target_link_options(odecraft_tests PRIVATE ${ODECRAFT_TEST_LINK_OPTIONS})
