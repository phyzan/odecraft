cmake_minimum_required(VERSION 3.15...3.26)

project(odecraft LANGUAGES CXX)

# ----------- Main odecraft interface target ---------------

add_library(odecraft INTERFACE)
add_library(odecraft::odecraft ALIAS odecraft)

target_compile_features(odecraft INTERFACE cxx_std_20)

target_include_directories(odecraft INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ----------- Add interface for macros ---------------

option(ODECRAFT_RK4_DENSE "Enable accurate RK4 dense output for the RK4 solver (extra memory, slightly slower performance)" OFF)
option(ODECRAFT_NO_WARN "Disable ODE solver console warnings" OFF)
option(ODECRAFT_NO_NAN_CHECK "Disable NaN/inf checks on solver output for performance" OFF)
option(DEBUG "Enable debug build (-O0 -g3, asserts enabled) instead of the optimized release build" OFF)

target_compile_definitions(odecraft INTERFACE
    $<$<BOOL:${ODECRAFT_RK4_DENSE}>:ODECRAFT_RK4_DENSE>
    $<$<BOOL:${ODECRAFT_NO_WARN}>:ODECRAFT_NO_WARN>
    $<$<BOOL:${ODECRAFT_NO_NAN_CHECK}>:ODECRAFT_NO_NAN_CHECK>
)

if((DEFINED PROJECT_IS_TOP_LEVEL AND PROJECT_IS_TOP_LEVEL) OR
   (NOT DEFINED PROJECT_IS_TOP_LEVEL AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR))
    set(ODECRAFT_BUILD_TESTS_DEFAULT ON)
else()
    set(ODECRAFT_BUILD_TESTS_DEFAULT OFF)
endif()
option(ODECRAFT_BUILD_TESTS "Build the odecraft_tests executable from tests/src/*.cpp" ${ODECRAFT_BUILD_TESTS_DEFAULT})

if(ODECRAFT_BUILD_TESTS)
    set(LAZY_ENABLE_MPREAL ON CACHE BOOL "" FORCE)
endif()

#===========================================================================================
#                                           xdiff
#===========================================================================================
add_subdirectory(external/xdiff)

#===========================================================================================
#                           ndspan (header-only, no CMakeLists of its own)
#===========================================================================================
add_library(odecraft_ndspan INTERFACE)
add_library(odecraft::ndspan ALIAS odecraft_ndspan)
target_include_directories(odecraft_ndspan INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/ndspan/include
)

#===========================================================================================
#                           polybox (header-only, no CMakeLists of its own)
#===========================================================================================
add_library(odecraft_polybox INTERFACE)
add_library(odecraft::polybox ALIAS odecraft_polybox)
target_include_directories(odecraft_polybox INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/polybox/include
)

#===========================================================================================
#                           qhull (static library)
#===========================================================================================
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_APPLICATIONS OFF CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(external/qhull EXCLUDE_FROM_ALL)

add_library(odecraft_qhull INTERFACE)
add_library(odecraft::qhull ALIAS odecraft_qhull)
target_include_directories(odecraft_qhull INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/external/qhull/src
)
target_link_libraries(odecraft_qhull INTERFACE qhullstatic_r)

# ----------- Link everything into the umbrella target ---------------

target_link_libraries(odecraft INTERFACE
    xdiff
    odecraft::ndspan
    odecraft::polybox
    odecraft::qhull
)

#===========================================================================================
#                           Tests (combined into a single executable)
#===========================================================================================
# Each tests/src/*.cpp (besides main.cpp) implements a void test_<name>() entry point declared
# in the matching tests/include/<name>.hpp, and tests/src/main.cpp includes those headers and
# calls each in turn. All of tests/src/*.cpp are compiled and linked into one odecraft_tests
# executable.
#
# Only built when odecraft is configured directly (not when pulled in via add_subdirectory by
# another project), so consumers who only want the interface never build or even see this
# target.

if(ODECRAFT_BUILD_TESTS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    file(GLOB ODECRAFT_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/src/*.cpp)

    add_executable(odecraft_tests ${ODECRAFT_TEST_SOURCES})
    target_include_directories(odecraft_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/include)
    target_link_libraries(odecraft_tests PRIVATE odecraft::odecraft lazy::mpfr)

    set(ODECRAFT_TEST_COMPILE_OPTIONS
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wno-sign-conversion
        -fopenmp
    )
    set(ODECRAFT_TEST_LINK_OPTIONS -fopenmp)

    set(ODECRAFT_DEBUG_BUILD OFF)
    if(DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(ODECRAFT_DEBUG_BUILD ON)
    endif()

    if(ODECRAFT_DEBUG_BUILD)
        list(APPEND ODECRAFT_TEST_COMPILE_OPTIONS -O0 -g3 -ggdb3 -fsanitize=address -fno-omit-frame-pointer -UNDEBUG -Wconversion)
        message(STATUS "ODECRAFT_DEBUG_BUILD=ON (asserts enabled)")
        list(APPEND ODECRAFT_TEST_LINK_OPTIONS -fsanitize=address)
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
        message(STATUS "ODECRAFT_DEBUG_BUILD=OFF (asserts disabled via NDEBUG)")
    endif()

    target_compile_options(odecraft_tests PRIVATE ${ODECRAFT_TEST_COMPILE_OPTIONS})
    target_link_options(odecraft_tests PRIVATE ${ODECRAFT_TEST_LINK_OPTIONS})
endif()
