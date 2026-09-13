cmake_minimum_required(VERSION 3.15...3.26)

project(odecraft LANGUAGES CXX)

# Only when odecraft is the project being configured.
if((DEFINED PROJECT_IS_TOP_LEVEL AND PROJECT_IS_TOP_LEVEL) OR
   (NOT DEFINED PROJECT_IS_TOP_LEVEL AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR))
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

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
option(ODECRAFT_USE_FLAT_AUTODIFF "Use the flat dual layout for compile-time sized systems (dynamic ones stay nested)" OFF)
option(ODECRAFT_USE_LAZY_MPREAL "Compile ode::crafted's mpreal_t as lazy::LazyType<mpfr::mpreal> instead of plain mpfr::mpreal" ON)
# ON when odecraft is configured directly, OFF when another project pulls it in via
# add_subdirectory/FetchContent: a consumer that only wants the header-only `ode` interface
# should not pay to compile a library it never links. PROJECT_IS_TOP_LEVEL needs CMake >= 3.21.
if((DEFINED PROJECT_IS_TOP_LEVEL AND PROJECT_IS_TOP_LEVEL) OR
   (NOT DEFINED PROJECT_IS_TOP_LEVEL AND CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR))
    set(ODECRAFT_BUILD_CRAFTED_DEFAULT ON)
else()
    set(ODECRAFT_BUILD_CRAFTED_DEFAULT OFF)
endif()
option(ODECRAFT_BUILD_CRAFTED "Build the pre-compiled ode::crafted interface (src/*.cpp) alongside the header-only one" ${ODECRAFT_BUILD_CRAFTED_DEFAULT})
option(DEBUG "Enable debug build (-O0 -g3, asserts enabled) instead of the optimized release build" OFF)

target_compile_definitions(odecraft INTERFACE
    $<$<BOOL:${ODECRAFT_RK4_DENSE}>:ODECRAFT_RK4_DENSE>
    $<$<BOOL:${ODECRAFT_NO_WARN}>:ODECRAFT_NO_WARN>
    $<$<BOOL:${ODECRAFT_NO_NAN_CHECK}>:ODECRAFT_NO_NAN_CHECK>
    $<$<BOOL:${ODECRAFT_USE_FLAT_AUTODIFF}>:ODECRAFT_USE_FLAT_AUTODIFF>
    $<$<BOOL:${ODECRAFT_USE_LAZY_MPREAL}>:ODECRAFT_USE_LAZY_MPREAL>
)


set(ODECRAFT_DEBUG_BUILD OFF)
if(DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ODECRAFT_DEBUG_BUILD ON)
endif()

# ode::crafted is compiled for mpreal_t as well as the built-in floats, so it needs lazy::mpfr
# (mpreal.h plus the mpfr/gmp system libraries). LAZY_ENABLE_MPREAL is consumed while xdiff is
# configured below, hence setting it here.
if(ODECRAFT_BUILD_CRAFTED)
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

# Keep qhull out of compile_commands.json. It is vendored C code we never edit, and it
# outnumbers odecraft's own entries roughly 8:1 -- which matters because clangd infers the
# flags for a *header* by interpolating from the nearest database entry. Pick a qhull entry
# for something under include/ and you get -Iexternal/qhull/src with no -Iinclude and no
# C++20, so every odecraft header lights up with bogus "file not found" errors.
set(ODECRAFT_EXPORT_CC_SAVED ${CMAKE_EXPORT_COMPILE_COMMANDS})
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)

add_subdirectory(external/qhull EXCLUDE_FROM_ALL)

set(CMAKE_EXPORT_COMPILE_COMMANDS ${ODECRAFT_EXPORT_CC_SAVED})

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
#                     Pre-compiled interface (namespace ode::crafted)
#===========================================================================================
# The headers under include/odecraft/Compiled/ expose the library with every template
# parameter but the scalar type pinned down (dynamic system size, std::function callables),
# and each one has exactly one translation unit here that explicitly instantiates it for some arithmetic types. Consumers of ode::crafted therefore never instantiate a
# solver themselves; they link these objects instead.
#
# The list is in dependency order, each unit only instantiating what no earlier one already
# has. Every Compiled/ header carries matching `extern template` declarations, so a later
# unit -- and any consumer -- reuses those instantiations rather than repeating them.
#
# This is additive: the header-only interface in namespace `ode` is untouched, still comes
# from odecraft::odecraft alone, and needs none of this.

if(ODECRAFT_BUILD_CRAFTED)
    set(ODECRAFT_CRAFTED_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Toolkit.cpp        # set_mpreal_prec
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Interpolators.cpp  # dense output
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Events.cpp         # event detection
        ${CMAKE_CURRENT_SOURCE_DIR}/src/OdeHistory.cpp     # OdeResult / OdeSolution
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Euler.cpp          # one unit per stepper
        ${CMAKE_CURRENT_SOURCE_DIR}/src/RungeKutta.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/RK23_DOPRI.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/RK45_DOPRI.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/DOP853.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/BDF.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Steppers.cpp       # runtime factories over the steppers
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ODE.cpp            # high-level driver
        ${CMAKE_CURRENT_SOURCE_DIR}/src/NdInterpolators.cpp # regular / scattered field interpolation
        ${CMAKE_CURRENT_SOURCE_DIR}/src/RegularVectorField.cpp   # one unit per vector field: each
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ScatteredVectorField.cpp # compiles its own streamline solvers

        # Variational solvers. Split one unit per stepper for the same reason the plain
        # steppers are: each compiles its whole stepper again for the augmented 2N system
        # (a different OdeType, so nothing is shared with the units above), and as a single
        # unit that work dominated the build's wall-clock.
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Chaos_Euler.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Chaos_RungeKutta.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Chaos_RK23_DOPRI.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Chaos_RK45_DOPRI.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Chaos_DOP853.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Chaos_BDF.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/Chaos.cpp          # variational driver + factory
    )

    add_library(odecraft_crafted STATIC ${ODECRAFT_CRAFTED_SOURCES})
    add_library(odecraft::crafted ALIAS odecraft_crafted)

    # PUBLIC: anything linking odecraft::crafted also gets the headers and the header-only
    # interface, so <odecraft/Compiled/odecraft.hpp> is all a consumer needs to include.
    target_link_libraries(odecraft_crafted PUBLIC odecraft::odecraft lazy::mpfr)

    # Tools.hpp includes <omp.h> and the steppers use `#pragma omp simd`, so OpenMP is not
    # optional here; it is PUBLIC so consumers link the runtime too.
    target_compile_options(odecraft_crafted PRIVATE
        -Wall -Wextra -Wpedantic -Wshadow -Wno-sign-conversion -Wno-unused-parameter
    )
    target_compile_options(odecraft_crafted PUBLIC -fopenmp)
    target_link_options(odecraft_crafted PUBLIC -fopenmp)

    if(ODECRAFT_DEBUG_BUILD)
        target_compile_options(odecraft_crafted PRIVATE -O0 -g3 -UNDEBUG)
    else()
        target_compile_options(odecraft_crafted PRIVATE -O3 -DNDEBUG -g0 -fno-math-errno -fno-trapping-math)
    endif()
endif()
