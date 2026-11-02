cmake_minimum_required(VERSION 3.24)
project(odecraft LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

option(ODECRAFT_RK4_DENSE "Enable accurate RK4 dense output for the RK4 solver (extra memory, slightly slower performance)" OFF)
option(ODECRAFT_NO_WARN "Disable ODE solver console warnings" OFF)
option(ODECRAFT_NO_NAN_CHECK "Disable NaN/inf checks on solver output for performance" OFF)
option(ODECRAFT_USE_FLAT_AUTODIFF "Use the flat dual layout for compile-time sized systems (dynamic ones stay nested)" OFF)
option(ODECRAFT_USE_LAZY_MPREAL "Compile ode::crafted's mpreal_t as lazex::LazyType<mpfr::mpreal> instead of plain mpfr::mpreal" ON)
option(ODECRAFT_RHS_THROWS "Allow Rhs and the other user callbacks to throw (otherwise the solvers call them noexcept)" OFF)
option(ODECRAFT_BUILD_CRAFTED "Build the pre-compiled ode::crafted library (ON by default only when odecraft is the top-level project)" ${PROJECT_IS_TOP_LEVEL})
option(DEBUG "Enable debug build (-O0 -g3, asserts enabled) instead of the optimized release build" OFF)

add_subdirectory(external/xdiff)

# ndspan and polybox are header-only without a CMakeLists of their own
add_library(odecraft_ndspan INTERFACE)
add_library(odecraft::ndspan ALIAS odecraft_ndspan)
target_include_directories(odecraft_ndspan INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/ndspan/include)

add_library(odecraft_polybox INTERFACE)
add_library(odecraft::polybox ALIAS odecraft_polybox)
target_include_directories(odecraft_polybox INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/polybox/include)

# qhull (reentrant C library, e.g. libqhull-dev) backs the Delaunay triangulation that <odecraft/odecraft.hpp> includes
find_path(ODECRAFT_QHULL_INCLUDE_DIR libqhull_r/qhull_ra.h REQUIRED)
find_library(ODECRAFT_QHULL_LIBRARY qhull_r REQUIRED)

add_library(odecraft_qhull INTERFACE)
add_library(odecraft::qhull ALIAS odecraft_qhull)
target_include_directories(odecraft_qhull INTERFACE ${ODECRAFT_QHULL_INCLUDE_DIR})
target_link_libraries(odecraft_qhull INTERFACE ${ODECRAFT_QHULL_LIBRARY})

# ----------- Header-only interface (namespace ode) ---------------

add_library(odecraft INTERFACE)
add_library(odecraft::odecraft ALIAS odecraft)
target_include_directories(odecraft INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(odecraft SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/external/mpreal)  # vendored mpreal.h
target_compile_features(odecraft INTERFACE cxx_std_20)
target_link_libraries(odecraft INTERFACE xdiff odecraft::ndspan odecraft::polybox odecraft::qhull)
target_compile_definitions(odecraft INTERFACE
    $<$<BOOL:${ODECRAFT_RK4_DENSE}>:ODECRAFT_RK4_DENSE>
    $<$<BOOL:${ODECRAFT_NO_WARN}>:ODECRAFT_NO_WARN>
    $<$<BOOL:${ODECRAFT_NO_NAN_CHECK}>:ODECRAFT_NO_NAN_CHECK>
    $<$<BOOL:${ODECRAFT_USE_FLAT_AUTODIFF}>:ODECRAFT_USE_FLAT_AUTODIFF>
    $<$<BOOL:${ODECRAFT_USE_LAZY_MPREAL}>:ODECRAFT_USE_LAZY_MPREAL>
    $<$<BOOL:${ODECRAFT_RHS_THROWS}>:ODECRAFT_RHS_THROWS>
)

# Building odecraft itself without ode::crafted: there are no sources to export, so a target
# compiling each header on its own (not built by default) gives compile_commands.json its entries.
if(PROJECT_IS_TOP_LEVEL AND NOT ODECRAFT_BUILD_CRAFTED)
    file(GLOB_RECURSE ODECRAFT_HEADERS CONFIGURE_DEPENDS include/*.hpp)
    target_sources(odecraft INTERFACE FILE_SET HEADERS BASE_DIRS include FILES ${ODECRAFT_HEADERS})
    set_target_properties(odecraft PROPERTIES VERIFY_INTERFACE_HEADER_SETS ON)
endif()

# ----------- Pre-compiled interface (namespace ode::crafted) ---------------
# Each unit explicitly instantiates the Compiled/ headers for the supported scalar types, in
# dependency order; the headers' `extern template` declarations let later units and consumers
# reuse those instantiations.

if(ODECRAFT_BUILD_CRAFTED)
    # The objects are built with -flto, and an archive of LTO objects is only indexed when made
    # through the compiler's wrapper (gcc-ar), which loads the matching LTO plugin.
    if(CMAKE_CXX_COMPILER_AR AND CMAKE_CXX_COMPILER_RANLIB)
        set(CMAKE_AR ${CMAKE_CXX_COMPILER_AR})
        set(CMAKE_RANLIB ${CMAKE_CXX_COMPILER_RANLIB})
    endif()

    add_library(odecraft_crafted STATIC
        src/Toolkit.cpp                 # set_mpreal_prec
        src/Interpolators.cpp           # dense output
        src/Events.cpp                  # event detection
        src/OdeHistory.cpp              # OdeResult / OdeSolution
        src/Euler.cpp                   # one unit per stepper
        src/RK4.cpp
        src/RK23_DOPRI.cpp
        src/RK45_DOPRI.cpp
        src/DOP853.cpp
        src/BDF.cpp
        src/Steppers.cpp                # runtime factories over the steppers
        src/ODE.cpp                     # high-level driver
        src/NdInterpolators.cpp         # regular / scattered field interpolation
        src/RegularVectorField.cpp      # one unit per vector field
        src/ScatteredVectorField.cpp
        src/Chaos_Euler.cpp             # variational solvers, one unit per stepper
        src/Chaos_RK4.cpp
        src/Chaos_RK23_DOPRI.cpp
        src/Chaos_RK45_DOPRI.cpp
        src/Chaos_DOP853.cpp
        src/Chaos_BDF.cpp
        src/Chaos.cpp                   # variational driver + factory
    )
    add_library(odecraft::crafted ALIAS odecraft_crafted)

    # Linked into shared objects such as Python extension modules
    set_target_properties(odecraft_crafted PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Also compiled for mpfr::mpreal, so it links the MPFR and GMP system libraries.
    find_library(MPFR_LIBRARY mpfr REQUIRED)
    find_library(GMP_LIBRARY gmp REQUIRED)

    # PUBLIC: <odecraft/Compiled/odecraft.hpp> is all a consumer needs to include
    target_link_libraries(odecraft_crafted PUBLIC odecraft::odecraft ${MPFR_LIBRARY} ${GMP_LIBRARY})

    # -fopenmp-simd enables `#pragma omp simd` without the OpenMP runtime; PUBLIC so that consumers
    # compiling the headers don't warn about an unknown pragma.
    target_compile_options(odecraft_crafted
        PUBLIC -fopenmp-simd
        PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Wno-sign-conversion -Wno-unused-parameter
    )

    if(DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(odecraft_crafted PRIVATE -O0 -g3 -UNDEBUG)
    else()
        target_compile_options(odecraft_crafted PRIVATE -O3 -flto -funroll-loops -DNDEBUG -g0 -fno-math-errno -fno-trapping-math)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(odecraft_crafted PRIVATE -finline-functions -finline-limit=10000)
        endif()
    endif()
endif()
