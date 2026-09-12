#===========================================================================================
#                        odecraft tutorials (standalone project)
#===========================================================================================
# Configure and build from this directory:
#
#     cmake -S . -B build && cmake --build build -j
#     ./build/Autodiff
#
# Or build just one: cmake --build build --target Autodiff
#
# Every *.cpp here is a self-contained example with its own main(), and each becomes an
# executable of the same name. They demonstrate the header-only interface in namespace ode,
# so they link odecraft::odecraft rather than the compiled ode::crafted library.

cmake_minimum_required(VERSION 3.15...3.26)

project(odecraft_tutorials LANGUAGES CXX)

# clangd and most IDEs read build/compile_commands.json. See the note in the top-level
# CMakeLists for why this tests the empty value rather than definedness.
if(CMAKE_EXPORT_COMPILE_COMMANDS STREQUAL "")
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# ArbitraryPrecision.cpp runs the solvers on mpfr::mpreal, so it needs lazy::mpfr, which only
# exists when lazy is configured with this option. lazy is configured from inside odecraft's
# own CMakeLists, so this has to be set before pulling that in below.
set(LAZY_ENABLE_MPREAL ON CACHE BOOL "" FORCE)

file(GLOB ODECRAFT_TUTORIAL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Most tutorials use only the header-only `ode` interface, so the compiled library is not built
# unless one of them actually includes <odecraft/Compiled/...>. Detected here rather than listed
# by hand, so dropping in a new ode::crafted tutorial needs no edit to this file -- and it has to
# happen before odecraft is added below, since that is when the option is read.
set(ODECRAFT_TUTORIALS_NEED_CRAFTED OFF)
foreach(tutorial_src IN LISTS ODECRAFT_TUTORIAL_SOURCES)
    file(STRINGS ${tutorial_src} ODECRAFT_CRAFTED_HIT REGEX "odecraft/Compiled/")
    if(ODECRAFT_CRAFTED_HIT)
        set(ODECRAFT_TUTORIALS_NEED_CRAFTED ON)
    endif()
endforeach()
set(ODECRAFT_BUILD_CRAFTED ${ODECRAFT_TUTORIALS_NEED_CRAFTED} CACHE BOOL "" FORCE)

# The library itself lives one level up. Building it here keeps this project self-contained.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/odecraft)

#-------------------------------------------------------------------------------------------

set(ODECRAFT_TUTORIAL_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wno-sign-conversion
    -Wno-unused-parameter
    -fopenmp
)
set(ODECRAFT_TUTORIAL_LINK_OPTIONS -fopenmp)

# DEBUG is the option odecraft itself defines; honour it here too so one flag covers both.
if(DEBUG OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND ODECRAFT_TUTORIAL_COMPILE_OPTIONS -O0 -g3 -UNDEBUG)
else()
    list(APPEND ODECRAFT_TUTORIAL_COMPILE_OPTIONS -O3 -DNDEBUG -g0)
endif()

foreach(tutorial_src IN LISTS ODECRAFT_TUTORIAL_SOURCES)
    get_filename_component(tutorial_name ${tutorial_src} NAME_WE)

    add_executable(${tutorial_name} ${tutorial_src})
    target_link_libraries(${tutorial_name} PRIVATE odecraft::odecraft)

    # ode::crafted tutorials additionally link the compiled library. Linking through CMake is
    # also what keeps ODECRAFT_USE_LAZY_MPREAL consistent between the library and the tutorial:
    # mpreal_t is a different type on each setting, so a mismatch is an undefined reference.
    file(STRINGS ${tutorial_src} ODECRAFT_CRAFTED_HIT REGEX "odecraft/Compiled/")
    if(ODECRAFT_CRAFTED_HIT AND TARGET odecraft::crafted)
        target_link_libraries(${tutorial_name} PRIVATE odecraft::crafted)
    endif()
    target_compile_options(${tutorial_name} PRIVATE ${ODECRAFT_TUTORIAL_COMPILE_OPTIONS})
    target_link_options(${tutorial_name} PRIVATE ${ODECRAFT_TUTORIAL_LINK_OPTIONS})
endforeach()

# ArbitraryPrecision additionally needs the MPFR-enabled lazy target.
if(TARGET ArbitraryPrecision)
    if(TARGET lazy::mpfr)
        target_link_libraries(ArbitraryPrecision PRIVATE lazy::mpfr)
    else()
        message(FATAL_ERROR "ArbitraryPrecision needs lazy::mpfr; configure with -DLAZY_ENABLE_MPREAL=ON")
    endif()
endif()
