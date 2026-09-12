#ifndef ODECRAFT_COMPILED_ODECRAFT_HPP
#define ODECRAFT_COMPILED_ODECRAFT_HPP

/**
 * @file Compiled/odecraft.hpp
 * @brief Single include for the whole pre-compiled `ode::crafted` interface.
 *
 * Include this and link `odecraft::compiled`. Nothing here instantiates a solver in your
 * translation unit -- the library ships those, for float, double and long double.
 *
 * The header-only interface in namespace `ode` is a separate thing and is not pulled in by
 * this header: use <odecraft/odecraft.hpp> for that. The two can coexist in one program,
 * and even in one translation unit.
 *
 * @author Foivos Zanias
 */

#include <odecraft/Compiled/Toolkit.hpp>        // IWYU pragma: export
#include <odecraft/Compiled/Interpolators.hpp>  // IWYU pragma: export
#include <odecraft/Compiled/Events.hpp>         // IWYU pragma: export
#include <odecraft/Compiled/OdeHistory.hpp>     // IWYU pragma: export
#include <odecraft/Compiled/SolverBase.hpp>     // IWYU pragma: export
#include <odecraft/Compiled/Steppers.hpp>       // IWYU pragma: export
#include <odecraft/Compiled/ODE.hpp>            // IWYU pragma: export
#include <odecraft/Compiled/Chaos.hpp>          // IWYU pragma: export

#endif // ODECRAFT_COMPILED_ODECRAFT_HPP
