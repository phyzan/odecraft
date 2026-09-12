#ifndef ODECRAFT_COMPILED_EULER_HPP
#define ODECRAFT_COMPILED_EULER_HPP

/**
 * @file Compiled/Euler.hpp
 * @brief Fixed-step explicit Euler. Compiled by src/Euler.cpp.
 */

#include <odecraft/Compiled/SolverBase.hpp>
#include <odecraft/Steppers/Euler.hpp>


namespace ode::crafted{

/// @brief First-order fixed-step stepper.
template<typename T>
using Euler = ::ode::Euler<T, 0, ::ode::SolverPolicy::RichVirtual, ode_t<T>>;

} // namespace ode::crafted


// Explicit instantiations live at global scope: an explicit instantiation must appear in a
// namespace enclosing its template's, and ODECRAFT_STEPPER_SET spells every name out fully.
ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_DECLARE_STEPPER, Euler)

#endif // ODECRAFT_COMPILED_EULER_HPP
