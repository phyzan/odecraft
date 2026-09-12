#ifndef ODECRAFT_COMPILED_RK45_DOPRI_HPP
#define ODECRAFT_COMPILED_RK45_DOPRI_HPP

/**
 * @file Compiled/RK45_DOPRI.hpp
 * @brief Dormand-Prince 5(4) adaptive stepper -- the general-purpose default. Compiled by src/RK45_DOPRI.cpp.
 */

#include <odecraft/Compiled/SolverBase.hpp>
#include <odecraft/Steppers/RK45_DOPRI.hpp>


namespace ode::crafted{

/// @brief Dormand-Prince 5(4) adaptive stepper -- the general-purpose default.
template<typename T>
using RK45 = ::ode::RK45<T, 0, ::ode::SolverPolicy::RichVirtual, ode_t<T>>;

} // namespace ode::crafted


// Explicit instantiations live at global scope: an explicit instantiation must appear in a
// namespace enclosing its template's, and ODECRAFT_STEPPER_SET spells every name out fully.
ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_DECLARE_STEPPER, RK45)

#endif // ODECRAFT_COMPILED_RK45_DOPRI_HPP
