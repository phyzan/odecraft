#pragma once

/**
 * @file Steppers/DOPRI/RK45/CraftedRK45.hpp
 * @brief Dormand-Prince 5(4) adaptive stepper -- the general-purpose default. Compiled by src/RK45_DOPRI.cpp.
 */

#include <odecraft/Core/CraftedBase.hpp>
#include "RK45_DOPRI.hpp"


namespace ode::crafted{

/// @brief Dormand-Prince 5(4) adaptive stepper -- the general-purpose default.
template<typename T>
using RK45 = ::ode::RK45<T, SD, ode_t<T>, ::ode::SolverPolicy::RichVirtual>;

} // namespace ode::crafted


// Explicit instantiations live at global scope: an explicit instantiation must appear in a
// namespace enclosing its template's, and ODECRAFT_STEPPER_SET spells every name out fully.
ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_STEPPER, RK45)
