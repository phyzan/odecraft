#pragma once

/**
 * @file Steppers/Euler/CraftedEuler.hpp
 * @brief Fixed-step explicit Euler. Compiled by src/Euler.cpp.
 */

#include <odecraft/Core/CraftedBase.hpp>
#include "Euler.hpp"


namespace ode::crafted{

/// @brief First-order fixed-step stepper.
template<typename T>
using Euler = ::ode::Euler<T, SD, ode_t<T>, ::ode::SolverPolicy::RichVirtual>;

} // namespace ode::crafted


// Explicit instantiations live at global scope: an explicit instantiation must appear in a
// namespace enclosing its template's, and ODECRAFT_STEPPER_SET spells every name out fully.
ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_STEPPER, Euler)
