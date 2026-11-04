#pragma once

/**
 * @file Steppers/RK4/CraftedRK4.hpp
 * @brief Classic fixed-step RK4. Compiled by src/RK4.cpp.
 *
 * RK4's constructor is a member template (it forwards a trailing pack to its base), so its
 * definition has to travel with the declarations for the class to be constructible here.
 */

#include <odecraft/Core/CraftedBase.hpp>
#include "RK4.hpp"


namespace ode::crafted{

/// @brief Fourth-order fixed-step stepper.
template<typename T>
using RK4 = ::ode::RK4<T, SD, ode_t<T>, ::ode::SolverPolicy::RichVirtual>;

} // namespace ode::crafted


// Explicit instantiations live at global scope: an explicit instantiation must appear in a
// namespace enclosing its template's, and ODECRAFT_STEPPER_SET spells every name out fully.
ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_STEPPER, RK4)
