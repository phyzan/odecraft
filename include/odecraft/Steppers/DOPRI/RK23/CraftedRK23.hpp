#pragma once

/**
 * @file Steppers/DOPRI/RK23/CraftedRK23.hpp
 * @brief Bogacki-Shampine 3(2) adaptive stepper. Compiled by src/RK23_DOPRI.cpp.
 */

#include <odecraft/Core/CraftedBase.hpp>
#include "RK23_DOPRI.hpp"


namespace ode::crafted{

/// @brief Bogacki-Shampine 3(2) adaptive stepper.
template<typename T>
using RK23 = ::ode::RK23<T, SD, ode_t<T>, ::ode::SolverPolicy::RichVirtual>;

} // namespace ode::crafted


// Explicit instantiations live at global scope: an explicit instantiation must appear in a
// namespace enclosing its template's, and ODECRAFT_STEPPER_SET spells every name out fully.
ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_STEPPER, RK23)
