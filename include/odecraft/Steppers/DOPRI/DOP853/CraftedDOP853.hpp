#pragma once

/**
 * @file Steppers/DOPRI/DOP853/CraftedDOP853.hpp
 * @brief Eighth-order Dormand-Prince stepper. Compiled by src/DOP853.cpp.
 */

#include <odecraft/Core/CraftedBase.hpp>
#include "DOP853.hpp"


namespace ode::crafted{

/// @brief Dormand-Prince 8(5,3) adaptive stepper, for tight tolerances.
template<typename T>
using DOP853 = ::ode::DOP853<T, SD, ode_t<T>, ::ode::SolverPolicy::RichVirtual>;

} // namespace ode::crafted


// Explicit instantiations live at global scope: an explicit instantiation must appear in a
// namespace enclosing its template's, and ODECRAFT_STEPPER_SET spells every name out fully.
ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_STEPPER, DOP853)
