#pragma once

/**
 * @file Steppers/BDF/CraftedBDF.hpp
 * @brief Implicit variable-order BDF stepper for stiff systems. Compiled by src/BDF.cpp.
 *
 * BDF's constructor is a member template, so its definition travels with the declarations.
 *
 * @note BDF needs a Jacobian. Leaving `ode_t<T>::Jac` null makes it fall back to finite
 *       differences, which works but costs accuracy and Rhs evaluations; supply one if you can.
 */

#include <odecraft/Core/CraftedBase.hpp>
#include "BDF.hpp"


namespace ode::crafted{

/// @brief Implicit backward-differentiation-formula stepper (orders 1-5).
template<typename T>
using BDF = ::ode::BDF<T, SD, ode_t<T>, ::ode::SolverPolicy::RichVirtual>;

} // namespace ode::crafted


// Explicit instantiations live at global scope: an explicit instantiation must appear in a
// namespace enclosing its template's, and ODECRAFT_STEPPER_SET spells every name out fully.
ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_STEPPER, BDF)
