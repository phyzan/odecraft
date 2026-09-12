#ifndef ODECRAFT_COMPILED_RK23_DOPRI_HPP
#define ODECRAFT_COMPILED_RK23_DOPRI_HPP

/**
 * @file Compiled/RK23_DOPRI.hpp
 * @brief Bogacki-Shampine 3(2) adaptive stepper. Compiled by src/RK23_DOPRI.cpp.
 */

#include <odecraft/Compiled/SolverBase.hpp>
#include <odecraft/Steppers/RK23_DOPRI.hpp>


namespace ode::crafted{

/// @brief Bogacki-Shampine 3(2) adaptive stepper.
template<typename T>
using RK23 = ::ode::RK23<T, 0, ::ode::SolverPolicy::RichVirtual, ode_t<T>>;

} // namespace ode::crafted


// Explicit instantiations live at global scope: an explicit instantiation must appear in a
// namespace enclosing its template's, and ODECRAFT_STEPPER_SET spells every name out fully.
ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_DECLARE_STEPPER, RK23)

#endif // ODECRAFT_COMPILED_RK23_DOPRI_HPP
