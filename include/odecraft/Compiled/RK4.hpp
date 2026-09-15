#ifndef ODECRAFT_COMPILED_RK4_HPP
#define ODECRAFT_COMPILED_RK4_HPP

/**
 * @file Compiled/RK4.hpp
 * @brief Classic fixed-step RK4. Compiled by src/RK4.cpp.
 *
 * RK4's constructor is a member template (it forwards a trailing pack to its base), so its
 * definition has to travel with the declarations for the class to be constructible here.
 */

#include <odecraft/Compiled/SolverBase.hpp>
#include <odecraft/Steppers/RK4.hpp>


namespace ode::crafted{

/// @brief Fourth-order fixed-step stepper.
template<typename T>
using RK4 = ::ode::RK4<T, 0, ::ode::SolverPolicy::RichVirtual, ode_t<T>>;

} // namespace ode::crafted


// Explicit instantiations live at global scope: an explicit instantiation must appear in a
// namespace enclosing its template's, and ODECRAFT_STEPPER_SET spells every name out fully.
ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_DECLARE_STEPPER, RK4)

#endif // ODECRAFT_COMPILED_RK4_HPP
