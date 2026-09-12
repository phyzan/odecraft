#ifndef ODECRAFT_COMPILED_DOP853_HPP
#define ODECRAFT_COMPILED_DOP853_HPP

/**
 * @file Compiled/DOP853.hpp
 * @brief Eighth-order Dormand-Prince stepper. Compiled by src/DOP853.cpp.
 */

#include <odecraft/Compiled/SolverBase.hpp>
#include <odecraft/Steppers/DOP853.hpp>


namespace ode::crafted{

/// @brief Dormand-Prince 8(5,3) adaptive stepper, for tight tolerances.
template<typename T>
using DOP853 = ::ode::DOP853<T, 0, ::ode::SolverPolicy::RichVirtual, ode_t<T>>;

} // namespace ode::crafted


// Explicit instantiations live at global scope: an explicit instantiation must appear in a
// namespace enclosing its template's, and ODECRAFT_STEPPER_SET spells every name out fully.
ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_DECLARE_STEPPER, DOP853)

#endif // ODECRAFT_COMPILED_DOP853_HPP
