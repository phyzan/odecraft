#ifndef ODECRAFT_COMPILED_ODE_HISTORY_HPP
#define ODECRAFT_COMPILED_ODE_HISTORY_HPP

/**
 * @file Compiled/OdeHistory.hpp
 * @brief Integration output of the compiled interface. Compiled by src/OdeHistory.cpp.
 *
 * Links after Interpolators (OdeSolution owns one).
 */

#include <odecraft/Compiled/Interpolators.hpp>
#include <odecraft/OdeResult/OdeResult.hpp>


namespace ode::crafted{

using ::ode::OrbitData,
      ::ode::EventData;

/// @brief The sampled points, event hits and status of a finished integration.
template<typename T>
using OdeResult = ::ode::OdeResult<T, 0>;

/// @brief An OdeResult that additionally answers `operator()(t)` anywhere in range.
template<typename T>
using OdeSolution = ::ode::OdeSolution<T, 0>;

} // namespace ode::crafted


namespace ode{

#define ODECRAFT_EXTERN_ODE_HISTORY(T)          \
    extern template struct OrbitData<T>;        \
    extern template class EventData<T>;         \
    extern template class OdeResult<T, 0>;      \
    extern template class OdeSolution<T, 0>;

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_ODE_HISTORY)

#undef ODECRAFT_EXTERN_ODE_HISTORY

} // namespace ode

#endif // ODECRAFT_COMPILED_ODE_HISTORY_HPP
