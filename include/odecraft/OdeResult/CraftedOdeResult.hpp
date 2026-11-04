#pragma once

/**
 * @file OdeResult/CraftedOdeResult.hpp
 * @brief Integration output of the compiled interface. Compiled by src/OdeHistory.cpp.
 *
 * Links after Interpolators (OdeSolution owns one).
 */

#include <odecraft/Interpolation/Univariate/CraftedStateInterp.hpp>
#include "OdeResult.hpp"


namespace ode::crafted{

// D is SD, or VarSD for the output of a variational driver.

template<typename T, typename D = SD>
using OrbitData = ::ode::OrbitData<T, D>;

template<typename T, typename D = SD>
using EventData = ::ode::EventData<T, D>;

/// @brief The sampled points, event hits and status of a finished integration.
template<typename T, typename D = SD>
using OdeResult = ::ode::OdeResult<T, D>;

/// @brief An OdeResult that additionally answers `operator()(out, t)` anywhere in range.
template<typename T, typename D = SD>
using OdeSolution = ::ode::OdeSolution<T, D>;

} // namespace ode::crafted


namespace ode{

#define ODECRAFT_EXTERN_ODE_HISTORY_OF(T, D)    \
    extern template struct OrbitData<T, D>;     \
    extern template class EventData<T, D>;      \
    extern template class OdeResult<T, D>;      \
    extern template class OdeSolution<T, D>;

#define ODECRAFT_EXTERN_ODE_HISTORY(T)                  \
    ODECRAFT_EXTERN_ODE_HISTORY_OF(T, crafted::SD)      \
    ODECRAFT_EXTERN_ODE_HISTORY_OF(T, crafted::VarSD)

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_ODE_HISTORY)

#undef ODECRAFT_EXTERN_ODE_HISTORY
#undef ODECRAFT_EXTERN_ODE_HISTORY_OF

} // namespace ode
