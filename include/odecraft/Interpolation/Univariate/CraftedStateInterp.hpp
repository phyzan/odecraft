#pragma once

/**
 * @file Interpolation/Univariate/CraftedStateInterp.hpp
 * @brief Dense-output interpolators of the compiled interface. Compiled by src/Interpolators.cpp.
 *
 * First link unit: depends on nothing else in the compiled library.
 */

#include <odecraft/Toolkit/CraftedToolkit.hpp>
#include "StateInterp.hpp"


namespace ode::crafted{

using ::ode::interp::uni::Interval;

/// @brief Abstract dense-output interpolant over one or more integration steps.
template<typename T, typename D = SD>
using Interpolator = ::ode::interp::uni::Interpolator<T, D>;

/// @brief Interpolant over a single step.
template<typename T, typename D = SD>
using LocalInterpolator = ::ode::interp::uni::LocalInterpolator<T, D>;

/// @brief Chain of interpolants covering a whole integration range.
template<typename T, typename D = SD>
using LinkedInterpolator = ::ode::interp::uni::LinkedInterpolator<T, D>;

/// @brief Owning handle to an Interpolator.
template<typename T, typename D = SD>
using InterpObj = ::ode::interp::uni::InterpObj<T, D>;

} // namespace ode::crafted


// The library ships these instantiations; suppress them everywhere else. src/Interpolators.cpp
// re-states each one as an explicit instantiation *definition*, which is allowed to follow
// these declarations in the same translation unit.
namespace ode::interp::uni{

// The two free function templates are listed for the same reason the CRTP bases are listed
// in Core/CraftedBase.hpp: at -O3 an implicit instantiation that gets fully inlined leaves
// no standalone body behind, and the stepper units call these across translation units.
#define ODECRAFT_EXTERN_INTERPOLATORS_OF(T, D, S)                                           \
    extern template class Interpolator<T, D>;                                               \
    extern template class LocalInterpolator<T, D>;                                          \
    extern template class LinkedInterpolator<T, D>;                                         \
    extern template void lin_interp<T, D>(S&, const T&, const T&, const T&,                 \
                                          const S&, const S&);                              \
    extern template void coef_mat_interp<T, D>(S&, const T&, const T&, const T&,            \
                                               const S&, const S&, const S*, size_t);

#define ODECRAFT_EXTERN_INTERPOLATORS(T)                                                    \
    extern template class Interval<T>;                                                      \
    ODECRAFT_EXTERN_INTERPOLATORS_OF(T, ::ode::crafted::SD, ::ode::crafted::State<T>)       \
    ODECRAFT_EXTERN_INTERPOLATORS_OF(T, ::ode::crafted::VarSD, ::ode::crafted::VarState<T>)

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_INTERPOLATORS)

#undef ODECRAFT_EXTERN_INTERPOLATORS
#undef ODECRAFT_EXTERN_INTERPOLATORS_OF

} // namespace ode::interp::uni
