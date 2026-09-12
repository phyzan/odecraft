#ifndef ODECRAFT_COMPILED_INTERPOLATORS_HPP
#define ODECRAFT_COMPILED_INTERPOLATORS_HPP

/**
 * @file Compiled/Interpolators.hpp
 * @brief Dense-output interpolators of the compiled interface. Compiled by src/Interpolators.cpp.
 *
 * First link unit: depends on nothing else in the compiled library.
 */

#include <odecraft/Compiled/Toolkit.hpp>
#include <odecraft/Interpolation/Univariate/StateInterp.hpp>


namespace ode::crafted{

using ::ode::interp::uni::Interval;

/// @brief Abstract dense-output interpolant over one or more integration steps.
template<typename T>
using Interpolator = ::ode::interp::uni::Interpolator<T, 0>;

/// @brief Interpolant over a single step.
template<typename T>
using LocalInterpolator = ::ode::interp::uni::LocalInterpolator<T, 0>;

/// @brief Chain of interpolants covering a whole integration range.
template<typename T>
using LinkedInterpolator = ::ode::interp::uni::LinkedInterpolator<T, 0>;

/// @brief Owning handle to an Interpolator.
template<typename T>
using InterpObj = ::ode::interp::uni::InterpObj<T, 0>;

} // namespace ode::crafted


// The library ships these instantiations; suppress them everywhere else. src/Interpolators.cpp
// re-states each one as an explicit instantiation *definition*, which is allowed to follow
// these declarations in the same translation unit.
namespace ode::interp::uni{

// The two free function templates are listed for the same reason the CRTP bases are listed
// in Compiled/SolverBase.hpp: at -O3 an implicit instantiation that gets fully inlined leaves
// no standalone body behind, and the stepper units call these across translation units.
#define ODECRAFT_EXTERN_INTERPOLATORS(T)                                                    \
    extern template class Interval<T>;                                                      \
    extern template class Interpolator<T, 0>;                                               \
    extern template class LocalInterpolator<T, 0>;                                          \
    extern template class LinkedInterpolator<T, 0>;                                         \
    extern template void lin_interp<T>(T*, const T&, const T&, const T&,                    \
                                       const T*, const T*, size_t);                         \
    extern template void coef_mat_interp<T>(T*, const T&, const T&, const T&,               \
                                            const T*, const T*, const T*, size_t, size_t);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_INTERPOLATORS)

#undef ODECRAFT_EXTERN_INTERPOLATORS

} // namespace ode::interp::uni

#endif // ODECRAFT_COMPILED_INTERPOLATORS_HPP
