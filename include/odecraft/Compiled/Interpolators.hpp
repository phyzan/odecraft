#ifndef ODECRAFT_COMPILED_INTERPOLATORS_HPP
#define ODECRAFT_COMPILED_INTERPOLATORS_HPP

#include <odecraft/Interpolation/Univariate/StateInterp.hpp>


namespace ode::crafted{

template<typename T>
using Interpolator = ::ode::interp::uni::Interpolator<T, 0>;

} // namespace ode::crafted

#endif // ODECRAFT_COMPILED_INTERPOLATORS_HPP