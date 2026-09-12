#include <odecraft/Compiled/Interpolators.hpp>
#include <odecraft/Interpolation/Univariate/StateInterp_impl.hpp>

namespace ode::interp::uni{

#define ODECRAFT_INSTANTIATE_INTERPOLATORS(T)                                       \
    template class Interval<T>;                                                     \
    template class Interpolator<T, 0>;                                              \
    template class LocalInterpolator<T, 0>;                                         \
    template class LinkedInterpolator<T, 0>;                                        \
    template void lin_interp<T>(T*, const T&, const T&, const T&,                   \
                                const T*, const T*, size_t);                        \
    template void coef_mat_interp<T>(T*, const T&, const T&, const T&,              \
                                     const T*, const T*, const T*, size_t, size_t);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_INTERPOLATORS)

#undef ODECRAFT_INSTANTIATE_INTERPOLATORS

} // namespace ode::interp::uni
