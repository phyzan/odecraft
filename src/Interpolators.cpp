#include <odecraft/Interpolation/Univariate/CraftedStateInterp.hpp>
#include <odecraft/Interpolation/Univariate/StateInterp_impl.hpp>

namespace ode::interp::uni{

#define ODECRAFT_INSTANTIATE_INTERPOLATORS_OF(T, D, S)                              \
    template class Interpolator<T, D>;                                              \
    template class LocalInterpolator<T, D>;                                         \
    template class LinkedInterpolator<T, D>;                                        \
    template void lin_interp<T, D>(S&, const T&, const T&, const T&,                \
                                   const S&, const S&);                             \
    template void coef_mat_interp<T, D>(S&, const T&, const T&, const T&,           \
                                        const S&, const S&, const S*, size_t);

#define ODECRAFT_INSTANTIATE_INTERPOLATORS(T)                                                   \
    template class Interval<T>;                                                                 \
    ODECRAFT_INSTANTIATE_INTERPOLATORS_OF(T, ::ode::crafted::SD, ::ode::crafted::State<T>)      \
    ODECRAFT_INSTANTIATE_INTERPOLATORS_OF(T, ::ode::crafted::VarSD, ::ode::crafted::VarState<T>)

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_INTERPOLATORS)

#undef ODECRAFT_INSTANTIATE_INTERPOLATORS
#undef ODECRAFT_INSTANTIATE_INTERPOLATORS_OF

} // namespace ode::interp::uni
