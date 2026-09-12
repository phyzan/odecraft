#include <odecraft/Compiled/OdeHistory.hpp>
#include <odecraft/OdeResult/OdeResult_impl.hpp>

namespace ode{

#define ODECRAFT_INSTANTIATE_ODE_HISTORY(T) \
    template struct OrbitData<T>;           \
    template class EventData<T>;            \
    template class OdeResult<T, 0>;         \
    template class OdeSolution<T, 0>;

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_ODE_HISTORY)

#undef ODECRAFT_INSTANTIATE_ODE_HISTORY

} // namespace ode
