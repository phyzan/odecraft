#include <odecraft/OdeResult/CraftedOdeResult.hpp>
#include <odecraft/OdeResult/OdeResult_impl.hpp>

namespace ode{

#define ODECRAFT_INSTANTIATE_ODE_HISTORY_OF(T, D) \
    template struct OrbitData<T, D>;            \
    template class EventData<T, D>;             \
    template class OdeResult<T, D>;             \
    template class OdeSolution<T, D>;

#define ODECRAFT_INSTANTIATE_ODE_HISTORY(T)                 \
    ODECRAFT_INSTANTIATE_ODE_HISTORY_OF(T, crafted::SD)     \
    ODECRAFT_INSTANTIATE_ODE_HISTORY_OF(T, crafted::VarSD)

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_ODE_HISTORY)

#undef ODECRAFT_INSTANTIATE_ODE_HISTORY
#undef ODECRAFT_INSTANTIATE_ODE_HISTORY_OF

} // namespace ode
