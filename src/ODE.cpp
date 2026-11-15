#include <odecraft/DenseOde/CraftedODE.hpp>
#include <odecraft/DenseOde/OdeInt_impl.hpp>

namespace ode{

#define ODECRAFT_INSTANTIATE_ODE(T)                                                     \
    template class EventCounter<T, crafted::SD>;                                        \
    template class EventCounter<T, crafted::VarSD>;                                     \
    template class ODE<T, crafted::SD>;                                                 \
    template class ODE<T, crafted::VarSD>;                                              \
    ODECRAFT_FOR_EACH_STEPPER(ODECRAFT_ODE_INIT, template, T)

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_ODE)

#undef ODECRAFT_INSTANTIATE_ODE

} // namespace ode
