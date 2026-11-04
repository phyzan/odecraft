#include <odecraft/DenseOde/CraftedODE.hpp>
#include <odecraft/DenseOde/OdeInt_impl.hpp>

namespace ode{

#define ODECRAFT_INSTANTIATE_ODE(T)                                                     \
    template class EventCounter<T, crafted::SD>;                                        \
    template class EventCounter<T, crafted::VarSD>;                                     \
    template class ODE<T, crafted::SD>;                                                 \
    template class ODE<T, crafted::VarSD>;                                              \
    template void ODE<T, crafted::SD>::init<crafted::ode_t<T>>(                         \
        crafted::ode_t<T>, T, crafted::State<T>, crafted::State<T>, crafted::State<T>, T, T, T, int, crafted::EventList<T>, Stepper);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_ODE)

#undef ODECRAFT_INSTANTIATE_ODE

} // namespace ode
