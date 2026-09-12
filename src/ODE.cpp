#include <odecraft/Compiled/ODE.hpp>
#include <odecraft/DenseOde/OdeInt_impl.hpp>

namespace ode{

#define ODECRAFT_INSTANTIATE_ODE(T)                                                     \
    template class EventCounter<T, 0>;                                                  \
    template class ODE<T, 0>;                                                           \
    template void ODE<T, 0>::init<crafted::ode_t<T>>(                                   \
        crafted::ode_t<T>, T, View1D<T, 0>, T, T, T, T, T, int, EventList<T>, Stepper);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_ODE)

#undef ODECRAFT_INSTANTIATE_ODE

} // namespace ode
