#ifndef ODECRAFT_ODE_INT_MEM_IMPL_HPP
#define ODECRAFT_ODE_INT_MEM_IMPL_HPP

#include "OdeInt.hpp"
#include <odecraft/Core/VirtualBase.hpp>
#include <odecraft/Events/Events.hpp>
#include <algorithm>


namespace ode{

template<typename T, size_t N>
template<hasRhsFunc<T> OdeType>
ODE<T, N>::ODE(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, EventList<T> events, Stepper method) : ODE(q0.size()){
    init(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, dir, std::move(events), method);
}

} // namespace ode

#endif // ODECRAFT_ODE_INT_MEM_IMPL_HPP
