#pragma once

#include "OdeInt.hpp"
#include <odecraft/Core/VirtualBase.hpp>
#include <odecraft/Events/Events.hpp>
#include <algorithm>


namespace ode{

template<typename T, typename SD>
template<hasRhsFunc<T, state_t<T, SD>> OdeType>
ODE<T, SD>::ODE(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step, T max_step, T stepsize, int dir, EventList<T, State> events, Stepper method){
    init(std::move(ode), std::move(t0), std::move(q0), std::move(rtol), std::move(atol), std::move(min_step), std::move(max_step), std::move(stepsize), dir, std::move(events), method);
}

} // namespace ode
