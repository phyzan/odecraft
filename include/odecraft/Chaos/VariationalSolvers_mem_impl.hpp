#pragma once

#include <odecraft/Chaos/VariationalSolvers.hpp>
#include <odecraft/DenseOde/OdeInt_mem_impl.hpp>
#include <odecraft/Toolkit/Tools.hpp>


namespace ode::chaos{

template<typename T, typename SD>
template<hasRhsFunc<T, state_t<T, SD>> OdeType>
VariationalODE<T, SD>::VariationalODE(OdeType ode, T t0, State q0, T period, RealState rtol, RealState atol, T min_step, T max_step, T stepsize, int dir, EventList<T, State> events, Stepper method) {
    // Must create solver BEFORE register_state(), since it accesses solver_
    this->solver_ = make_variational_solver<UtilPolicy::RichVirtual, T, SD>(method, std::move(ode), std::move(t0), std::move(q0), std::move(period), std::move(rtol), std::move(atol), std::move(min_step), std::move(max_step), std::move(stepsize), dir, std::move(events));

    const EventCollection<T, State>& event_coll = this->solver()->get_event_col();

    this->cached_idx_.resize(event_coll.size(), 0);
    Base::register_state();
    for (size_t i=0; i<event_coll.size(); i++){
        this->event_data_.allocate_event(event_coll.event(i).name());
    }
}


} // namespace ode::chaos
