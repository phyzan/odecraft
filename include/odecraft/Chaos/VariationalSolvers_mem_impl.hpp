#ifndef ODECRAFT_VARIATIONAL_SOLVERS_MEM_IMPL_HPP
#define ODECRAFT_VARIATIONAL_SOLVERS_MEM_IMPL_HPP

#include <odecraft/Chaos/VariationalSolvers.hpp>
#include <odecraft/DenseOde/OdeInt_mem_impl.hpp>
#include <odecraft/Toolkit/Tools.hpp>


namespace ode::chaos{

template<typename T, size_t N>
template<hasRhsFunc<T> OdeType>
VariationalODE<T, N>::VariationalODE(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, EventList<T> events, Stepper method) : Base(2*q0.size()){
    assert(q0.size() == delta_q0.size() && "q0 and delta_q0 must have the same size in VariationalODE");
    // Must create solver BEFORE register_state(), since it accesses solver_
    this->solver_ = make_variational_solver<UtilPolicy::RichVirtual>(method, ode, t0, q0, delta_q0, period, rtol, atol, min_step, max_step, stepsize, dir, std::move(events));

    const EventCollection<T>& event_coll = this->solver()->get_event_col();

    this->cached_idx_.resize(event_coll.size(), 0);
    Base::register_state();
    for (size_t i=0; i<event_coll.size(); i++){
        this->event_data_.allocate_event(event_coll.event(i).name());
    }
}


} // namespace ode::chaos

#endif // ODECRAFT_VARIATIONAL_SOLVERS_MEM_IMPL_HPP
