#pragma once

#include "odecraft/Toolkit/Tools.hpp"
#include "Euler.hpp"

namespace ode{

namespace detail{

template<typename T, typename SD>
real_state_t<T, SD> alloc_real_state(const state_t<T, SD>& proto){
    real_state_t<T, SD> real_state;
    SD::resize_like(real_state, proto);
    SD::for_each([&](T& elem){
        elem = 0;
    }, real_state);
    return real_state;
}

} // namespace ode::detail


template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
Euler<T, SD, OdeType, SP, Derived>::Euler(OdeType ode, T t0, const State& q0, T stepsize, int dir, EventList<T, State> events) : Base(std::move(ode), std::move(t0), q0, detail::alloc_real_state<T, SD>(q0), detail::alloc_real_state<T, SD>(q0), 0, 0, stepsize, dir, std::move(events)) {}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
Euler<T, SD, OdeType, SP, Derived>::Euler(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T /*min_step*/, T /*max_step*/, T stepsize, int dir, EventList<T, State> events) : Base(std::move(ode), std::move(t0), std::move(q0), std::move(rtol), std::move(atol), 0, 0, stepsize, dir, std::move(events)) {}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
StepResult Euler<T, SD, OdeType, SP, Derived>::adapt_impl(T& t_new, T& habs_new, State& q_new, const T& t, const T& habs, const State& q){
    const T h = habs * this->direction();
    t_new = t + h;
    habs_new = habs;

    this->rhs(q_new, t, q);
    SD::for_each([&h](value_t<T, SD>& qi_new, const value_t<T, SD>& qi){
        qi_new = qi + qi_new * h;
    }, q_new, q);
    if (!all_are_finite<T, SD>(q_new)){
        return StepResult::NonFiniteError;
    }else{
        return StepResult::Success;
    }
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
auto Euler<T, SD, OdeType, SP, Derived>::local_interp_impl() const{
    return make_light_functor([](State& out, const T& t, const auto& t1, const auto& t2, const auto& y1, const auto& y2){
        return lin_interp<T, SD>(out, t, t1, t2, y1, y2);
    }, this->t_old(), this->t_new(), this->vector_old(), this->vector_new());
}


} // namespace ode
