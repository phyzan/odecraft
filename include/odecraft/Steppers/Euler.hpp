#pragma once

#include <odecraft/Core/RichSolver/RichBase.hpp>

namespace ode{

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived = void>
class Euler : public detail::BaseDispatcher<GetDerived<Euler<T, SD, OdeType, SP, Derived>, Derived>, T, SD, OdeType, SP>{

    ODECRAFT_DECLARE_STEPPER;
    
    using Base = detail::BaseDispatcher<GetDerived<Euler<T, SD, OdeType, SP, Derived>, Derived>, T, SD, OdeType, SP>;

public:
    using State = state_t<T, SD>;
    using RealState = real_state_t<T, SD>;

    static constexpr int ERR_EST_ORDER = 1;
    static constexpr bool IS_IMPLICIT = false;

    DEFAULT_RULE_OF_FOUR(Euler)
    Euler(OdeType ode, T t0, State q0, T stepsize, int dir=1, EventList<T, State> events = {});

    // Constructor signature that follows the main constructor pattern
    Euler(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T, State> events = {});

    Stepper method() const;

protected:

    StepResult  adapt_impl(T& t_new, T& habs_new, State& q_new, const T& t, const T& habs, const State&);
    auto        local_interp_impl() const;

};


namespace detail{

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
struct SolverTypeGetter<Stepper::Euler, SP, T, SD, OdeType, Derived>{
    using type = Euler<T, SD, OdeType, SP, Derived>;
};

} // namespace ode::detail


} // namespace ode
