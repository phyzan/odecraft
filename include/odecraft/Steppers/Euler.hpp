#ifndef ODECRAFT_EULER_HPP
#define ODECRAFT_EULER_HPP

#include <odecraft/Core/RichSolver/RichBase.hpp>

namespace ode{

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived = void>
class Euler : public detail::BaseDispatcher<GetDerived<Euler<T, SD, JD, OdeType, SP, Derived>, Derived>, T, SD, JD, OdeType, SP>{

    using Base = detail::BaseDispatcher<GetDerived<Euler<T, SD, JD, OdeType, SP, Derived>, Derived>, T, SD, JD, OdeType, SP>;

public:
    using State = state_t<T, SD>;
    using RealState = Base::RealState;

    static constexpr int ERR_EST_ORDER = 1;
    static constexpr bool IS_IMPLICIT = false;

    DEFAULT_RULE_OF_FOUR(Euler)
    Euler(OdeType ode, T t0, State q0, T stepsize, int dir=1, EventList<T, State> events = {});

    // Constructor signature that follows the main constructor pattern
    Euler(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T, State> events = {});

    Stepper method() const;

protected:

    StepResult  adapt_impl(T& t_new, T& habs_new, State& q_new, const T& t, const T& habs, const State&);
    void        interp_impl(State& result, const T& t) const;
    auto        local_interp() const;

};


namespace detail{

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
struct SolverTypeGetter<Stepper::Euler, SP, T, SD, JD, OdeType, Derived>{
    using type = Euler<T, SD, JD, OdeType, SP, Derived>;
};

} // namespace ode::detail


} // namespace ode

#endif // ODECRAFT_EULER_HPP
