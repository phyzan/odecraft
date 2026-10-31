#pragma once


#include <odecraft/Core/RichSolver/RichBase.hpp>

namespace ode{

// Compile with ODECRAFT_RK4_DENSE to use rk4 steps for interpolation instead of the default Hermite polynomials
// This increases accuracy at the cost of performance

namespace detail{

/// @brief One classic Runge-Kutta 4 step from (t, q). On entry K[0] holds the derivative at (t, q);
/// the step fills K[1..3] with the other stages, and r with the stage inputs.
template<typename T, typename SD, typename RhsFn>
void rk4_step(state_t<T, SD>& q_new, cache_t<T> t, cache_t<T> h, const state_t<T, SD>& q,
              std::array<state_t<T, SD>, 5>& K, state_t<T, SD>& r, RhsFn&& rhs);

/// @brief Cubic Hermite interpolation between (t1, y1) and (t2, y2), whose derivatives are f1 and f2.
template<typename T, typename SD>
void rk4_interp(state_t<T, SD>& out, const T& t, const T& t1, const T& t2, const state_t<T, SD>& y1, const state_t<T, SD>& y2,
                const state_t<T, SD>& f1, const state_t<T, SD>& f2);

} // namespace ode::detail


template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived = void>
class RK4 : public detail::BaseDispatcher<GetDerived<RK4<T, SD, JD, OdeType, SP, Derived>, Derived>, T, SD, JD, OdeType, SP>{

    ODECRAFT_DECLARE_STEPPER;

    using Base = detail::BaseDispatcher<GetDerived<RK4<T, SD, JD, OdeType, SP, Derived>, Derived>, T, SD, JD, OdeType, SP>;

public:
    using State = state_t<T, SD>;
    using RealState = Base::RealState;

    static constexpr size_t         Nstages = 4;
    static constexpr int            ERR_EST_ORDER = 4;
    static constexpr size_t         INTERP_ORDER = 4;
    static constexpr bool           IS_IMPLICIT = false;

    RK4(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T, State> events = {});

    DEFAULT_RULE_OF_FOUR(RK4)

    Stepper method() const;

protected:

    StepResult  adapt_impl(T& t_new, T& habs_new, State& q_new, const T& t, const T& habs, const State& q);
    auto        local_interp_impl() const;

private:

    void        cache_interp_data() const;

    // the stages of the last step: K_[0] the derivative at its start, and K_[Nstages] the
    // derivative at its end, cached by cache_interp_data
    mutable std::array<State, Nstages+1>        K_;
    mutable bool interp_data_is_cached_ = false;

    // the stage input of a step
    worker_t<State>                             r_scratch_;

#ifdef ODECRAFT_RK4_DENSE
    // the stages of an interpolating step
    worker_t<std::array<State, Nstages+1>>      K_scratch_;
#endif

};


namespace detail{

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
struct SolverTypeGetter<Stepper::RK4, SP, T, SD, JD, OdeType, Derived>{
    using type = RK4<T, SD, JD, OdeType, SP, Derived>;
};

} // namespace ode::detail


} // namespace ode
