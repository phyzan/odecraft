#ifndef ODECRAFT_RK4_IMPL_HPP
#define ODECRAFT_RK4_IMPL_HPP

#include "RK4.hpp" // IWYU pragma: keep

namespace ode{

namespace detail{

template<typename T, typename SD, typename RhsFn>
void rk4_step(state_t<T, SD>& q_new, cache_t<T> t, cache_t<T> h, const state_t<T, SD>& q,
              std::array<state_t<T, SD>, 5>& K, state_t<T, SD>& r, RhsFn&& rhs){
    using S = value_t<T, SD>;

    SD::for_each([&](S& r_i, const S& q_i, const S& k0_i){
        r_i = q_i + h * k0_i / T(2);
    }, r, q, K[0]);
    rhs(K[1], t + h/2, r);

    SD::for_each([&](S& r_i, const S& q_i, const S& k1_i){
        r_i = q_i + h * k1_i / T(2);
    }, r, q, K[1]);
    rhs(K[2], t + h/2, r);

    SD::for_each([&](S& r_i, const S& q_i, const S& k2_i){
        r_i = q_i + h * k2_i;
    }, r, q, K[2]);
    rhs(K[3], t + h, r);

    SD::for_each([&](S& qn_i, const S& q_i, const S& k0_i, const S& k1_i, const S& k2_i, const S& k3_i){
        qn_i = q_i + h * (k0_i + T(2)*k1_i + T(2)*k2_i + k3_i) / T(6);
    }, q_new, q, K[0], K[1], K[2], K[3]);
}

template<typename T, typename SD>
void rk4_interp(state_t<T, SD>& out, const T& t, const T& t1, const T& t2, const state_t<T, SD>& y1, const state_t<T, SD>& y2,
                const state_t<T, SD>& f1, const state_t<T, SD>& f2){
    assert( (t != t1 && t != t2) && "rk4_interp must only be called on the open interval");
    using S = value_t<T, SD>;

    const T h = t2 - t1;
    const T theta = (t - t1) / h;

    const T x2 = theta * theta;
    const T x3 = x2 * theta;
    const T h00 = 1 - 3*x2 + 2*x3;
    const T h10 = theta - 2*x2 + x3;
    const T h01 = 3*x2 - 2*x3;
    const T h11 = -x2 + x3;

    SD::for_each([&](S& out_i, const S& y1_i, const S& y2_i, const S& f1_i, const S& f2_i){
        out_i = h00 * y1_i + h * h10 * f1_i + h01 * y2_i + h * h11 * f2_i;
    }, out, y1, y2, f1, f2);
}

} // namespace ode::detail


template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
RK4<T, SD, JD, OdeType, SP, Derived>::RK4(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T /*min_step*/, T /*max_step*/, T stepsize, int dir, EventList<T, State> events)
    : Base(std::move(ode), t0, q0, std::move(rtol), std::move(atol), 0, 0, stepsize, dir, std::move(events)),
      r_scratch_(q0) {
    // min_step and max_step are not used in RK4
    // the arrays of states take the shape of q0 too
    K_.fill(q0);
#ifdef ODECRAFT_RK4_DENSE
    K_scratch_.get().fill(q0);
#endif
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
Stepper RK4<T, SD, JD, OdeType, SP, Derived>::method() const{
    return Stepper::RK4;
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
void RK4<T, SD, JD, OdeType, SP, Derived>::Reset(){
    Base::Reset();
    interp_data_set = false;
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
void RK4<T, SD, JD, OdeType, SP, Derived>::ReAdjust(const State& new_vector){
    Base::ReAdjust(new_vector);
    set_interp_data();
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
StepResult RK4<T, SD, JD, OdeType, SP, Derived>::adapt_impl(T& t_new, T& habs_new, State& q_new, const T& t, const T& habs, const State& q){
    // standard Runge-Kutta-4 with fixed step size

    // the derivative at the end of the step being replaced is no longer valid
    interp_data_set = false;

    const T h = habs * this->direction();

    t_new = t + h;
    habs_new = habs;

    decltype(auto) rhs_scratch = r_scratch_.get();

    this->rhs(K_[0], t, q);
    detail::rk4_step<T, SD>(q_new, t, h, q, K_, rhs_scratch, [this](State& out, const T& time, const State& y){
        this->rhs(out, time, y);
    });

    if (!all_are_finite<T, SD>(q_new)){
        return StepResult::NonFiniteError;
    }else{
        return StepResult::Success;
    }
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
void RK4<T, SD, JD, OdeType, SP, Derived>::interp_impl(State& out, const T& t) const{
#ifdef ODECRAFT_RK4_DENSE
    // a step straight from the start of the interval to t, which starts from the same derivative K_[0]
    decltype(auto) stage_scratch = K_scratch_.get();
    decltype(auto) rhs_scratch   = r_scratch_.get();
    copy_state<SD>(stage_scratch[0], K_[0]);
    detail::rk4_step<T, SD>(out, this->t_old(), t - this->t_old(), this->vector_old(), stage_scratch, rhs_scratch, [this](State& out_, const T& t_, const State& y_){
        this->rhs(out_, t_, y_);
    });
#else
    set_interp_data();
    detail::rk4_interp<T, SD>(out, t, this->t_old(), this->t_new(), this->vector_old(), this->vector_new(), K_[0], K_[Nstages]);
#endif
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
auto RK4<T, SD, JD, OdeType, SP, Derived>::local_interp() const{
#ifdef ODECRAFT_RK4_DENSE
    /*
    TODO : This is expensive. If we could guarantee that the RK4 object does not go out of scope,
    we can just reference the solver without copying the ode and the stages
    in the lambda below
    */
    return [ode = this->ode(), t1 = this->t_old(), y1 = this->vector_old(), K = K_, r = K_[0]](State& out, const T& t) mutable {
        // K[0] is the derivative at t1, which the steps from t1 only read
        detail::rk4_step<T, SD>(out, t1, t - t1, y1, K, r, [&ode](State& out_, const T& t_, const State& y_){
            ode.Rhs(out_, t_, y_);
        });
    };
#else
    set_interp_data();
    return [t1 = this->t_old(), t2 = this->t_new(), y1 = this->vector_old(), y2 = this->vector_new(), f1 = K_[0], f2 = K_[Nstages]](State& out, const T& t){
        detail::rk4_interp<T, SD>(out, t, t1, t2, y1, y2, f1, f2);
    };
#endif
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
void RK4<T, SD, JD, OdeType, SP, Derived>::set_interp_data() const{
    if (!interp_data_set){
        this->rhs(K_[Nstages], this->t_new(), this->vector_new());
        interp_data_set = true;
    }
}

} // namespace ode

#endif // ODECRAFT_RK4_IMPL_HPP
