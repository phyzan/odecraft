#ifndef ODECRAFT_RK45_DOPRI_IMPL_HPP
#define ODECRAFT_RK45_DOPRI_IMPL_HPP

#include <odecraft/Steppers/DOPRI_impl.hpp>
#include <odecraft/Steppers/RK45_DOPRI.hpp>

namespace ode{

namespace detail{

template<typename T, typename SD, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
XDIFF_FORCEINLINE T rk45_step_impl(T& t_new, state_t<T, SD>& q_new, cache_t<T> t, cache_t<T> h, const state_t<T, SD>& q,
                 std::array<state_t<T, SD>, 7>& K, state_t<T, SD>& r, real_state_t<T, SD>& worker,
                 const real_state_t<T, SD>& rtol, const real_state_t<T, SD>& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs){
    using S = value_t<T, SD>;

    // Stage i: r = q + h * sum_j A(i, j) K[j], then K[i] = f(t + C[i] h, r)
    rk_stage<SD>(r, q, h, K, A, 1, stages<0>);              rhs(K[1], t + C[1]*h, r);
    rk_stage<SD>(r, q, h, K, A, 2, stages<0, 1>);           rhs(K[2], t + C[2]*h, r);
    rk_stage<SD>(r, q, h, K, A, 3, stages<0, 1, 2>);        rhs(K[3], t + C[3]*h, r);
    rk_stage<SD>(r, q, h, K, A, 4, stages<0, 1, 2, 3>);     rhs(K[4], t + C[4]*h, r);
    rk_stage<SD>(r, q, h, K, A, 5, stages<0, 1, 2, 3, 4>);  rhs(K[5], t + h,      r);

    // Solution update (b2 = 0)
    rk_stage<SD>(q_new, q, h, K, B, stages<0, 2, 3, 4, 5>);

    // FSAL: K[6] = f(t+h, q_new)
    rhs(K[6], t + h, q_new);
    t_new = t + h;

    // Error norm calculation
    SD::for_each([&](const S& q_i, const S& k0_i, const S& k2_i, const S& k3_i, const S& k4_i, const S& k5_i, const S& k6_i, const T& rtol_i, const T& atol_i, T& w){
        const auto err   = h * (E[0]*k0_i + E[2]*k2_i + E[3]*k3_i + E[4]*k4_i + E[5]*k5_i + E[6]*k6_i);
        const auto scale = atol_i + rtol_i * (abs(q_i) + abs(k0_i * h));
        w = abs(err) / scale;
    }, q, K[0], K[2], K[3], K[4], K[5], K[6], rtol, atol, worker);

    return max_elem<T, SD>(worker);
}


} // namespace ode::detail


template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
RK45<T, SD, JD, OdeType, SP, Derived>::RK45(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step, T max_step, T stepsize, int direction, EventList<T, State> events)
    : Base(std::move(ode), t0, q0, rtol, std::move(atol), min_step, max_step, stepsize, direction, std::move(events)),
      r_scratch_(q0), real_scratch(std::move(rtol)) {
    // the arrays of states take the shape of q0 too
    K_.fill(q0);
    coef_mat.fill(q0);
    this->rhs(K_[Nstages], t0, q0);
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
Stepper RK45<T, SD, JD, OdeType, SP, Derived>::method() const{
    return Stepper::RK45;
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
void RK45<T, SD, JD, OdeType, SP, Derived>::Reset(){
    Base::Reset();
    mat_is_set = false;
    this->rhs(K_[Nstages], this->t_new(), this->vector_new());
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
void RK45<T, SD, JD, OdeType, SP, Derived>::ReAdjust(const State& new_vector){
    Base::ReAdjust(new_vector);
    // freeze the interpolation coefficients (from t_old to t) before the last stage is overwritten
    this->set_coef_matrix();
    this->rhs(K_[Nstages], this->t(), new_vector);
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
void RK45<T, SD, JD, OdeType, SP, Derived>::set_coef_matrix() const{
    if (!mat_is_set){
        // K_ still holds the stages of the step just taken. The row of P for K_[1] is zero.
        detail::rk_interp_matrix<T, SD, Nstages, INTERP_ORDER>(coef_mat, K_, P.table, detail::stages<0, Nstages, 2, 3, 4, 5>);
        mat_is_set = true;
    }
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
T RK45<T, SD, JD, OdeType, SP, Derived>::step_impl(T& t_new, State& q_new, const T& t, const T& h, const State& q){
    decltype(auto) rhs_scratch = r_scratch_.get();
    decltype(auto) real_buf = real_scratch.get();
    return detail::rk45_step_impl<T, SD>(
        t_new,
        q_new,
        t,
        h,
        q,
        K_,
        rhs_scratch,
        real_buf,
        this->rtol(),
        this->atol(),
        A, B, C, E,
        [this](State& out, const T& time, const State& y){
            this->rhs(out, time, y);
        }
    );
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
StepResult RK45<T, SD, JD, OdeType, SP, Derived>::adapt_impl(T& t_new, T& habs_new, State& q_new, const T& t, const T& habs, const State& q){
    mat_is_set = false;
    // FSAL: the previous step's last stage becomes this step's first
    copy_state<SD>(K_[0], K_[Nstages]);
    return detail::rk_adapt_step<T, SD>(
        t_new, habs_new, q_new, t, habs, q,
        this->min_step(),
        this->max_step(),
        this->tiny_factor(),
        this->safety_factor(),
        this->max_factor(),
        this->min_factor(),
        ERR_EXP,
        INC_EXP,
        MIN_ERR,
        this->direction(),
        [this](T& t_out, State& q_out, const T& t_in, const T& h, const State& q_in){
            return this->step_impl(t_out, q_out, t_in, h, q_in);
        }
    );
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
void RK45<T, SD, JD, OdeType, SP, Derived>::interp_impl(State& result, const T& t) const{
    this->set_coef_matrix();
    coef_mat_interp<T, SD>(result, t, this->t_old(), this->t_new(), this->vector_old(), this->vector_new(), coef_mat.data(), INTERP_ORDER);
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
auto RK45<T, SD, JD, OdeType, SP, Derived>::local_interp() const{
    this->set_coef_matrix();
    return [cm=this->coef_mat, t1=this->t_old(), t2=this->t_new(), y1=this->vector_old(), y2=this->vector_new()](State& out, const T& t){
        coef_mat_interp<T, SD>(out, t, t1, t2, y1, y2, cm.data(), INTERP_ORDER);
    };
}

} // namespace ode

#endif // ODECRAFT_RK45_DOPRI_IMPL_HPP
