#ifndef ODECRAFT_RK45_DOPRI_IMPL_HPP
#define ODECRAFT_RK45_DOPRI_IMPL_HPP

#include <odecraft/Steppers/RK45_DOPRI.hpp>

namespace ode{

namespace detail{

template<size_t NSYS, typename T, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
XDIFF_FORCEINLINE T rk45_step_impl(T* result, const T* state, const T& h, size_t nsys,
                 const T* K0, T* K, T* KF, T* r,
                 const T& rtol, const T& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs){
    // NSYS is a constant, so this folds to a compile-time bound for fixed-size systems and
    // keeps the loops unrolled whether or not the call itself gets inlined.
    const size_t n = NSYS ? NSYS : nsys;
    const T& t = state[0];
    T* __restrict__       q_new = result + 2;
    const T* __restrict__ q     = state + 2;

    T* __restrict__ K1 = K;
    T* __restrict__ K2 = K +   n;
    T* __restrict__ K3 = K + 2*n;
    T* __restrict__ K4 = K + 3*n;
    T* __restrict__ K5 = K + 4*n;
    T* __restrict__ K6 = KF;

    // Stage 2
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(1,0)*K0[j]); }
    rhs(K1, t + C[1]*h, r);

    // Stage 3
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(2,0)*K0[j] + A(2,1)*K1[j]); }
    rhs(K2, t + C[2]*h, r);

    // Stage 4
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(3,0)*K0[j] + A(3,1)*K1[j] + A(3,2)*K2[j]); }
    rhs(K3, t + C[3]*h, r);

    // Stage 5
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(4,0)*K0[j] + A(4,1)*K1[j] + A(4,2)*K2[j] + A(4,3)*K3[j]); }
    rhs(K4, t + C[4]*h, r);

    // Stage 6
    for (size_t j = 0; j < n; j++) { r[j] = q[j] + h * (A(5,0)*K0[j] + A(5,1)*K1[j] + A(5,2)*K2[j] + A(5,3)*K3[j] + A(5,4)*K4[j]); }
    rhs(K5, t + h, r);

    // Solution update (b2 = 0)
    for (size_t j = 0; j < n; j++) {
        q_new[j] = q[j] + h * (B(0)*K0[j] + B(2)*K2[j] + B(3)*K3[j] + B(4)*K4[j] + B(5)*K5[j]);
    }

    // FSAL: K6 = f(t+h, q_new)
    rhs(K6, t + h, q_new);
    result[0] = t + h;

    // Error norm (e2 = 0; scale uses the derivative at the start of the step)
    T err_max = 0;
    for (size_t j = 0; j < n; j++) {
        const auto err   = h * (E[0]*K0[j] + E[2]*K2[j] + E[3]*K3[j] + E[4]*K4[j] + E[5]*K5[j] + E[6]*K6[j]);
        const auto scale = atol + rtol * (abs<T>(q[j]) + abs<T>(K0[j] * h));
        err_max = ndspan::max<T>(err_max, abs<T>(err) / scale);
    }
    return err_max;
}


} // namespace ode::detail


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
RK45<T, N, SP, OdeType, Derived>::RK45(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, EventList<T> events)
    : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, direction, std::move(events)),
      scratch_space(q0.size()),
      K0_(q0.size()),
      KF_(q0.size()), coef_mat(q0.size(), INTERP_ORDER) {
    if (q0.data() != nullptr){
        this->rhs(KF_.data(), t0, q0.data());
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Stepper RK45<T, N, SP, OdeType, Derived>::method() const{
    return Stepper::RK45;
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::Reset(){
    Base::Reset();
    K0_.fill(0);
    KF_.fill(0);
    mat_is_set = false;
    const T* state = this->new_state_ptr();
    this->rhs(KF_.data(), state[0], state+2);
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::ReAdjust(const T* new_vector){
    Base::ReAdjust(new_vector);
    this->set_coef_matrix();
    this->rhs(KF_.data(), this->t(), new_vector);
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::set_coef_matrix() const{
    if (!mat_is_set){
        // The stages of the step just taken are needed to build the dense-output polynomial,
        // but the step's stage scratch is transient, so replay the sweep over the same
        // interval. K0_ still holds the derivative at the start of that step and h_last_ the
        // step size it used, so the replay reproduces the stages exactly. It writes into its
        // own result/FSAL scratch, leaving KF_ - the next step's starting derivative - alone.
        // Cost: Nstages RHS evaluations, paid only when an interpolation is actually asked for.
        const size_t n = this->nsys();
        decltype(auto) r_buf     = scratch_space.rhs_scratch();
        decltype(auto) K_buf     = scratch_space.stage_scratch();
        decltype(auto) kf_buf    = scratch_space.fsal_scratch();
        decltype(auto) state_buf = scratch_space.state_scratch();

        detail::rk45_step_impl<N>(state_buf.data(), this->old_state_ptr(), h_last_, n,
                               K0_.data(), K_buf.data(), kf_buf.data(), r_buf.data(),
                               this->rtol(), this->atol(), A, B, C, E,
                               [this](T* out, const T& time, const T* y){ this->rhs(out, time, y); });

        detail::rk_interp_matrix(coef_mat.data(), K_buf.data(), K0_.data(), kf_buf.data(),
                                 P.data(), Nstages, INTERP_ORDER, n);
        mat_is_set = true;
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T RK45<T, N, SP, OdeType, Derived>::step_impl(T* result, const T* state, const T& h){
    decltype(auto) rhs_scratch   = scratch_space.rhs_scratch();
    decltype(auto) stage_scratch = scratch_space.stage_scratch();
    h_last_ = h; // remembered so set_coef_matrix can replay this sweep exactly
    return detail::rk45_step_impl<N>(result, state, h, this->nsys(),
                                  K0_.data(), stage_scratch.data(), KF_.data(), rhs_scratch.data(),
                                  this->rtol(), this->atol(), A, B, C, E,
                                  [this](T* out, const T& time, const T* y){ this->rhs(out, time, y); });
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
StepResult RK45<T, N, SP, OdeType, Derived>::adapt_impl(T* res, const T* state){
    mat_is_set = false;
    std::copy(KF_.data(), KF_.data() + this->nsys(), K0_.data());
    return detail::rk_adapt_step(res, state, this->nsys(),
                          this->min_step(), this->max_step(), this->MIN_STEP,
                          this->SAFETY, this->MAX_FACTOR, this->MIN_FACTOR,
                          ERR_EXP, INC_EXP, MIN_ERR, this->direction(),
                          [this](T* r, const T* s, const T& h){ return this->step_impl(r, s, h); });
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK45<T, N, SP, OdeType, Derived>::interp_impl(T* result, const T& t) const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    coef_mat_interp(result, t, this->t_old(), d[0], this->old_state_ptr()+2, d+2, coef_mat.data(), INTERP_ORDER, this->nsys());
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
auto RK45<T, N, SP, OdeType, Derived>::local_interp() const{
    this->set_coef_matrix();
    const T* d = this->interp_new_state_ptr();
    return [cm=this->coef_mat, t1=this->t_old(), t2=d[0], y1=Array1D<T, N>(this->old_state_ptr()+2, this->nsys()), y2=Array1D<T, N>(d+2, this->nsys()), n=this->nsys()](T* out, const T& t){
        coef_mat_interp(out, t, t1, t2, y1.data(), y2.data(), cm.data(), INTERP_ORDER, n);
    };
}

} // namespace ode

#endif // ODECRAFT_RK45_DOPRI_IMPL_HPP
