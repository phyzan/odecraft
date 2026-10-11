#ifndef ODECRAFT_RK23_DOPRI_IMPL_HPP
#define ODECRAFT_RK23_DOPRI_IMPL_HPP

#include "odecraft/Toolkit/Scratch.hpp"
#include <odecraft/Steppers/RK23_DOPRI.hpp>

namespace ode{

namespace detail{

template<size_t NSYS, typename T, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
XDIFF_FORCEINLINE T rk23_step_impl(T& t_new, T* __restrict__ q_new, const T& t, const T& h, const T* __restrict__ q, size_t nsys,
                 const T* K0, T* K, T* KF, T* r,
                 const T& rtol, const T& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs){
    const size_t n = NSYS ? NSYS : nsys;
    T* __restrict__ K1 = K;
    T* __restrict__ K2 = K + n;
    T* __restrict__ K3 = KF;

    // Stage 2
    for (size_t j = 0; j < n; j++) {
        r[j] = q[j] + h * (A(1,0)*K0[j]);
    }
    rhs(K1, t + C[1]*h, r);

    // Stage 3 (a31 = 0, so K0 absent)
    for (size_t j = 0; j < n; j++) {
        r[j] = q[j] + h * (A(2,1)*K1[j]);
    }
    rhs(K2, t + C[2]*h, r);

    // Solution update
    for (size_t j = 0; j < n; j++) {
        q_new[j] = q[j] + h * (B(0)*K0[j] + B(1)*K1[j] + B(2)*K2[j]);
    }

    // FSAL: K3 = f(t+h, q_new)
    rhs(K3, t + h, q_new);
    t_new = t + h;

    // Error norm calculation

    T* __restrict__ scaled_err = r;
    for (size_t j = 0; j < n; j++) {
        const auto err   = h * (E[0]*K0[j] + E[1]*K1[j] + E[2]*K2[j] + E[3]*K3[j]);
        const auto scale = atol + rtol * (abs(q[j]) + abs(K0[j] * h));
        scaled_err[j] = abs(err) / scale;
    }
    return *std::max_element(scaled_err, scaled_err + n);
}

} // namespace ode::detail


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
RK23<T, N, SP, OdeType, Derived>::RK23(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, EventList<T> events)
    : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, direction, std::move(events)),
      K0_(q0.size()), KF_(q0.size()), coef_mat_(q0.size(), INTERP_ORDER),
      rk_scratch_(request_scratch_page<T, Nstages*N>(Nstages*q0.size())),
      rk_input_scratch_(request_scratch_page<T, 2*N>(2*q0.size())),
      rk_scalar_scratch_(request_scratch_page<T, 1>()) {
    if (q0.data() != nullptr){
        this->rhs(KF_.data(), t0, q0.data());
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Stepper RK23<T, N, SP, OdeType, Derived>::method() const{
    return Stepper::RK23;
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::Reset(){
    Base::Reset();
    K0_.fill(0);
    KF_.fill(0);
    mat_is_set_ = false;
    this->rhs(KF_.data(), this->t_new(), this->vector_new().data());
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::ReAdjust(const T* new_vector){
    Base::ReAdjust(new_vector);
    // freeze the interpolation coefficients (from t_old to t) before KF_ is overwritten
    this->set_coef_matrix();
    this->rhs(KF_.data(), this->t(), new_vector);
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::set_coef_matrix() const{
    if (!mat_is_set_){
        // See RK45::set_coef_matrix: the stage scratch is transient, so the finished step is
        // replayed over the same interval (K0_ and h_last_ still describe it) into private
        // buffers, leaving KF_ - the next step's starting derivative - untouched.
        const size_t n = this->nsys();
        auto r_buf     = rk_input_scratch_.template request<N>(n);
        auto K_buf     = rk_scratch_.template request<N*(Nstages-1)>(n*(Nstages-1));
        auto kf_buf    = rk_scratch_.template request<N>(n);
        // the replayed step's end point is discarded, only its stages are kept
        auto t_buf = rk_scalar_scratch_.template request<1>(1);
        auto q_buf = rk_input_scratch_.template request<N>(n);

        detail::rk23_step_impl<N>(
            t_buf[0],
            q_buf.data(),
            this->t_old(),
            h_last_,
            this->vector_old().data(),
            n,
            K0_.data(),
            K_buf.data(),
            kf_buf.data(),
            r_buf.data(),
            this->rtol(),
            this->atol(),
            A, B, C, E,
            [this](T* out, const T& time, const T* y){
                this->rhs(out, time, y);
            }
        );

        detail::rk_interp_matrix(
            coef_mat_.data(),
            K_buf.data(),
            K0_.data(),
            kf_buf.data(),
            P.data(),
            Nstages,
            INTERP_ORDER,
            n
        );
        mat_is_set_ = true;
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T RK23<T, N, SP, OdeType, Derived>::step_impl(T& t_new, T* __restrict__ q_new, const T& t, const T& h, const T* __restrict__ q){
    auto rhs_scratch   = rk_input_scratch_.template request<N>(this->nsys());
    auto stage_scratch = rk_scratch_.template request<N*(Nstages-1)>(this->nsys()*(Nstages-1));
    h_last_ = h; // remembered so set_coef_matrix can replay this sweep exactly
    return detail::rk23_step_impl<N>(t_new, q_new, t, h, q, this->nsys(),
                                     K0_.data(), stage_scratch.data(), KF_.data(), rhs_scratch.data(),
                                     this->rtol(), this->atol(), A, B, C, E,
                                     [this](T* out, const T& time, const T* y){ this->rhs(out, time, y); });
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
StepResult RK23<T, N, SP, OdeType, Derived>::adapt_impl(T& t_new, T& habs_new, T* q_new, const T& t, const T& habs, const T* q){
    mat_is_set_ = false;
    // FSAL: the previous step's last stage becomes this step's first
    std::copy(KF_.data(), KF_.data() + this->nsys(), K0_.data());
    return detail::rk_adapt_step<N>(t_new, habs_new, q_new, t, habs, q, this->nsys(),
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
        [this](T& t_out, T* q_out, const T& t_in, const T& h, const T* q_in){
            return this->step_impl(t_out, q_out, t_in, h, q_in);
        }
    );
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK23<T, N, SP, OdeType, Derived>::interp_impl(T* result, const T& t) const{
    this->set_coef_matrix();
    coef_mat_interp(result, t, this->t_old(), this->t_new(), this->vector_old().data(), this->vector_new().data(), coef_mat_.data(), INTERP_ORDER, this->nsys());
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
auto RK23<T, N, SP, OdeType, Derived>::local_interp() const{
    this->set_coef_matrix();
    return [cm=this->coef_mat_, t1=this->t_old(), t2=this->t_new(), y1=Array1D<T, N>(this->vector_old().data(), this->nsys()), y2=Array1D<T, N>(this->vector_new().data(), this->nsys()), n=this->nsys()](T* out, const T& t){
        coef_mat_interp(out, t, t1, t2, y1.data(), y2.data(), cm.data(), INTERP_ORDER, n);
    };
}

} // namespace ode

#endif // ODECRAFT_RK23_DOPRI_IMPL_HPP
