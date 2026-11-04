#pragma once


#include "BDF.hpp"
#include "odecraft/Toolkit/Tools.hpp"


namespace ode{

namespace detail{

template<typename T>
BDFCONSTS<T>::BDFCONSTS(){
    KAPPA = {T(0), -T(185)/1000, -T(1)/9, -T(823)/10000, -T(415)/10000, T(0)};
    GAMMA[0] = 0;
    T cumulative{0};
    const T one{1};
    for (size_t i = 1; i < BDF_MAX_ORDER+1; ++i) {
        cumulative += one / i;
        GAMMA[i] = cumulative;
    }

    for (size_t i=0; i<BDF_MAX_ORDER+1; i++){
        ALPHA[i] = (1-KAPPA[i])*GAMMA[i];
        ERR_CONST[i] = KAPPA[i]*GAMMA[i] + T(1)/T(1UL+i);
    }
}

template<typename T>
void compute_R(BDFMatrix<T>& R, size_t order, const T& factor) {
    size_t n = order+1;
    // Initialize first row to 1
    for (size_t j=0; j<n; j++){
        R(0, j) = 1;
    }

    // Compute M[i,j] = (i-1 - factor*j) / i for i,j >= 1, then cumulative product
    for (size_t i = 1; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            R(i, j) = R(i-1, j)*(i-1 - factor * j) / i;
        }
    }
}

template<typename T, typename SD>
void bdf_interp(state_t<T, SD>& result, const T& t, const T& t2, const T& h, const std::array<state_t<T, SD>, BDF_MAX_ORDER+3>& D, size_t order){
    using S = value_t<T, SD>;

    // result = D[0] + sum_j D[j+1] * p_j, where p_j = prod_{m <= j} (t - (t2 - h*m)) / (h*(1+m))
    SD::for_each([](S& r_i){
        r_i = 0;
    }, result);

    T p = 1;
    for (size_t j=0; j<order; j++){
        p *= (t - (t2 - h*j))/(h*(1+j));
        SD::for_each([&p](S& r_i, const S& D_i){
            r_i += D_i * p;
        }, result, D[j+1]);
    }

    SD::for_each([](S& r_i, const S& D0_i){
        r_i = D0_i + r_i;
    }, result, D[0]);
}

} // namespace ode::detail


template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
BDF<T, SD, OdeType, SP, Derived>::BDF(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step, T max_step, T stepsize, int dir, EventList<T, State> events)
    : Base(std::move(ode), t0, q0, std::move(rtol), std::move(atol), min_step, max_step, stepsize, dir, std::move(events)),
      ypred_scratch_(q0), psi_scratch_(q0), d_scratch_(q0), work_scratch_(q0), scale_scratch_(this->rtol()) {

    // the Jacobians take the shape of q0
    SD::JD::resize_like(J_, q0);
    SD::JD::resize_like(A_, q0);
    jac_scratch_ = J_;

    // the differences take the shape of q0, reset_impl_alone gives them their values
    for (Differences& D : D_){
        D.fill(q0);
    }

    // The Newton tolerance follows the tightest relative tolerance, where a zero one counts as 100 eps
    const T rtol_floor = 100*std::numeric_limits<T>::epsilon();
    T rtol_min = inf<T>();
    [[maybe_unused]] bool zero_rtol = false; // only read for the warning
    SD::for_each([&](const T& rtol_i){
        if (rtol_i == 0){
            zero_rtol = true;
            set_min(rtol_min, rtol_min, rtol_floor);
        } else {
            set_min(rtol_min, rtol_min, rtol_i);
        }
    }, this->rtol());

#ifndef ODECRAFT_NO_WARN
    if (zero_rtol){
        this->cerr(GetStr("Warning: rtol=0 not allowed in the BDF method. Setting rtol = ", rtol_floor));
    }
#endif
    newton_tol_ = ndspan::max<T>(10 * std::numeric_limits<T>::epsilon() / rtol_min, ndspan::min<T>(T(3)/100, pow(rtol_min, T(1)/T(2))));

    if (this->is_running()){
        if (this->validate_ics_impl(t0, q0)){
            this->reset_impl_alone();
        }else{
            this->set_status(StepperStatus::BadICS);
        }
    }
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void BDF<T, SD, OdeType, SP, Derived>::ReAdjust(const State& new_vector) {
    using S = value_t<T, SD>;
    Base::ReAdjust(new_vector);
    for (Differences& D : D_){
        for (State& D_k : D){
            SD::for_each([](S& d){
                d = 0;
            }, D_k);
        }
    }
    copy_state<SD>(D_[0][0], new_vector);
    this->Rhs(D_[0][1], this->t(), new_vector);
    const T h = this->stepsize() * this->direction();
    SD::for_each([&h](S& f_i){
        f_i *= h;
    }, D_[0][1]);
    this->Jac(J_, this->t(), new_vector);
    order_ = 1;
    n_eq_steps_ = 0;
    valid_LU_ = false;
    idx_D_ = 0;
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
bool BDF<T, SD, OdeType, SP, Derived>::validate_ics_impl(const T& t0, const State& q0) const{
    if (!Base::validate_ics_impl(t0, q0)){
        return false;
    }

    decltype(auto) J = jac_scratch_.get();
    this->Jac(J, t0, q0);
    return jac_all_are_finite<T, SD>(J);
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
Stepper BDF<T, SD, OdeType, SP, Derived>::method() const {
    return Stepper::BDF;
}


template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void BDF<T, SD, OdeType, SP, Derived>::Reset(){
    Base::Reset();
    this->reset_impl_alone();
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void BDF<T, SD, OdeType, SP, Derived>::reset_impl_alone(){
    // Only runs right after the base has been reset to the initial conditions,
    // so stepsize() is still the initial step size.
    using S = value_t<T, SD>;
    const T& t0 = this->t_ics();
    const T h0 = this->stepsize() * this->direction();
    const State& q0 = this->vector_ics();
    decltype(auto) f = work_scratch_.get();
    this->Rhs(f, t0, q0);

    for (Differences& D : D_){
        for (State& D_k : D){
            SD::for_each([](S& d){
                d = 0;
            }, D_k);
        }
        copy_state<SD>(D[0], q0);
        SD::for_each([&h0](S& D1_i, const S& f_i){
            D1_i = f_i * h0;
        }, D[1], f);
    }
    this->Jac(J_, t0, q0);
    order_ = 1;
    n_eq_steps_ = 0;
    valid_LU_ = false;
    idx_D_ = 0;
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
StepResult BDF<T, SD, OdeType, SP, Derived>::adapt_impl(T& t_out, T& habs_out, State& q_out, const T& t, const T& stepsize, const State& q_in){
    using S = value_t<T, SD>;

    T safety, best_factor, factor, c, error_norm;
    int delta_order;
    bool converged;
    bool currentjac = false;
    bool step_accepted = false;
    detail::NewtConv conv_result;

    const cache_t<T> h_min = this->min_step();
    const cache_t<T> max_step = this->max_step();
    const RealState& atol = this->atol();
    const RealState& rtol = this->rtol();

    // ----------- Scratch pad ---------------
    auto  scratch    = make_scratch(t_out, habs_out, q_out);
    auto& [t_new, habs, y_new] = scratch;
    auto  write_back = make_write_back(scratch, t_out, habs_out, q_out);
    // ---------------------------------------

    decltype(auto) y_pred = ypred_scratch_.get();
    decltype(auto) psi    = psi_scratch_.get();
    decltype(auto) d      = d_scratch_.get();
    decltype(auto) work   = work_scratch_.get();
    decltype(auto) scale  = scale_scratch_.get();

    t_new = t;

    copy_state<SD>(y_new, q_in);

    if (stepsize > max_step){
        habs = max_step;
        this->change_D(max_step/stepsize);
    } else if (stepsize < h_min){
        habs = h_min;
        this->change_D(h_min/stepsize);
    } else {
        habs = stepsize;
    }

    while(!step_accepted){

        // Checks go at the top so that continue statements inside the loop will still trigger them.
        if (habs < h_min){
            return StepResult::MinStepError; // TODO: The algorithm does not work properly when we go below min_step.
        } else if (habs > max_step){
            this->change_D(max_step/habs);
            habs = max_step;
            return StepResult::Success; // The step is accepted, but the stepsize is limited by max_step.
        } else if (habs < this->tiny_factor()){
            return StepResult::TinyStepError;
        }

        t_new = t + habs * this->direction();

        this->set_prediction(y_pred);
        SD::for_each([](T& scale_i, const T& atol_i, const T& rtol_i, const S& y_pred_i){
            scale_i = atol_i + rtol_i * abs(y_pred_i);
        }, scale, atol, rtol, y_pred);
        this->set_psi(psi);

        converged = false;
        c = habs * this->direction() / coefs_.ALPHA[order_];
        while (!converged){
            if (!valid_LU_){
                // The Newton iterations solve with I - c*J
                SD::JD::linalg::identity_minus(A_, c, J_);
                if (!SD::JD::linalg::lu_factor(A_, piv_)){
                    return StepResult::SingularMatrixError;
                }
                valid_LU_ = true;
            }

            conv_result = this->solve_bdf_system(y_new, y_pred, d, work, t_new, c, psi, scale);
            if (conv_result.flag != StepResult::Success){
                return conv_result.flag;
            }
            converged = conv_result.converged;

            if (!converged){
                if (currentjac){
                    break;
                }
                this->Jac(J_, t_new, y_pred);
                valid_LU_ = false;
                currentjac = true;
            }
        }

        if (!converged){
            factor = T(1)/2;
            habs *= factor;
            this->change_D(factor);
            valid_LU_ = false;
            continue;
        }

        safety = T(18UL * NEWTON_MAXITER + 9UL)/(20UL * NEWTON_MAXITER + 10UL*conv_result.n_iter);
        const cache_t<T> err_const = coefs_.ERR_CONST[order_];
        SD::for_each([&err_const](T& scale_i, S& error_i, const T& atol_i, const T& rtol_i, const S& y_new_i, const S& d_i){
            scale_i = atol_i + rtol_i * abs(y_new_i);
            error_i = err_const * d_i;
        }, scale, work, atol, rtol, y_new, d);
        error_norm = rms_norm<SD, T>(work, scale);
        if (error_norm > 1){
            factor = ndspan::max<T>(this->min_factor(), safety * pow(error_norm, T(-1)/(order_+1)));
            habs *= factor;
            this->change_D(factor);
        } else {
            step_accepted = true;
        }
    }

    n_eq_steps_++;

    Differences& D = D_[idx_D_];

    SD::for_each([](S& D_p2, S& D_p1, const S& d_i){
        D_p2 = d_i - D_p1;
        D_p1 = d_i;
    }, D[order_+2], D[order_+1], d);
    for (size_t i = order_ + 1; i -- > 0;) {
        SD::for_each([](S& D_i, const S& D_i1){
            D_i += D_i1;
        }, D[i], D[i+1]);
    }

    if (n_eq_steps_ < order_ + 1){
        return StepResult::Success;
    }

    // the error norms of the orders order_-1, order_ and order_+1
    std::array<T, 3> error_norms;
    error_norms[1] = error_norm;

    if (order_ > 1){
        const cache_t<T> err_const = coefs_.ERR_CONST[order_-1];
        SD::for_each([&err_const](S& error_i, const S& D_i){
            error_i = err_const * D_i;
        }, work, D[order_]);
        error_norms[0] = rms_norm<SD, T>(work, scale);
    } else {
        error_norms[0] = inf<T>();
    }

    if (order_ < BDF_MAX_ORDER){
        const cache_t<T> err_const = coefs_.ERR_CONST[order_+1];
        SD::for_each([&err_const](S& error_i, const S& D_i){
            error_i = err_const * D_i;
        }, work, D[order_+2]);
        error_norms[2] = rms_norm<SD, T>(work, scale);
    }
    else{
        error_norms[2] = inf<T>();
    }

    delta_order = -1;
    best_factor = pow(error_norms[0], T(-1)/order_);
    for (size_t i=1; i<3; i++){
        T tmp = pow(error_norms[i], T(-1)/(order_+i));
        if (tmp > best_factor){
            best_factor = tmp;
            delta_order = static_cast<int>(i) - 1;
        }
    }

    order_ += delta_order;
    factor = ndspan::min<T>(this->max_factor(), safety * best_factor);
    habs *= factor;
    this->change_D(factor);
    valid_LU_ = false;
    return StepResult::Success;
}


template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
auto BDF<T, SD, OdeType, SP, Derived>::local_interp_impl() const{
    return make_light_functor([](State& out, const T& t, const std::array<state_t<T, SD>, BDF_MAX_ORDER + 3>& D, const T& t2, const T& habs, int d, size_t order){
        return detail::bdf_interp<T, SD>(out, t, t2, habs*d, D, order);
    },
    D_[idx_D_], this->t_new(), this->stepsize(), this->direction(), this->order_);
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
detail::NewtConv BDF<T, SD, OdeType, SP, Derived>::solve_bdf_system(State& y, const State& y_pred, State& d, State& work, const T& t_new, const T& c, const State& psi, const RealState& scale){
    using S = value_t<T, SD>;
    SD::for_each([](S& d_i){
        d_i = 0;
    }, d);
    copy_state<SD>(y, y_pred);
    T dy_norm = 0;
    T dy_norm_old = 0;
    T rate = 0;
    bool converged = false;
    size_t j=0;
    StepResult flag = StepResult::Success;
    for (size_t k=0; k<NEWTON_MAXITER; k++){
        // work holds f(t_new, y), then the right hand side c*f - psi - d, then the correction dy
        this->Rhs(work, t_new, y);
        SD::for_each([&c](S& w_i, const S& psi_i, const S& d_i){
            w_i = c * w_i - psi_i - d_i;
        }, work, psi, d);
        SD::JD::linalg::lu_solve(work, A_, piv_, work);
        dy_norm = rms_norm<SD, T>(work, scale);

        if (dy_norm_old == 0){
            rate = 0;
        } else {
            rate = dy_norm/dy_norm_old;
        }

        if (rate != 0 && (rate >= 1 || (pow(rate, NEWTON_MAXITER-k)*dy_norm > newton_tol_ * (1-rate)))){
            break;
        }

        SD::for_each([](S& y_i, S& d_i, const S& dy_i){
            y_i += dy_i;
            d_i += dy_i;
        }, y, d, work);

        if (!all_are_finite<T, SD>(y)){
            flag = StepResult::NonFiniteError;
            break;
        } else if (dy_norm == 0 || ((rate != 0) && (rate * dy_norm < newton_tol_ * (1-rate)))){
            converged = true;
            break;
        }

        dy_norm_old = dy_norm;
        j++;
    }

    return {.converged = converged, .n_iter = j+1, .flag = flag};

}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void BDF<T, SD, OdeType, SP, Derived>::change_D(const T& factor){
    using S = value_t<T, SD>;
    const size_t n = order_+1;
    detail::compute_R(R_, order_, factor);
    // U depends on the order alone
    if (U_order_ != order_){
        detail::compute_R(U_, order_, T(1));
        U_order_ = order_;
    }
    matmul(RU_, R_, U_, n);

    Differences& D_new = D_[1-idx_D_];
    const Differences& D = D_[idx_D_];

    // D_new[i] = sum_k RU(k, i) * D[k]
    for (size_t i = 0; i < n; i++) {
        SD::for_each([](S& out_j){
            out_j = 0;
        }, D_new[i]);

        for (size_t k = 0; k < n; k++) {
            const cache_t<T> r = RU_(k, i);
            SD::for_each([&r](S& out_j, const S& D_kj){
                out_j += r * D_kj;
            }, D_new[i], D[k]);
        }
    }

    idx_D_ = 1 - idx_D_;
    n_eq_steps_ = 0;
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void BDF<T, SD, OdeType, SP, Derived>::set_prediction(State& y) const{
    using S = value_t<T, SD>;
    const Differences& D = D_[idx_D_];

    SD::for_each([](S& y_i){
        y_i = 0;
    }, y);

    for (size_t i=0; i < order_+1; i++){
        SD::for_each([](S& y_i, const S& D_i){
            y_i += D_i;
        }, y, D[i]);
    }
}

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void BDF<T, SD, OdeType, SP, Derived>::set_psi(State& psi) const{
    using S = value_t<T, SD>;
    const Differences& D = D_[idx_D_];
    const cache_t<T> a = coefs_.ALPHA[order_];

    SD::for_each([](S& psi_i){
        psi_i = 0;
    }, psi);
    for (size_t i=1; i<order_+1; i++){
        const cache_t<T> g = coefs_.GAMMA[i];
        SD::for_each([&g, &a](S& psi_i, const S& D_i){
            psi_i += D_i * g / a;
        }, psi, D[i]);
    }
}

} // namespace ode
