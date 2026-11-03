#pragma once

#include <odecraft/Chaos/VariationalSolvers.hpp>
#include "VariationalSolvers_mem_impl.hpp" // IWYU pragma: keep


namespace ode::chaos{

// ============================ VariationalODE ==============================

template<typename T, typename SD, typename OdeType>
VariationalOdeSys<T, SD, OdeType>::VariationalOdeSys(OdeType ode, const State& q0, const RealState& atol) : ode_(std::move(ode)), atol_(atol.main),
    x_scratch_(q0.main), y1_scratch_(q0.main), y2_scratch_(q0.main), col_scratch_(q0.main) {

    if constexpr (AD_RHS){
        decltype(auto) duals = dual1_scratch_.get();
        decltype(auto) seeds = seed1_scratch_.get();
        SD::resize_like(duals, q0.main);
        SD::resize_like(seeds, q0.main);
    }

    if constexpr (AD_DIR || AD_JAC_DIR){
        decltype(auto) dir_in = dir_in_scratch_.get();
        SD::resize_like(dir_in, q0.main);
    }

    if constexpr (AD_DIR){
        decltype(auto) dir_out = dir_out_scratch_.get();
        SD::resize_like(dir_out, q0.main);
    }

    if constexpr (AD_JAC_DIR){
        decltype(auto) duals = jac_out_scratch_.get();
        decltype(auto) seeds = jac_seed_scratch_.get();
        SD::resize_like(duals, q0.main);
        SD::resize_like(seeds, q0.main);
    }

    if constexpr (JAC_RHS){
        MainJacMat J{};
        SD::JD::resize_like(J, q0.main);
        jm_scratch_ = J;
    }

    if constexpr (AD_JAC){
        decltype(auto) duals = dual2_scratch_.get();
        decltype(auto) seeds = seed2_scratch_.get();
        SD::resize_like(duals, q0.main);
        SD::resize_like(seeds, q0.main);
    }

    const char* err_msg = "Variational ODE systems that do not provide a Jacobian function for the main ODE system require a non-zero `atol` to compute the stepsize for finite differences";

    bool findiff_jac = (JP_MAIN == JacPolicy::Approx);
    if constexpr (JP_MAIN == JacPolicy::Nullable){
        findiff_jac = (ode_.Jac == nullptr);
    }
    if (findiff_jac && !SD::all_true([](const T& a){ return a != 0; }, atol_)){
        throw std::runtime_error(err_msg);
    }
}


// ----------------------------------------------------------------------------
// VariationalOdeSys accessors
// ----------------------------------------------------------------------------

template<typename T, typename SD, typename OdeType>
const OdeType& VariationalOdeSys<T, SD, OdeType>::ode() const{
    return ode_;
}



// ----------------------------------------------------------------------------
// VariationalOdeSys Rhs
// ----------------------------------------------------------------------------

// inline: otherwise GCC keeps it out of line at -O3, and the steppers pay for the call on every evaluation
template<typename T, typename SD, typename OdeType>
inline void VariationalOdeSys<T, SD, OdeType>::Rhs(State& out, const T& t, const State& q) const{
    using S = value_t<T, SD>;
    if constexpr (AD_DIR){
        // One main Rhs call with duals that carry a single derivative, along delta_q: for
        // q + eps*delta_q, f_i is the value of each, and (J*delta_q)_i its derivative
        using D = detail::dir_dual_t<T, SD>;
        decltype(auto) q_dir = dir_in_scratch_.get();
        decltype(auto) f_dir = dir_out_scratch_.get();
        SD::for_each([](D& x_i, const S& q_i, const S& dq_i){
            x_i = D(q_i, typename D::state_type{dq_i});
        }, q_dir, q.main, q.variation);
        ode_.Rhs(f_dir, t, q_dir);
        SD::for_each([](S& f_i, S& df_i, const D& y_i){
            f_i = y_i.value();
            df_i = y_i.grad()[0];
        }, out.main, out.variation, f_dir);
    } else if constexpr (AD_RHS){
        // One main Rhs call with 1st order duals: f_i is the value of each, and (J*delta_q)_i
        // its gradient dotted with delta_q, without forming J
        decltype(auto) out_duals = dual1_scratch_.get();
        decltype(auto) seeds = seed1_scratch_.get();
        xdiff::set_seeds<T, SD>(seeds, q.main);
        ode_.Rhs(out_duals, t, seeds);
        SD::for_each([&q](S& f_i, S& df_i, const dual_t<T, SD, 1>& fi){
            f_i = fi.value();
            df_i = 0;
            SD::for_each([&df_i](const S& dfi_dqj, const S& dq_j){
                df_i += dfi_dqj * dq_j;
            }, fi.grad(), q.variation);
        }, out.main, out.variation, out_duals);
    } else {
        ode_.Rhs(out.main, t, q.main); // Main system filled

        // The variational part: J times delta_q if J is given, else the difference along delta_q
        if constexpr (JAC_RHS){
            if constexpr (JP_MAIN == JacPolicy::Nullable){
                if (ode_.Jac == nullptr){
                    this->jvp_findiffs(out.variation, t, q);
                    return;
                }
            }
            decltype(auto) J = jm_scratch_.get();
            ode_.Jac(J, t, q.main);
            SD::JD::linalg::dot(out.variation, J, q.variation);
        } else {
            this->jvp_findiffs(out.variation, t, q);
        }
    }
}


// ----------------------------------------------------------------------------
// VariationalOdeSys Jac
// ----------------------------------------------------------------------------

template<typename T, typename SD, typename OdeType>
void VariationalOdeSys<T, SD, OdeType>::Jac(JacMat& out, const T& t, const State& q) const{
    using S = value_t<T, SD>;
    if constexpr (AD_JAC_DIR){
        // One main Rhs call with duals over the main state, whose scalars carry the derivative
        // along delta_q. The gradient of f_i holds row i of J in its values, and row i of
        // dJ/dq*delta_q in their derivatives: no second derivatives along other directions
        using D = detail::dir_dual_t<T, SD>;
        decltype(auto) q_dir = dir_in_scratch_.get();
        decltype(auto) seeds = jac_seed_scratch_.get();
        decltype(auto) out_duals = jac_out_scratch_.get();
        SD::for_each([](D& x_i, const S& q_i, const S& dq_i){
            x_i = D(q_i, typename D::state_type{dq_i});
        }, q_dir, q.main, q.variation);
        xdiff::set_seeds<D, SD, 1>(seeds, q_dir);
        ode_.Rhs(out_duals, t, seeds);

        const auto row_i = [](const detail::jac_dual_t<T, SD>& fi){
            return std::tie(fi.grad());
        };
        SD::JD::for_each_row([](S& J_ij, const D& dfi_dqj){
            J_ij = dfi_dqj.value();
        }, row_i, std::tie(out[0]), out_duals);

        SD::JD::for_each_row([](S& B_ij, const D& dfi_dqj){
            B_ij = dfi_dqj.grad()[0];
        }, row_i, std::tie(out[1]), out_duals);
    } else if constexpr (AD_JAC){
        // One main Rhs call with 2nd order duals. The gradient of f_i holds row i of J in its
        // values, and row i of dJ/dq*delta_q in its own gradients, dotted with delta_q
        decltype(auto) out_duals = dual2_scratch_.get();
        decltype(auto) seeds = seed2_scratch_.get();
        xdiff::set_seeds<T, SD>(seeds, q.main);
        ode_.Rhs(out_duals, t, seeds);

        const auto row_i = [](const dual_t<T, SD, 2>& fi){
            return std::tie(fi.grad());
        };
        SD::JD::for_each_row([](S& J_ij, const dual_t<T, SD, 1>& dfi_dqj){
            J_ij = dfi_dqj.value();
        }, row_i, std::tie(out[0]), out_duals);

        SD::JD::for_each_row([&q](S& B_ij, const dual_t<T, SD, 1>& dfi_dqj){
            B_ij = 0;
            SD::for_each([&B_ij](const S& d2fi_dqj_dqk, const S& dq_k){
                B_ij += d2fi_dqj_dqk * dq_k;
            }, dfi_dqj.grad(), q.variation);
        }, row_i, std::tie(out[1]), out_duals);
    } else {
        this->JacMain(out[0], t, q.main);
        this->delta_J(out[1], t, q);
    }
}

template<typename T, typename SD, typename OdeType>
void VariationalOdeSys<T, SD, OdeType>::JacMain(MainJacMat& out, const T& t, const MainState& q) const{
    using S = value_t<T, SD>;
    if constexpr (JP_MAIN == JacPolicy::Exact){
        ode_.Jac(out, t, q);
    } else if constexpr (JP_MAIN == JacPolicy::Nullable){
        if (ode_.Jac != nullptr){
            ode_.Jac(out, t, q);
        } else { // Finite differences fallback
            this->jacmat_findiffs(out, t, q);
        }
    } else if constexpr (JP_MAIN == JacPolicy::Autodiff){
        decltype(auto) out_duals = dual1_scratch_.get();
        decltype(auto) seeds = seed1_scratch_.get();
        xdiff::set_seeds<T, SD>(seeds, q);
        ode_.Rhs(out_duals, t, seeds);

        // Row i of the Jacobian is the gradient of f_i
        SD::JD::for_each_row([](S& out_ij, const S& dfi_dqj){
            out_ij = dfi_dqj;
        }, [](const dual_t<T, SD, 1>& fi){
            return std::tie(fi.grad());
        }, std::tie(out), out_duals);
    } else { // Finite differences fallback
        this->jacmat_findiffs(out, t, q);
    }
}


// ----------------------------------------------------------------------------
// VariationalOdeSys finite differences
// ----------------------------------------------------------------------------

template<typename T, typename SD, typename OdeType>
void VariationalOdeSys<T, SD, OdeType>::jacmat_findiffs(MainJacMat& mat, const T& t, const MainState& q) const{
    decltype(auto) x = x_scratch_.get();
    decltype(auto) y1 = y1_scratch_.get();
    decltype(auto) y2 = y2_scratch_.get();
    jac_approx_autostep<T, SD>(
        [this](MainState& out_, const T& t_, const MainState& q_){
            this->ode_.Rhs(out_, t_, q_);
        },
        mat, x, y1, y2, t, q, atol_
    );
}

template<typename T, typename SD, typename OdeType>
void VariationalOdeSys<T, SD, OdeType>::jvp_findiffs(MainState& out, const T& t, const State& q) const{
    // A central difference along delta_q, with the step that moves q by sqrt(EPS) of its scale,
    // max(atol, |q|) in the rms sense, as each column of jac_approx does along one axis
    using S = value_t<T, SD>;
    const T EPS_SQRT = sqrt(std::numeric_limits<T>::epsilon());

    T scale_sqr = 0;
    T dq_sqr = 0;
    SD::for_each([&scale_sqr, &dq_sqr](const S& q_i, const S& dq_i, const T& atol_i){
        const T w_i = max_ref(atol_i, T(abs(q_i)));
        scale_sqr += w_i*w_i;
        dq_sqr += SD::norm_sqr(dq_i);
    }, q.main, q.variation, atol_);

    if (dq_sqr == 0){
        SD::for_each([](S& out_i){
            out_i = 0;
        }, out);
        return;
    }

    const T h = EPS_SQRT * sqrt(scale_sqr/dq_sqr);
    decltype(auto) x = x_scratch_.get();
    decltype(auto) y = y1_scratch_.get();

    SD::for_each([&h](S& x_i, const S& q_i, const S& dq_i){
        x_i = q_i + h*dq_i;
    }, x, q.main, q.variation);
    ode_.Rhs(out, t, x);

    SD::for_each([&h](S& x_i, const S& q_i, const S& dq_i){
        x_i = q_i - h*dq_i;
    }, x, q.main, q.variation);
    ode_.Rhs(y, t, x);

    const T two_h = 2*h;
    SD::for_each([&two_h](S& out_i, const S& y_i){
        out_i = (out_i - y_i) / two_h;
    }, out, y);
}

template<typename T, typename SD, typename OdeType>
void VariationalOdeSys<T, SD, OdeType>::delta_J(MainJacMat& mat, const T& t, const State& q) const{
    /*
    Column j of the block is d/dq_j of J*delta_q: the mixed second derivative of f along the axis j
    and along delta_q. A 4-point stencil over these two directions gives it in 4 rhs evaluations,
    so the block takes 4n of them.
    */
    using S = value_t<T, SD>;

    const T EPS = std::numeric_limits<T>::epsilon();
    // Second derivatives amplify roundoff by 1/h^2, so the sqrt(EPS) step used for
    // first derivatives (see jac_approx) would leave no signal at all here:
    // the roundoff error would be ~ EPS/h^2 = O(1). The optimal step for a
    // second difference balances truncation O(h^2) against roundoff O(EPS/h^2),
    // giving h ~ EPS^(1/4) and an error of ~ sqrt(EPS).
    const T EPS_4TH = sqrt(sqrt(EPS));

    decltype(auto) x = x_scratch_.get();
    decltype(auto) y1 = y1_scratch_.get();
    decltype(auto) y2 = y2_scratch_.get();
    decltype(auto) col = col_scratch_.get();

    // The step along delta_q moves q by EPS^(1/4) of its scale, max(atol, |q|) in the rms sense
    T scale_sqr = 0;
    T dq_sqr = 0;
    SD::for_each([&scale_sqr, &dq_sqr](const S& q_i, const S& dq_i, const T& atol_i){
        const T w_i = max_ref(atol_i, T(abs(q_i)));
        scale_sqr += w_i*w_i;
        dq_sqr += SD::norm_sqr(dq_i);
    }, q.main, q.variation, atol_);

    if (dq_sqr == 0){
        // J*delta_q is zero for every q, and so is its derivative
        SD::for_each([](S& col_i){
            col_i = 0;
        }, col);
        SD::JD::for_each_column([](S& mat_ij, const S& col_i){
            mat_ij = col_i;
        }, [&col](const S& /*qj*/){
            return std::tie(col);
        }, std::tie(mat), q.main);
        return;
    }

    const T eps = EPS_4TH * sqrt(scale_sqr/dq_sqr);

    // Column j: [f(+h, +eps) - f(-h, +eps) - f(+h, -eps) + f(-h, -eps)] / (4*h*eps),
    // with f(a, b) evaluated at q + a*e_j + b*delta_q
    SD::JD::for_each_column([](S& mat_ij, const S& col_i){
        mat_ij = col_i;
    }, [&](S& xj, const S& qj, const T& atol_j){

        const T h_j = EPS_4TH * max_ref(atol_j, T(abs(qj)));

        // Along +delta_q
        SD::for_each([&eps](S& x_i, const S& q_i, const S& dq_i){
            x_i = q_i + eps*dq_i;
        }, x, q.main, q.variation);
        const S xj_plus = xj;
        xj = xj_plus + h_j;
        ode_.Rhs(y1, t, x);
        xj = xj_plus - h_j;
        ode_.Rhs(y2, t, x);
        SD::for_each([](S& col_i, const S& y1_i, const S& y2_i){
            col_i = y1_i - y2_i;
        }, col, y1, y2);

        // Along -delta_q
        SD::for_each([&eps](S& x_i, const S& q_i, const S& dq_i){
            x_i = q_i - eps*dq_i;
        }, x, q.main, q.variation);
        const S xj_minus = xj;
        xj = xj_minus + h_j;
        ode_.Rhs(y1, t, x);
        xj = xj_minus - h_j;
        ode_.Rhs(y2, t, x);

        const T four_h_eps = 4*h_j*eps;
        SD::for_each([&four_h_eps](S& col_i, const S& y1_i, const S& y2_i){
            col_i = (col_i - y1_i + y2_i) / four_h_eps;
        }, col, y1, y2);

        return std::tie(col);
    }, std::tie(mat), x, q.main, atol_);
}
// ==========================================================================


// ============================ VariationalODE ==============================

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
VariationalSolver<S, T, SD, OdeType, SP, Derived>::VariationalSolver(OdeType ode, T t0, State q0, T period, RealState rtol, RealState atol, T min_step, T max_step, T stepsize, int dir, EventList<T, State> events) :
    Base(VariationalOdeSys<T, SD, OdeType>(std::move(ode), q0, atol), t0, normalized_ics(q0), std::move(rtol), atol, min_step, max_step, stepsize, dir, std::move(events)),
    flag_state_(this->vector_ics()), current_scratch_(this->vector_ics()), period_(period), t_next_(t0+period*dir), t_last_(t0) {

    if (period <= 0){
        throw std::runtime_error("The renormalization period must be positive");
    }

    if constexpr (is_rich<SP>){
        //make sure there are no masked events, as they would interfere with the renormalization times.
        for (size_t i=0; i<this->event_col().size(); i++){
            if (this->event_col().event(i).is_masked()){
                throw std::runtime_error("VariationalSolver does not support masked events, as they would interfere with the renormalization times.");
            }
        }
    }
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
auto VariationalSolver<S, T, SD, OdeType, SP, Derived>::normalized_ics(State q0) -> State {
    detail::normalized<T, SD>(q0.variation, q0.variation);
    return q0;
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::elapsed_time() const{
    return this->t() - this->t_ics();
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::stretching_number() const{
    if (flagged){
        // The deviation vector was just renormalized, so the new period starts at unit norm.
        // The solver holds the old one until the next Adv_Impl applies it.
        return T(0);
    }
    // TODO : cache the stretching number, since calling fill_current_vector interpolates every time.
    decltype(auto) q = current_scratch_.get();
    this->fill_current_vector(q);
    return log(detail::state_norm<T, SD>(q.variation));
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::kick() const{
    if (flagged){
        // At the renormalization time t() == t_last_: the kick of the period that just ended
        return (logksi_ - logksi_last_)/(period_*this->direction());
    }
    return stretching_number()/(this->t() - t_last_);
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::period() const{
    return period_;
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::log_ksi() const{
    return logksi_ + this->stretching_number();
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::lyapunov_exponent() const{
    return np == 0 ? T{0} : T(log_ksi()/elapsed_time());
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void VariationalSolver<S, T, SD, OdeType, SP, Derived>::Reset(){
    Base::Reset();
    t_last_ = this->t_ics();
    t_next_ = t_last_ + period_*this->direction();
    np = 0;
    flagged = false;
    logksi_ =  0;
    logksi_last_ = 0;
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
bool VariationalSolver<S, T, SD, OdeType, SP, Derived>::SetIcs(T t0, State y0, T stepsize, int direction){
    // The norm of y0 must be non-zero
    if (SD::all_true([](const value_t<T, SD>& v){
        // value_t itself is not guaranteed to be comparable to T
        return SD::norm_sqr(v) == 0;
    }, y0.variation)){
        return false;
    } else {
        return Base::SetIcs(std::move(t0), normalized_ics(std::move(y0)), std::move(stepsize), direction);
    }
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void VariationalSolver<S, T, SD, OdeType, SP, Derived>::RhsMain(MainState& out, const T& t, const MainState& q) const{
    this->ode().ode().Rhs(out, t, q);
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void VariationalSolver<S, T, SD, OdeType, SP, Derived>::JacMain(MainJacMat& out, const T& t, const MainState& q) const{
    // The dispatch of VariationalOdeSys: an explicitly provided Jacobian is used
    // whenever it exists, and a Nullable one that is null at runtime falls back to
    // finite differences rather than being called.
    this->ode().JacMain(out, t, q);
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void VariationalSolver<S, T, SD, OdeType, SP, Derived>::ReAdjust(const State& /*new_vector*/){
    // assertion, not an error, as the ReAdjust member function is restricted to the internal machinery and not intended for public use.
    assert(false && "ReAdjust is not supported in VariationalSolver because it would interfere with the renormalization process.");
}

// ----------------------------------------------------------------------------
// VIRTUAL INTERFACE ALIASES
// Overrides of the ChaoticSolver pure virtuals, forwarding to the non-virtual
// accessors above so that internal callers never pay for a virtual dispatch.
// ----------------------------------------------------------------------------

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void VariationalSolver<S, T, SD, OdeType, SP, Derived>::get_rhs_main(MainState& out, const T& t, const MainState& q) const{
    RhsMain(out, t, q);
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
void VariationalSolver<S, T, SD, OdeType, SP, Derived>::get_jac_main(MainJacMat& out, const T& t, const MainState& q) const{
    JacMain(out, t, q);
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::get_elapsed_time() const{
    return elapsed_time();
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::get_kick() const{
    return kick();
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::get_period() const{
    return period();
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::get_log_ksi() const{
    return log_ksi();
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::get_lyapunov_exponent() const{
    return lyapunov_exponent();
}

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
T VariationalSolver<S, T, SD, OdeType, SP, Derived>::get_stretching_number() const{
    return stretching_number();
}


template<UtilPolicy UP, typename T, typename SD, typename OdeType, typename... Args>
pbox::Box<ChaoticSolver<T, SD, UP>> make_variational_solver(Stepper method, OdeType ode, T t0, state_t<T, VarSD<SD>> q0, T period, Args&&... args){

    constexpr SolverPolicy SP = UP == UtilPolicy::Virtual ? SolverPolicy::Virtual : SolverPolicy::RichVirtual;

    return choose_integrator_case<pbox::Box<ChaoticSolver<T, SD, UP>>>(method,
        [&]<Stepper S>(){
            return pbox::make_box<VariationalSolver<S, T, SD, OdeType, SP>>(std::move(ode), std::move(t0), std::move(q0), std::move(period), std::forward<Args>(args)...);
        }
    );
}


template<SolverPolicy SP, Stepper S, typename T, typename SD, typename OdeType>
auto getVariationalSolver(OdeType ode, T t0, state_t<T, VarSD<SD>> q0, T period, real_state_t<T, VarSD<SD>> rtol, real_state_t<T, VarSD<SD>> atol, T min_step, T max_step, T stepsize, int direction, EventList<T, state_t<T, VarSD<SD>>> events){
    return VariationalSolver<S, T, SD, OdeType, SP, void>(std::move(ode), std::move(t0), std::move(q0), std::move(period), std::move(rtol), std::move(atol), min_step, max_step, stepsize, direction, std::move(events));
}


template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
template<typename... Args>
bool VariationalSolver<S, T, SD, OdeType, SP, Derived>::Adv_Impl(Args&&... args) {
    if (flagged){
        Base::ReAdjust(flag_state_);
        flagged = false;
    }

    const int d = this->direction();
    const bool success = Base::Adv_Impl(t_next_, std::forward<Args>(args)...);
    if (success && (this->t() == t_next_)){
        t_last_ = t_next_;
        t_next_ = this->t_ics() + (++np + 1UL)*period_*d;
        this->fill_current_vector(flag_state_);
        logksi_last_ = logksi_;
        logksi_ += log(detail::state_norm<T, SD>(flag_state_.variation));
        detail::normalized<T, SD>(flag_state_.variation, flag_state_.variation);
        flagged = true;
        return true;
    } else if (success){
        return true;
    } else {
        return false;
    }
}
// ==========================================================================


// ============================ VariationalODE ==============================

template<typename T, typename SD>
std::unique_ptr<ODE<T, VarSD<SD>>> VariationalODE<T, SD>::clone() const{
    return std::make_unique<VariationalODE<T, SD>>(*this);
}

template<typename T, typename SD>
const std::vector<T>& VariationalODE<T, SD>::renorm_times() const{
    return renorm_times_;
}

template<typename T, typename SD>
const std::vector<T>& VariationalODE<T, SD>::lyap_values() const{
    return lyap_values_;
}

template<typename T, typename SD>
const std::vector<T>& VariationalODE<T, SD>::kick_values() const{
    return kick_values_;
}

template<typename T, typename SD>
void VariationalODE<T, SD>::clear(){
    Base::clear();
    renorm_times_ = std::vector<T>{};
    lyap_values_ = std::vector<T>{};
    kick_values_ = std::vector<T>{};
}

template<typename T, typename SD>
void VariationalODE<T, SD>::reset(){
    Base::reset();
    renorm_times_ = std::vector<T>{};
    lyap_values_ = std::vector<T>{};
    kick_values_ = std::vector<T>{};
}

template<typename T, typename SD>
const ChaoticSolver<T, SD, UtilPolicy::RichVirtual>* VariationalODE<T, SD>::solver() const {
    return static_cast<const ChaoticSolver<T, SD, UtilPolicy::RichVirtual>*>(Base::solver());
}

template<typename T, typename SD>
void VariationalODE<T, SD>::register_state(){
    Base::register_state();
    renorm_times_.push_back(this->solver()->get_time());
    lyap_values_.push_back(this->solver()->get_lyapunov_exponent());
    kick_values_.push_back(this->solver()->get_kick());
}
// ==========================================================================

} // namespace ode::chaos


// TODO Inlining seems to speedup some functions noticably