#pragma once

#include <odecraft/Chaos/VariationalSolvers.hpp>
#include <odecraft/Toolkit/Tools.hpp>
#include <odecraft/Toolkit/FinDiff.hpp>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "VariationalSolvers_mem_impl.hpp" // IWYU pragma: keep


namespace ode::chaos{

template<typename T, typename SD, typename OdeType>
VariationalOdeSys<T, SD, OdeType>::VariationalOdeSys(OdeType ode, const State& q0, const RealState& atol) : ode_(std::move(ode)), atol_(atol.main),
    x_scratch_(q0.main), y1_scratch_(q0.main), y2_scratch_(q0.main), f0_scratch_(q0.main), col_scratch_(q0.main) {

    if constexpr (AD_RHS){
        decltype(auto) duals = dual1_scratch_.get();
        decltype(auto) seeds = seed1_scratch_.get();
        SD::resize_like(duals, q0.main);
        SD::resize_like(seeds, q0.main);
    } else {
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

template<typename T, typename SD, typename OdeType>
void VariationalOdeSys<T, SD, OdeType>::Rhs(State& out, const T& t, const State& q) const{
    using S = value_t<T, SD>;
    if constexpr (AD_RHS){
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

        // The Jacobian of the main system, then the variational part
        decltype(auto) J = jm_scratch_.get();
        this->JacMain(J, t, q.main);
        SD::JD::linalg::dot(out.variation, J, q.variation);
    }
}


// ----------------------------------------------------------------------------
// VariationalOdeSys Jac
// ----------------------------------------------------------------------------

template<typename T, typename SD, typename OdeType>
void VariationalOdeSys<T, SD, OdeType>::Jac(JacMat& out, const T& t, const State& q) const{
    using S = value_t<T, SD>;
    if constexpr (AD_JAC){
        // One main Rhs call with 2nd order duals. The gradient of f_i holds row i of J in its
        // values, and row i of dJ/dq*delta_q in its own gradients, dotted with delta_q
        decltype(auto) out_duals = dual2_scratch_.get();
        decltype(auto) seeds = seed2_scratch_.get();
        xdiff::set_seeds<T, SD>(seeds, q.main);
        ode_.Rhs(out_duals, t, seeds);

        const auto row_i = [](const dual_t<T, SD, 2>& fi){
            return std::tie(fi.grad());
        };
        SD::JD::linalg::for_each_row([](S& J_ij, const dual_t<T, SD, 1>& dfi_dqj){
            J_ij = dfi_dqj.value();
        }, row_i, std::tie(out[0]), out_duals);

        SD::JD::linalg::for_each_row([&q](S& B_ij, const dual_t<T, SD, 1>& dfi_dqj){
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
        SD::JD::linalg::for_each_row([](S& out_ij, const S& dfi_dqj){
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
void VariationalOdeSys<T, SD, OdeType>::delta_J(MainJacMat& mat, const T& t, const State& q) const{
    /*
    Requires 4 rhs evaluations per derivative, but only for the bottom left half
    So should be more efficient than using central finite differences on the
    full (augmented) system.
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
    decltype(auto) f0 = f0_scratch_.get();
    decltype(auto) col = col_scratch_.get();

    copy_state<SD>(x, q.main);
    ode_.Rhs(f0, t, q.main); // unperturbed evaluation, for the pure second derivatives

    // Column j of the block, d/dq_j of J*delta_q, into col
    SD::JD::linalg::for_each_column([](S& mat_ij, const S& col_i){
        mat_ij = col_i;
    }, [&](S& xj, const S& qj, const T& atol_j){

        const T h_j = EPS_4TH * max_ref(atol_j, T(abs(qj)));
        SD::for_each([](S& col_i){
            col_i = 0;
        }, col);

        // Now iterating over the index k,
        // which is contracted with delta_q[k] in the final result
        // for each (i, j) element of the bottom left block
        SD::for_each([&](S& xk, const S& qk, const S& dq_k, const T& atol_k){ // d/dq_k

            if (&xk == &xj){
                // The 4-point stencil below degenerates when both perturbations
                // act on the same axis: it collapses to f[j+1] - f[j-1], which is a
                // first derivative, not a second one. The pure second derivative
                // needs the central 3-point stencil instead.
                xj = qj + h_j;
                ode_.Rhs(y1, t, x);

                xj = qj - h_j;
                ode_.Rhs(y2, t, x);

                xj = qj; // x[j] restored
                const T h_sq = h_j*h_j;
                SD::for_each([&](S& col_i, const S& y1_i, const S& y2_i, const S& f0_i){
                    // final value = delta_q[j] * d^2f^i/dx_j^2
                    col_i += dq_k * (y1_i + y2_i - f0_i - f0_i) / h_sq;
                }, col, y1, y2, f0);
                return;
            }

            const T h_k = EPS_4TH * max_ref(atol_k, T(abs(qk)));

            /*
            f[j-1, k+1]     f[j+1, k+1]
                ------------------
                |        |       |
                |        |       |
                |-----f[j,k]-----|
                |        |       |
                |        |       |
                ------------------
            f[j-1, k-1]     f[j+1, k-1]
            */

            // Bottom left evaluation
            xj = qj - h_j;
            xk = qk - h_k;
            ode_.Rhs(y1, t, x);

            // Upper right evaluation
            xj = qj + h_j;
            xk = qk + h_k;
            ode_.Rhs(y2, t, x);

            // Combine them into one of the two temporaries
            SD::for_each([](S& y1_i, const S& y2_i){
                y1_i += y2_i;
            }, y1, y2);

            // Upper left evaluation
            xj = qj - h_j;
            ode_.Rhs(y2, t, x);
            SD::for_each([](S& y1_i, const S& y2_i){
                y1_i -= y2_i;
            }, y1, y2);

            // Lower right evaluation
            xj = qj + h_j;
            xk = qk - h_k;
            ode_.Rhs(y2, t, x);
            const T four_h_sq = 4*h_j*h_k;
            SD::for_each([&](S& col_i, const S& y1_i, const S& y2_i){
                // final value = delta_q[k] * df^i/(dx_j dx_k)
                col_i += dq_k * (y1_i - y2_i) / four_h_sq;
            }, col, y1, y2);

            xj = qj; // reset x[j] and x[k] to their original values
            xk = qk;
        }, x, q.main, q.variation, atol_);

        return std::tie(col);
    }, std::tie(mat), x, q.main, atol_);
}




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
auto VariationalSolver<S, T, SD, OdeType, SP, Derived>::normalized_ics(State q0) -> State{
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



// template<typename T, size_t N>
// std::unique_ptr<ODE<T, N>> VariationalODE<T, N>::clone() const{
//     return std::make_unique<VariationalODE<T, N>>(*this);
// }

// template<typename T, size_t N>
// const std::vector<T>& VariationalODE<T, N>::renorm_times() const{
//     return renorm_times_;
// }

// template<typename T, size_t N>
// const std::vector<T>& VariationalODE<T, N>::lyap_values() const{
//     return lyap_values_;
// }

// template<typename T, size_t N>
// const std::vector<T>& VariationalODE<T, N>::kick_values() const{
//     return kick_values_;
// }

// template<typename T, size_t N>
// void VariationalODE<T, N>::clear(){
//     Base::clear();
//     renorm_times_ = std::vector<T>{};
//     lyap_values_ = std::vector<T>{};
//     kick_values_ = std::vector<T>{};
// }

// template<typename T, size_t N>
// void VariationalODE<T, N>::reset(){
//     Base::reset();
//     renorm_times_ = std::vector<T>{};
//     lyap_values_ = std::vector<T>{};
//     kick_values_ = std::vector<T>{};
// }

// template<typename T, size_t N>
// const ChaoticSolver<T, N, UtilPolicy::RichVirtual>* VariationalODE<T, N>::solver() const {
//     return static_cast<const ChaoticSolver<T, N, UtilPolicy::RichVirtual>*>(Base::solver());
// }

// template<typename T, size_t N>
// void VariationalODE<T, N>::register_state(){
//     Base::register_state();
//     renorm_times_.push_back(this->solver()->get_time());
//     lyap_values_.push_back(this->solver()->get_lyapunov_exponent());
//     kick_values_.push_back(this->solver()->get_kick());
// }


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

} // namespace ode
