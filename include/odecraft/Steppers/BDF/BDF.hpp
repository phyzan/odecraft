#pragma once

#include <odecraft/Core/RichSolver/RichBase.hpp>

namespace ode{

constexpr size_t BDF_MAX_ORDER = 5;

// ============================================================================
// DECLARATIONS
// ============================================================================

namespace detail{

template<typename T>
struct BDFCONSTS{
    std::array<T, BDF_MAX_ORDER+1> KAPPA;
    std::array<T, BDF_MAX_ORDER+1> GAMMA;
    std::array<T, BDF_MAX_ORDER+1> ALPHA;
    std::array<T, BDF_MAX_ORDER+1> ERR_CONST;

    BDFCONSTS();
};

struct NewtConv{
    bool converged;
    size_t n_iter;
    StepResult flag;
};

// The matrices of change_D, at their largest size: an order uses their leading (order+1) x (order+1) block
template<typename T>
using BDFMatrix = FArray2D<T, BDF_MAX_ORDER+1, BDF_MAX_ORDER+1>;

template<typename T>
void compute_R(BDFMatrix<T>& R, size_t order, const T& factor);

/// @brief The solution at t, out of the backward differences D at t2 of order `order`, for steps h.
template<typename T, typename SD>
void bdf_interp(state_t<T, SD>& result, const T& t, const T& t2, const T& h, const std::array<state_t<T, SD>, BDF_MAX_ORDER+3>& D, size_t order);

} // namespace ode::detail


template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived = void>
class BDF : public detail::BaseDispatcher<GetDerived<BDF<T, SD, OdeType, SP, Derived>, Derived>, T, SD, OdeType, SP>{

    ODECRAFT_DECLARE_STEPPER;
    
    using Base = detail::BaseDispatcher<GetDerived<BDF<T, SD, OdeType, SP, Derived>, Derived>, T, SD, OdeType, SP>;
    

public:
    using State = state_t<T, SD>;
    using RealState = real_state_t<T, SD>;
    using JacMat = jacmat_t<T, SD>;

    static constexpr int ERR_EST_ORDER = 1;
    static constexpr bool IS_IMPLICIT = true;
    static constexpr size_t NEWTON_MAXITER = 4;

    BDF(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T, State> events = {});

    DEFAULT_RULE_OF_FOUR(BDF)

    Stepper method() const;

    void            Reset();

protected:

    StepResult      adapt_impl(T& t_out, T& habs_out, State& q_out, const T& t, const T& stepsize, const State& q_in);
    auto            local_interp_impl() const;
    void            ReAdjust(const State& new_vector);
    bool            validate_ics_impl(const T& t0, const State& q0) const;

private:

    // The backward differences of the solution: D[k] is the k-th
    using Differences = std::array<State, BDF_MAX_ORDER+3>;

    void    reset_impl_alone();

    detail::NewtConv solve_bdf_system(State& y, const State& y_pred, State& d, State& work, const T& t_new, const T& c, const State& psi, const RealState& scale);

    void    change_D(const T& factor);

    void    set_prediction(State& y) const;

    void    set_psi(State& psi) const;

    JacMat                      J_;
    JacMat                      A_;   // I - c*J, then its LU factors
    SD::JD::linalg::pivots_type piv_; // the row pivots of the factors in A_
    std::array<Differences, 2>  D_;  // change_D writes into the one that is not current
    detail::BDFMatrix<T>        R_, U_, RU_;
    detail::BDFCONSTS<T>        coefs_;
    T                           newton_tol_;
    size_t                      order_ = 1;
    size_t                      n_eq_steps_ = 0;
    size_t                      idx_D_ = 0;
    size_t                      U_order_ = 0; // the order U_ was computed for, 0 for none
    bool                        valid_LU_ = false;

    // the predicted solution, psi, the accumulated correction and a working state of a step, and its error scale
    worker_t<State>             ypred_scratch_;
    worker_t<State>             psi_scratch_;
    worker_t<State>             d_scratch_;
    worker_t<State>             work_scratch_;
    worker_t<RealState>         scale_scratch_;

    // the Jacobian at the initial conditions that validate_ics_impl checks
    worker_t<JacMat>            jac_scratch_;

};

namespace detail{

template<typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
struct SolverTypeGetter<Stepper::BDF, SP, T, SD, OdeType, Derived>{
    using type = BDF<T, SD, OdeType, SP, Derived>;
};

} // namespace ode::detail


} // namespace ode
