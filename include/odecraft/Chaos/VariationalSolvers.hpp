#pragma once

#include "odecraft/Core/SolverFactory.hpp"
#include <odecraft/Events/Events.hpp>
#include <odecraft/DenseOde/OdeInt.hpp>
#include <odecraft/Core/VirtualBase.hpp>
#include <odecraft/Core/VirtualTraits.hpp>
#include <odecraft/Steppers/Steppers.hpp>
#include <odecraft/Toolkit/Toolkit.hpp>
#include "VariationalContract.hpp"
#include "odecraft/Toolkit/Tools.hpp"
#include <odecraft/Contracts/DenseDescriptor.hpp>

namespace ode::chaos {

// ============================================================================
// DECLARATIONS
// ============================================================================
using contracts::VarSD;

namespace detail{

template<typename T, typename SD>
T state_norm(const state_t<T, SD>& src){
    T norm_sqr{0};
    SD::for_each([&norm_sqr](const value_t<T, SD>& x){
        norm_sqr += SD::norm_sqr(x);
    }, src);
    return sqrt(norm_sqr);
}


template<typename T, typename SD>
state_t<T, SD>& normalized(state_t<T, SD>& out, const state_t<T, SD>& src){
    T norm = state_norm<T, SD>(src);
    SD::for_each([&norm](value_t<T, SD>& delta_x, const value_t<T, SD>& x){
        delta_x = x / norm;
    }, out, src);
    return out;
}

} // namespace ode::chaos::detail


template<typename T, typename SD, UtilPolicy UP>
class ChaoticSolver;

template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived = void>
class VariationalSolver;

template<UtilPolicy UP, typename T, typename SD, typename OdeType, typename... Args>
pbox::Box<ChaoticSolver<T, SD, UP>> make_variational_solver(Stepper method, OdeType ode, T t0, state_t<T, VarSD<SD>> q0, T period, Args&&... args);


// The solver integrates states of VarSD<SD>, the main state and its variation. MainState and
// MainJacMat are those of the main system alone.
template<typename T, typename SD, UtilPolicy UP>
class ChaoticSolver : public std::conditional_t<UP==UtilPolicy::RichVirtual, OdeRichSolver<T, VarSD<SD>>, OdeSolver<T, VarSD<SD>>>{

public:

    using MainState = state_t<T, SD>;
    using MainJacMat = jacmat_t<T, SD>;

    // ACCESSORS
    virtual void    get_rhs_main(MainState& out, const T& t, const MainState& q) const = 0;

    virtual void    get_jac_main(MainJacMat& out, const T& t, const MainState& q) const = 0;

    virtual T       get_elapsed_time() const = 0;

    virtual T       get_kick() const = 0;

    virtual T       get_period() const = 0;

    virtual T       get_log_ksi() const = 0;

    virtual T       get_lyapunov_exponent() const = 0;

    virtual T       get_stretching_number() const = 0;

};


namespace detail{

/*
Trait templates backing the constraints on VariationalOdeSys' templated Rhs/Jac.

They live at namespace scope rather than as members of the class because an
out-of-line definition of a constrained member template must repeat its
requires-clause with the *same tokens* as the declaration, and a member-scope name
is not visible in the requires-clause that precedes the declarator. Qualifying it
there (VariationalOdeSys<...>::template Trait<Order>) yields a different token
sequence, so the definition no longer matches the declaration.
*/

// Duals of the *main* system's width (N). These guard the non-templated Rhs/Jac,
// which seed duals over the N original variables only.
template<int Order, typename OdeType, typename T, typename SD>
inline constexpr bool MainRhsSupportsDuals = supportsDualRhs<OdeType, Order, T, SD>;

// A dual with a single derivative component: the derivative along one direction
template<typename T, typename SD>
using dir_dual_t = xdiff::Dual<value_t<T, SD>, xdiff::StaticState<1>, 1>;

template<typename T, typename SD>
using dir_state_t = SD::template state_type<dir_dual_t<T, SD>>;

// Whether the main Rhs takes states of such duals, as its derivative along delta_q
template<typename OdeType, typename T, typename SD>
concept MainRhsSupportsDirDuals =
    requires(OdeType f, dir_state_t<T, SD>& out, T t, const dir_state_t<T, SD>& q) {
        { f.Rhs(out, t, q) } -> std::same_as<void>;
    };

// A dual over the main state whose scalars are directional duals along delta_q: the gradient of
// f_i holds row i of J in its values, and row i of dJ/dq*delta_q in their derivatives
template<typename T, typename SD>
using jac_dual_t = xdiff::Dual<dir_dual_t<T, SD>, SD, 1>;

template<typename T, typename SD>
using jac_state_t = SD::template state_type<jac_dual_t<T, SD>>;

template<typename T, typename SD>
using jac_seed_state_t = xdiff::SeedState<dir_dual_t<T, SD>, SD, 1>;

// Whether the main Rhs takes states of such duals
template<typename OdeType, typename T, typename SD>
concept MainRhsSupportsJacDuals =
    requires(OdeType f, jac_state_t<T, SD>& out, T t, const jac_seed_state_t<T, SD>& q) {
        { f.Rhs(out, t, q) } -> std::same_as<void>;
    };

} // namespace ode::chaos::detail


template<typename T, typename SD, typename OdeType>
struct VariationalOdeSys{
    static_assert(hasRhsFunc<OdeType, T, state_t<T, SD>>, "The given OdeType does not have a proper Rhs");
public:
    using State = state_t<T, VarSD<SD>>;
    using RealState = real_state_t<T, VarSD<SD>>;
    using JacMat = jacmat_t<T, VarSD<SD>>;
    using MainState = state_t<T, SD>;
    using MainRealState = real_state_t<T, SD>;
    using MainJacMat = jacmat_t<T, SD>;
    // same as in BaseSolver, but we need to redefine it here for the variational system
    static constexpr JacPolicy JP_MAIN = getJacPolicy<T, SD, OdeType>();

    /// @throws std::runtime_error if finite differences of the main system may be needed and `atol.main` contains zeros.
    VariationalOdeSys(OdeType ode, const State& q0, const RealState& atol);

    // Readable aliases for the namespace-scope traits above. The constraints below
    // must spell out the detail:: form (see the note there), but the function bodies
    // are free to use these.
    template<int Order>
    static constexpr bool MainRhsSupportsDuals = detail::MainRhsSupportsDuals<Order, OdeType, T, SD>;

    /*
    For the variational part, J*delta_q, the derivative of f along delta_q:
        If (JP_MAIN == Exact) : J from it, times delta_q
        else if (JP_MAIN == Nullable) use that,
            with possible fallback to a finite difference along delta_q
        else if (JP_MAIN == Autodiff) : one main Rhs call with duals that carry the derivative along delta_q
        else : a finite difference along delta_q, 2 more main Rhs calls
    */
    void    Rhs(State& out, const T& t, const State& q) const;

    // Providing a Jacobian and not letting the solver use finite differences,
    // only using them here as a final fallback. Both blocks come from 2nd order duals
    // if the main Rhs accepts them, otherwise J from JacMain and dJ/dq*delta_q from delta_J.
    void    Jac(JacMat& out, const T& t, const State& q) const;

    // The Jacobian of the main system, with the same dispatch as the variational part of Rhs
    void    JacMain(MainJacMat& out, const T& t, const MainState& q) const;

    const OdeType& ode() const;

private:

    // 1st order duals give J for JacMain, and f and J*delta_q in one main Rhs call
    static constexpr bool AD_RHS = (JP_MAIN == JacPolicy::Autodiff);
    // Duals with only the derivative along delta_q do the latter at the cost of about two main Rhs calls
    static constexpr bool AD_DIR = AD_RHS && detail::MainRhsSupportsDirDuals<OdeType, T, SD>;
    // An explicit Jacobian gives J*delta_q as J times delta_q
    static constexpr bool JAC_RHS = (JP_MAIN == JacPolicy::Exact || JP_MAIN == JacPolicy::Nullable);
    // Duals over the main state with directional duals as scalars give both blocks of the full
    // Jacobian, carrying 2(n+1) numbers per element instead of the (n+1)^2 of 2nd order duals
    static constexpr bool AD_JAC_DIR = detail::MainRhsSupportsJacDuals<OdeType, T, SD>;
    // 2nd order duals do the same otherwise
    static constexpr bool AD_JAC = !AD_JAC_DIR && detail::MainRhsSupportsDuals<2, OdeType, T, SD>;

    void    jacmat_findiffs(MainJacMat& mat, const T& t, const MainState& q) const;

    // J*delta_q into out, by a central difference along delta_q: 2 main Rhs calls
    void    jvp_findiffs(MainState& out, const T& t, const State& q) const;

    // Computes the bottom left matrix of the full jacobian matrix,
    // which is d(J)/dq * delta_q, of dims = (N x N)
    // by mixed central differences along each axis and along delta_q: 4 main Rhs calls per column
    void    delta_J(MainJacMat& mat, const T& t, const State& q) const;

    OdeType ode_;
    MainRealState atol_;
    [[no_unique_address]] ::ode::detail::scratch_if<JAC_RHS, MainJacMat> jm_scratch_;  // J, for J*delta_q
    worker_t<MainState> x_scratch_, y1_scratch_, y2_scratch_;                   // finite differences
    worker_t<MainState> col_scratch_;                                            // delta_J
    [[no_unique_address]] ::ode::detail::scratch_if<AD_DIR || AD_JAC_DIR, detail::dir_state_t<T, SD>> dir_in_scratch_;
    [[no_unique_address]] ::ode::detail::scratch_if<AD_DIR, detail::dir_state_t<T, SD>> dir_out_scratch_;
    [[no_unique_address]] ::ode::detail::scratch_if<AD_JAC_DIR, detail::jac_state_t<T, SD>> jac_out_scratch_;
    [[no_unique_address]] ::ode::detail::scratch_if<AD_JAC_DIR, detail::jac_seed_state_t<T, SD>> jac_seed_scratch_;
    [[no_unique_address]] ::ode::detail::scratch_if<AD_RHS, dual_state_t<T, SD, 1>> dual1_scratch_;
    [[no_unique_address]] ::ode::detail::scratch_if<AD_RHS, seed_state_t<T, SD, 1>> seed1_scratch_;
    [[no_unique_address]] ::ode::detail::scratch_if<AD_JAC, dual_state_t<T, SD, 2>> dual2_scratch_;
    [[no_unique_address]] ::ode::detail::scratch_if<AD_JAC, seed_state_t<T, SD, 2>> seed2_scratch_;
};


template<Stepper S, typename T, typename SD, typename OdeType, SolverPolicy SP, typename Derived>
class VariationalSolver : public ::ode::detail::SolverTypeGetter<S, SP, T, VarSD<SD>, VariationalOdeSys<T, SD, OdeType>, GetDerived<VariationalSolver<S, T, SD, OdeType, SP, Derived>, Derived>>::type {

    // As ODECRAFT_DECLARE_STEPPER, qualified: the unqualified names would declare new templates in ode::chaos
    template<typename, typename, typename, typename, SolverPolicy> friend class ::ode::BaseSolver;
    template<typename, typename, typename, typename, SolverPolicy> friend class ::ode::RichSolver;

    /**
    This solvers integrates the coupled / augmented ODE system consisting of the original system and the variational equations. The state vector has shape {main_vector, main_vector}, where the first entry corresponds to the state of the original system and the second corresponds to the deviation vector. The deviation vector is renormalized every "period" time units, and log_ksi keeps track of the logarithm of the stretching factor since the last renormalization. The user can retrieve the current value of the largest Lyapunov exponent using lyapunov_exponent(), which is computed as log_ksi divided by the elapsed time since the beginning of the integration.

    The augmented system is:

    dq/dt = f(q, t)
    d(delta_q)/dt = J(q, t) * delta_q

    The Jacobian of the original system is required to construct the variational equations.
    For implicit solvers, the Jacobian of the full system (original + variational) is also required.
    If the solver is provided with a generic Rhs function, then the Jacobian is computed using optimized automatic directional differentiation. If not, then if the original ODE system provides a jacobian function, it will be used to compute the Rhs of the augmented system. If the Rhs is not generic enough or the Jacobian is not provided, there are efficient fallbacks to compute the augmented Rhs and Jacobian using first and second order finite differences, respectively.
    
    The Jacobian of the augmented system schematically looks like:

    [           J            0  ]
    [ d(J)/dq * delta_q      J  ]

    */

    using Base = ::ode::detail::SolverTypeGetter<S, SP, T, VarSD<SD>, VariationalOdeSys<T, SD, OdeType>, GetDerived<VariationalSolver<S, T, SD, OdeType, SP, Derived>, Derived>>::type;

public:
    using State = contracts::VarState<state_t<T, SD>>; //same as state_t<T, VarSD<SD>>;
    using RealState = contracts::VarState<real_state_t<T, SD>>; // //same as real_state_t<T, VarSD<SD>>;
    using JacMat = jacmat_t<T, VarSD<SD>>;
    using MainState = state_t<T, SD>;
    using MainJacMat = jacmat_t<T, SD>;

    // q0.main is the initial state of the main system, and q0.variation the deviation vector, which is normalized
    VariationalSolver(OdeType ode, T t0, State q0, T period, RealState rtol, RealState atol, T min_step=0, T max_step=0, T stepsize=0, int dir = 1, EventList<T, State> events={});

    void    RhsMain(MainState& out, const T& t, const MainState& q) const;
    void    JacMain(MainJacMat& out, const T& t, const MainState& q) const;

    T       elapsed_time() const;
    T       stretching_number() const;
    T       kick() const;
    T       period() const;
    T       log_ksi() const;
    T       lyapunov_exponent() const;
    void    Reset();
    bool    SetIcs(T t0, State y0, T stepsize, int direction = 0);

    // VIRTUAL INTERFACE ALIASES (overrides that forward to the non-virtual accessors)
    void    get_rhs_main(MainState& out, const T& t, const MainState& q) const;
    void    get_jac_main(MainJacMat& out, const T& t, const MainState& q) const;
    T       get_elapsed_time() const;
    T       get_kick() const;
    T       get_period() const;
    T       get_log_ksi() const;
    T       get_lyapunov_exponent() const;
    T       get_stretching_number() const;

protected:

    void    ReAdjust(const State& new_vector);

    template<typename... Args>
    bool Adv_Impl(Args&&... args);

private:

    // q0 with its variation normalized
    static State normalized_ics(State q0);

    State flag_state_; // the state at the last renormalization time, normalized, until ReAdjust applies it
    worker_t<State> current_scratch_; // the current state, for stretching_number
    T period_;
    T t_next_; // the next time at which to renormalize
    T t_last_;
    T logksi_ = 0; // log of the norm of the deviation vector at the last renormalization, used for computing the Lyapunov exponent
    T logksi_last_ = 0; // log of the norm of the deviation vector at the previous renormalization, used for computing the Lyapunov exponent
    size_t np = 0;
    bool flagged = false;
};


template<typename T, typename SD>
class VariationalODE : public ODE<T, VarSD<SD>>{

public:

    using Base = ODE<T, VarSD<SD>>;

    using State = state_t<T, VarSD<SD>>;
    using RealState = real_state_t<T, VarSD<SD>>;

    template<hasRhsFunc<T, state_t<T, SD>> OdeType>
    VariationalODE(OdeType ode, T t0, State q0, T period, RealState rtol, RealState atol, T min_step = 0, T max_step = 0, T stepsize = 0, int dir = 1, EventList<T, State> events = {}, Stepper method = Stepper::RK45);

    std::unique_ptr<ODE<T, VarSD<SD>>> clone() const override;

    const std::vector<T>& renorm_times() const;

    const std::vector<T>& lyap_values() const;

    const std::vector<T>& kick_values() const;

    DEFAULT_RULE_OF_FOUR(VariationalODE)

    void clear() override;

    void reset() override;

    const ChaoticSolver<T, SD, UtilPolicy::RichVirtual>* solver() const;

protected:

    void register_state() override;

private:

    std::vector<T> renorm_times_ = {};
    std::vector<T> lyap_values_ = {};
    std::vector<T> kick_values_ = {};

};



template<UtilPolicy UP, typename T, typename SD, typename OdeType, typename... Args>
pbox::Box<ChaoticSolver<T, SD, UP>> make_variational_solver(Stepper method, OdeType ode, T t0, state_t<T, VarSD<SD>> q0, T period, Args&&... args);


template<SolverPolicy SP, Stepper S, typename T, typename SD, typename OdeType>
auto getVariationalSolver(OdeType ode, T t0, state_t<T, VarSD<SD>> q0, T period, real_state_t<T, VarSD<SD>> rtol, real_state_t<T, VarSD<SD>> atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T, state_t<T, VarSD<SD>>> events = {});

namespace common{


template<typename T, size_t... Shape>
using VarState = contracts::VarState<::ode::contracts::common::State<T, Shape...>>;

template<SolverPolicy SP, Stepper S, typename T, typename OdeType, size_t... Shape>
auto getVariationalSolver(OdeType ode, T t0, VarState<T, Shape...> q0, T period, VarState<T, Shape...> rtol, VarState<T, Shape...> atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T, VarState<T, Shape...>> events = {}){
    return ::ode::chaos::getVariationalSolver<SP, S, T, ::ode::contracts::common::SD<Shape...>>(std::move(ode), std::move(t0), std::move(q0), std::move(period), std::move(rtol), std::move(atol), std::move(min_step), std::move(max_step), std::move(stepsize), dir, std::move(events));
}

template<UtilPolicy UP, typename T, typename OdeType, size_t... Shape, typename... Args>
pbox::Box<ChaoticSolver<T, ::ode::contracts::common::SD<Shape...>, UP>> make_variational_solver(Stepper method, OdeType ode, T t0, VarState<T, Shape...> q0, T period, Args&&... args){
    return ::ode::chaos::make_variational_solver<UP, T, ::ode::contracts::common::SD<Shape...>>(method, std::move(ode), std::move(t0), std::move(q0), std::move(period), std::forward<Args>(args)...);
}

} // namespace ode::chaos::common

} // namespace ode::chaos


namespace ode::traits{

// The base solver asks with the SD it integrates, VarSD<SD>
template<Stepper S, typename T, typename SD, typename OdeType, typename DerivedVS>
struct SolverVirtualTypeTraits<::ode::chaos::VariationalSolver<S, T, SD, OdeType, SolverPolicy::Virtual, DerivedVS>, T, chaos::VarSD<SD>, SolverPolicy::Virtual> {
    using type = chaos::ChaoticSolver<T, SD, UtilPolicy::Virtual>;
};


template<Stepper S, typename T, typename SD, typename OdeType, typename DerivedVS>
struct SolverVirtualTypeTraits<::ode::chaos::VariationalSolver<S, T, SD, OdeType, SolverPolicy::RichVirtual, DerivedVS>, T, chaos::VarSD<SD>, SolverPolicy::RichVirtual> {
    using type = chaos::ChaoticSolver<T, SD, UtilPolicy::RichVirtual>;
};

} // namespace ode::traits
