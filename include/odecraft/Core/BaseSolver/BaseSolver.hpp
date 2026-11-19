#pragma once

/**
 * @file BaseSolver.hpp
 * @brief Base class template for ODE solvers using CRTP pattern.
 *
 * This file defines the BaseSolver template class, which provides the core
 * functionality for adaptive-step ODE integration. Derived solver classes
 * (e.g., RK45, DOP853) inherit from BaseSolver using the Curiously Recurring
 * Template Pattern (CRTP) to enable static polymorphism and avoid virtual
 * function overhead.
 *
 * The solver maintains internal state including current/previous solution
 * vectors, time values, step sizes, and integration status. It supports
 * forward and backward integration, adaptive step size control, dense
 * output interpolation, and event detection.
 */

#include "odecraft/Core/SolverFactory.hpp"
#include "odecraft/Toolkit/Toolkit.hpp" // IWYU pragma: keep
#include "../VirtualTraits.hpp"
#include "../VirtualBase.hpp" // IWYU pragma: keep
#include <string_view>



namespace ode{

namespace detail {

struct adl_tag {};

template<typename F>
concept has_stepper_tag = requires(const F* p) { stepper_tag(p, adl_tag{}); };

template<typename F>
struct stepper_type_getter {
    using type = void;
};

template<typename F> requires has_stepper_tag<F>
struct stepper_type_getter<F> {
    using type = decltype(stepper_tag(std::declval<const F*>(), adl_tag{}));
};


template<typename T>
struct StaticSolverFactors{

    /// @brief Safety factor for step size control (typically 0.8-0.95).
    inline static constexpr T SAFETY = T(9)/10;

    /// @brief Maximum step size increase factor per step.
    inline static constexpr T MAX_FACTOR = 10;

    /// @brief Minimum step size decrease factor per step.
    inline static constexpr T MIN_FACTOR = T(2)/10;
    
    /// @brief Absolute minimum step size before solver terminates.
    inline static constexpr T MIN_STEP = 100*std::numeric_limits<T>::epsilon();
};

template<typename T>
struct DynamicSolverFactors{
    T SAFETY = T(9)/10;
    T MAX_FACTOR = 10;
    T MIN_FACTOR = T(2)/10;
    T MIN_STEP = 100*std::numeric_limits<T>::epsilon();
};

template<typename T>
using StepperFactors = std::conditional_t<
    std::is_trivially_copyable_v<T>,
    StaticSolverFactors<T>,
    DynamicSolverFactors<T>
    >;

} // namespace ode::detail

/*
Method naming, for the CRTP chains of odecraft (BaseSolver, RichSolver, EventBase, NdInterpolator,
VectorField) and every class deriving from them:
    lower()       concrete; not meant to be overridden
    Upper()       may be overridden; always has a default
    lower_impl()  protected, must be overridden; its default is a static_assert
Upper() and lower_impl() are called through THIS, so that the most derived version runs. The only
direct calls are qualified ones: Base::Upper() from an override, and the calls where THIS cannot be
used, in a constructor (the parts derived from it do not exist yet) or in a class that closes its
chain (it has no Derived parameter). A class that closes two chains at once redeclares the lower()
methods that both of them give it.

Each chain starts at a hook base (StepperHook, EventHook, InterpolatorHook, VectorFieldHook), which
derives from the polymorphic interface, or an empty class, and holds the lower_impl() defaults: an
explicit instantiation of the CRTP base, as in the compiled library, does not reach them.
*/

/**
 * @brief The base of BaseSolver: it derives from the interface a solver is reached through, and
 * holds the defaults of the methods every stepper must implement.
 *
 * The interface is the one SolverVirtualTypeTraits gives: OdeSolver or OdeRichSolver for the
 * virtual policies, EmptySolver for the static ones.
 */
template<typename Derived, typename T, typename SD, SolverPolicy SP>
class StepperHook : public traits::SolverVirtualTypeTraits<Derived, T, SD, SP>::type {

    using Interface = traits::SolverVirtualTypeTraits<Derived, T, SD, SP>::type;

public:

    /// @brief The interface the solver is reached through (OdeSolver, OdeRichSolver, or one
    /// a solver specializes SolverVirtualTypeTraits for), void for the static policies.
    using virtual_base = std::conditional_t<std::is_same_v<Interface, EmptySolver>, void, Interface>;

protected:

    /**
     * @brief Perform one adaptive integration step, returning a StepResult.
     * @note Writes the new vector to y_new, which may be q_old_ itself (see BaseSolver::take_step),
     *       so it must not read vector_old() or interpolate. noexcept unless ODECRAFT_RHS_THROWS is defined.
     */
    auto adapt_impl(T& /*t_new*/, T& /*habs_new*/, state_t<T, SD>& /*y_new*/, const T& /*t*/, const T& /*habs*/, const state_t<T, SD>& /*q*/) noexcept(!detail::rhs_may_throw) {
        static_assert(sizeof(T) == 0, "StepperHook::adapt_impl not implemented");
    }

    /**
     * @brief Lightweight non-owning interpolator in the open-interval (t_old, t_new).
     Only guaranteed to be valid while the solver remains unchanged.
     Call local_interp_impl().detach() for an owning, independent interpolator for that interval
     which constructs a full copy of the required data members
     @return LightFunctor, callable with (State& out, const T& t),  t_old < t < t_new
    */
    auto local_interp_impl() const {
        static_assert(sizeof(T) == 0, "StepperHook::local_interp_impl not implemented");
    }
};


/**
 * @brief Base class for adaptive-step ODE solvers using CRTP.
 *
 * BaseSolver provides the common infrastructure for ODE integration including
 * state management, step size adaptation, interpolation, and event detection.
 * Derived classes must implement the core stepping algorithm via static
 * method overrides.
 *
 * @tparam Derived The derived solver class (CRTP pattern). Must define:
 *                 - static constexpr bool IS_IMPLICIT
 *                 - static constexpr int ERR_EST_ORDER
 *                 - adapt_impl(), local_interp_impl()
 * @tparam T       Scalar type for computations (e.g., double, float).
 * @tparam SD      The State Descriptor
 * @tparam OdeType class containing the Rhs and Jac members
 * @tparam SP      Solver policy controlling behavior (see SolverPolicy enum).
 *
 * @note This class uses static polymorphism (CRTP). The derived class passes itself as the
 *       first template parameter.
 */

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
class BaseSolver : public StepperHook<Derived, T, SD, SP> {

    static_assert(hasRhsFunc<OdeType, T, state_t<T, SD>>, "Invalid OdeType::Rhs signature");

public:

    using typename StepperHook<Derived, T, SD, SP>::virtual_base;
    /// @brief What clone() points to: the solver itself for the static policies, OdeSolver for
    /// the virtual ones. Not virtual_base: clone() overrides OdeSolver::clone(), and a
    /// std::unique_ptr return type cannot be covariant.
    using clone_type = std::conditional_t<std::is_void_v<virtual_base>, Derived, OdeSolver<T, SD>>;
    using state_descriptor = SD;
    using real_type = T;
    using value_type = value_t<T, SD>;
    using ode_type = OdeType;
    using State = state_t<T, SD>;
    using JacMat = jacmat_t<T, SD>;
    using RealState = SD::template state_type<T>;

    template<int Order>
    using DualState = dual_state_t<T, SD, Order>;

    template<int Order>
    using SeedState = seed_state_t<T, SD, Order>;

    static constexpr JacPolicy JP = getJacPolicy<T, SD, OdeType>();
    static constexpr SolverPolicy Policy = SP;
    
    BaseSolver() = delete;

    // ODE PROPERTIES

    /**
     * @brief Evaluate the right-hand side of the ODE system.
     * @param[out] out  Output state for the derivative dq/dt.
     * @param[in]  t    Integration parameter.
     * @param[in]  q    State vector
     * @note noexcept unless ODECRAFT_RHS_THROWS is defined.
    */
    void                 rhs(State& out, const T& t, const State& q) const noexcept(!detail::rhs_may_throw);

    /**
     * @brief Compute the Jacobian matrix of the ODE system.
     * 
     * Uses the exact Jacobian if provided, otherwise falls back to
     * finite difference approximation
     *
     * @param[out] out  Output Jacobian matrix in column-major order (size Nsys x Nsys).
     * @param[in]  t    Integration parameter.
     * @param[in]  q    State vector
     * @note noexcept unless ODECRAFT_RHS_THROWS is defined.
    */
    void                 jac(JacMat& out, const T& t, const State& q) const noexcept(!detail::rhs_may_throw);

    void                 jac(JacMat& out, const T& t, const State& q, const State& dq) const noexcept(!detail::rhs_may_throw);

    /**
     * @brief Approximate the Jacobian using central finite differences.
     * @param[out] out  Output Jacobian matrix in column-major order (size Nsys x Nsys).
     * @param[in]  t    Integration parameter.
     * @param[in]  q    State vector
    */
    void                 jac_approx(JacMat& out, const T& t, const State& q) const noexcept(!detail::rhs_may_throw);

    /**
     * @brief Approximate the Jacobian using central finite differences.
     * @param[out] out  Output Jacobian matrix in column-major order (size Nsys x Nsys).
     * @param[in]  t    Integration parameter.
     * @param[in]  q    State vector
     * @param[in] dq    Perturbation state vector to use for finite differences
    */
    void                 jac_approx(JacMat& out, const T& t, const State& q, const State& dq) const noexcept(!detail::rhs_may_throw);

    // ACCESSORS

    /// @brief Get the current time value.
    const T&            t() const;

    /// @brief Get the time value of the newest computed step.
    const T&            t_new() const;

    /// @brief Get the time value from the previous accepted step.
    const T&            t_old() const;

    /// @brief Get a view of the state vector from the newest computed step.
    const State&        vector_new() const;

    /// @brief Get a view of the state vector from the previous accepted step.
    const State&        vector_old() const;

    /// @brief Get the initial integration time.
    const T&            t_ics() const;

    /// @brief Get a view of the initial state vector.
    const State&        vector_ics() const;

    /// @brief Get the current step size (absolute value).
    const T&            stepsize() const;

    /// @brief Get the integration direction (+1 forward, -1 backward).
    int                 direction() const;

    /// @brief Get the relative tolerance for error control.
    const RealState&    rtol() const;

    /// @brief Get the absolute tolerance for error control.
    const RealState&    atol() const;

    /// @brief Get the minimum allowed step size.
    const T&            min_step() const;

    /// @brief Get the maximum allowed step size.
    const T&            max_step() const;

    bool                is_running() const;
    
    /// @brief Get the number of successful integration steps taken.
    size_t              step_count() const;

    /// @brief Fills the values of an array with those of the current true state vector
    void                FillCurrentState(State& out) const;
    StepperStatus       status() const;
    std::string_view    message() const;

    /**
     * @brief Print the current solver state to stdout.
     * @param prec Number of decimal places for floating-point output.
     */
    void                ShowState(std::ostream& out, int prec=8) const;

    /**
     * @brief Validate proposed initial conditions.
     * @param t0 Initial time.
     * @param q0 Initial state vector.
     * @return True if the ICs are valid (finite values, finite RHS evaluation).
     * @note An override calls Base::ValidateICs first, then adds its own checks.
    */
    bool                ValidateICs(const T& t0, const State& q0) const;

    bool                has_valid_ics() const;
    
    /**
     * @brief Compute an appropriate initial step size.
     * @param t Initial time.
     * @param q Initial state vector.
     * @return Suggested step size based on local scales and tolerances.
     */
    T                   auto_step(const T& t, const State& q) const;

    /// @brief Compute an appropriate step size from the current state.
    T                   auto_step() const;

    /**
     * @brief Create a dynamically allocated copy of this solver.
     * @return Pointer to a new solver instance. Caller owns the memory.
     */
    std::unique_ptr<clone_type> clone() const;

    // MODIFIERS

    /**
     * @brief Advance the solver by one integration step.
     * @return True if the step was successful, false if otherwise
     */
    bool                advance();

    
    /**
     * @param time Target time to integrate to.
     * @returns `true` if the solver reached `time`, `false` if stopped early
    */
    bool    advance_until(const T& time);

    /**
     * @brief Integrate until the specified time is reached.
     * @param time Target time to integrate to.
     * @param observer Callable `bool(t, q, t_ptr)`, invoked after every successful step.
     *     `t`     : Current integration time.
     *     `q` : Current state vector.
     *     `t_ptr` : const T*, null on every call but the last, where it is `&time`
     *     Return `false` to stop advancing immediately.
     * @returns `true` if the solver reached `time`, `false` if stopped early.
     */
    template<isObserver<T, State> Callable>
    bool    advance_until(
        const T& time,
        Callable&& observer
    );

    /**
     * @brief Integrate until the specified time is reached.
     * @param time Target time to integrate to.
     * @param observer Callable `bool(t, q, t_ptr)`, invoked after every successful step.
     *     `t`     : Current integration time.
     *     `q` : Current state vector.
     *     `t_ptr` : Address of the checkpoint the solver lands on, else null.
     *               If checkpoints.size() == 0, it is null on every call but the last,
     *               where it is `&time`.
     *     Return `false` to stop advancing immediately.
     * @return `true` if the solver reached `time`, `false` if stopped early
     */
    template<isObserver<T, State> Callable, isArray<T> ArrayType>
    bool    advance_until(
        const T& time,
        Callable&& observer,
        ArrayType&& checkpoints
    );

    /**
     * @brief Integrate until `time`, returning the dense output of the span integrated
     * @note If the integration stops early, this will be reflected in the interpolator
     */
    BoxedInterp<T, SD>   interpolate_until(const T& time);

    /// @brief Same as interpolate_until(time), calling `observer` like advance_until does.
    template<isObserver<T, State> Callable>
    BoxedInterp<T, SD>   interpolate_until(const T& time, Callable&& observer);

    /**
     * @brief Advance until (this->t() + interval*direction) is reached
     * @param interval Unsigned interval to integrate
     * @throws std::logic_error if the interval is negative
    */
    bool                advance_by(T interval);

    /**
     * @brief Set new initial conditions without reallocating memory.
     * @param t0      New initial time.
     * @param y0      New initial state vector (size Nsys).
     * @param stepsize Initial step size (0 = auto-compute).
     * @param direction Integration direction for the new ICs (+1 forward, -1 backward, 0 for unchanged).
     * @return True if ICs were valid and set successfully. Otherwise returns false and nothing happens. 
     * @throws std::domain_error if the direction is not any of [-1, 0, 1]
     */
    bool                SetICs(T t0, State y0, T stepsize = 0, int direction = 0);

    /// @brief Reset implementation hook. Derived should call base first.
    void                Reset();

    /// @brief Non-owning interpolator for the closed interval [t_old, t_new]. Valid while the solver is not advanced or goes out of scope
    auto light_interp() const;

    /// @brief Interpolator for the closed interval [t_old, t_new], detached from the Solver. Can be expensive to construct
    auto heavy_interp() const;

    /**
     * @brief Create an interpolator for dense output between two boundaries.
     * @param bdr1 First boundary index.
     * @param bdr2 Second boundary index.
     * @return Unique pointer to an interpolator object, which owns its data (it is detached from the solver).
     * @note Built from the derived class's local_interp_impl().
     */
    BoxedInterp<T, SD> state_interpolator(int bdr1, int bdr2) const;

    // VIRTUAL INTERFACE ALIASES (inline overrides to avoid virtual calls)
    // Accessors
    void                get_rhs(State& out, const T& t, const State& q) const { rhs(out, t, q); }
    void                get_jac(JacMat& out, const T& t, const State& q) const { jac(out, t, q); }
    void                get_jac(JacMat& out, const T& t, const State& q, const State& h) const { jac(out, t, q, h); }
    const T&            get_time() const { return t(); }
    const T&            get_new_time() const { return t_new(); }
    const T&            get_old_time() const { return t_old(); }
    void                get_current_vector(State& out) const { THIS->FillCurrentState(out); }
    const State&        get_new_vector() const { return vector_new(); }
    const State&        get_old_vector() const { return vector_old(); }
    const T&            get_ics_time() const { return t_ics(); }
    const State&        get_ics_vector() const {return vector_ics();}
    const T&            get_stepsize() const { return stepsize(); }
    int                 get_direction() const { return direction(); }
    const RealState&    get_rtol() const { return rtol(); }
    const RealState&    get_atol() const { return atol(); }
    const T&            get_min_step() const { return min_step(); }
    const T&            get_max_step() const { return max_step(); }
    size_t              get_step_count() const { return step_count(); }
    bool                get_is_running() const {return is_running();}
    StepperStatus       get_status() const { return status(); }
    std::string_view    get_message() const { return message(); }
    bool                get_validate_ics(const T& t0, const State& q0) const { return THIS->ValidateICs(t0, q0); }
    bool                get_has_valid_ics() const { return has_valid_ics(); }
    std::string_view    get_method() const { return method(); }
    BoxedInterp<T, SD>  get_state_interpolator(int bdr1, int bdr2) const { return state_interpolator(bdr1, bdr2); }
    T                   get_auto_step(const T& t, const State& q) const { return auto_step(t, q); }
    T                   get_auto_step() const { return auto_step(); }
    void                show_state(std::ostream& out, int prec=8) const { THIS->ShowState(out, prec); }

    // Modifiers
    bool                do_advance() { return advance(); }
    bool                do_advance_by(T interval) { return advance_by(interval); }

    bool                do_advance_until(T time) { return advance_until(time); }

    bool                do_advance_until(T time, const observer_t<T, State>& observer) {
        if (observer == nullptr){
            return advance_until(time);
        } else {
            return advance_until(time, observer);
        }
    }

    bool                do_advance_until(T time, const observer_t<T, State>& observer, std::span<const T> checkpoints) {
        if (observer == nullptr){
            return generic_advance_until(time, nullptr, checkpoints);
        } else {
            return generic_advance_until(time, observer, checkpoints);
        }
    }

    BoxedInterp<T, SD>   do_interpolate_until(T time) { return interpolate_until(time); }
    BoxedInterp<T, SD>   do_interpolate_until(T time, const observer_t<T, State>& observer) {
        if (observer == nullptr){
            return interpolate_until(time);
        } else {
            return interpolate_until(time, observer);
        }
    }

    void                do_reset() { THIS->Reset(); }
    bool                do_set_ics(T t0, State y0, T stepsize = 0, int direction = 0) { return THIS->SetICs(std::move(t0), std::move(y0), stepsize, direction); }

    /// @brief Name of the stepper
    std::string_view method() const {
        static_assert(detail::has_stepper_tag<Derived>, "No stepper tag found: declare the stepper with ODECRAFT_DECLARE_STEPPER");
        return detail::stepper_type_getter<Derived>::type::name;
    }

protected:

    using MainSolverType = BaseSolver;

    // =================== STATIC OVERRIDES (NECESSARY) ===============================
    // Derived classes MUST implement these attributes, and the methods declared in StepperHook.

    /// @brief Whether the method is implicit (must be defined in Derived).
    static constexpr bool       IS_IMPLICIT = Derived::IS_IMPLICIT;
    /// @brief Order of the error estimator (must be defined in Derived).
    static constexpr int        ERR_EST_ORDER = Derived::ERR_EST_ORDER;

    // ================================================================================

    // ========================= STATIC OVERRIDES (OPTIONAL) ==========================
    // Derived classes MAY override these methods. Call base implementation first.

    /// @brief Advance implementation. If t_lim is nullptr, advance normally. Otherwise, if the implementation yields a time beyond t_lim, it should only cache the step information (e.g. counters, or event or state register etc) since the solver is "not to go there yet" and only apply them once PerformStep is called with t_lim beyond the implementation's predicted step.
    template<typename... Args>
    bool        PerformStep(Args&&... args);

    constexpr bool    RequestTimeFloor(T& /*out*/) {
        return false;
    }

    /**
    @brief Re-adjustment hook right before the new state is modified. Derived should call base first.
    @param new_state New state vector values (size Nsys).
    @note Nothing has changed yet when this is called; it's a chance to update any internal data before the state is modified. The new state will be set to (t(), stepsize(), new_vector),
    where t() is the true current time, which might lie between t_old() and t_new() (e.g. if an event occurred).
    */
    void    ReAdjust(const State& new_state);

    // ================================================================================

    // =========================== HELPER METHODS =====================================

    inline const OdeType& ode() const {return ode_;}

    void        set_status(StepperStatus new_status);

    /// @brief Print a warning that the solver is dead.
    void        warn_dead() const;

    /// @brief Check if the current true state matches the new state.
    bool        is_at_new_state() const;

    void cerr(const std::string& message) const;

    template<typename... U>
    T   nearest_time(const U&... t) const;

    // ================== Constructors ==================

    BaseSolver(const BaseSolver&) = default;
    BaseSolver(BaseSolver&&) noexcept = default;
    BaseSolver& operator=(const BaseSolver&) = default;
    BaseSolver& operator=(BaseSolver&&) noexcept = default;
    ~BaseSolver() = default;

    // Main protected constructor
    BaseSolver(OdeType ode,
        T t0,
        State q0,
        RealState rtol,
        RealState atol,
        T min_step,
        T max_step,
        T stepsize,
        int direction,
        EventList<T, State> events={});

    // ==================================================

    inline constexpr const T& safety_factor() const {
        return factors_.SAFETY;
    }

    inline constexpr const T& max_factor() const {
        return factors_.MAX_FACTOR;
    }

    inline constexpr const T& min_factor() const {
        return factors_.MIN_FACTOR;
    }

    inline constexpr const T& tiny_factor() const {
        return factors_.MIN_STEP;
    }

    // Scratch for the temporaries of the member functions. The array page holds the deepest
    // nesting they reach, with one level of re-entry from an observer: advance_until holds a
    // vector while the observer calls auto_step(), which holds another while auto_step(t, q)
    // takes 4 more. RichSolver::show_state and BDF borrow from it too.
    
    worker_t<State>         scratch_0;
    worker_t<State>         scratch_1;
    worker_t<State>         scratch_2;
    worker_t<RealState>     real_scratch;
    worker_t<DualState<1>>  dual_scratch;
    worker_t<SeedState<1>>  seed_scratch;

private:

    bool    validate_it(StepResult result, const T& new_time);

    StepResult adapt_it(State& q_out, T& t_out, T& habs_out);

    /// @brief Takes a step from the newest state and, if it is accepted, makes it the newest
    /// state. Afterwards t_out and habs_out hold the previous t_old and step size.
    bool    take_step(T& t_out, T& habs_out);

    bool lt(const T& t_a, const T& t_b) const;
    bool le(const T& t_a, const T& t_b) const;

    template<typename Callable, typename ArrayType>
    bool    generic_advance_until(
        const T& time,
        Callable&& observer,
        ArrayType&& checkpoints
    );

    template<typename Callable>
    BoxedInterp<T, SD>   generic_interpolate_until(
        const T& time,
        Callable&& observer
    );

    template<typename A, typename... Rest>
    const T&    nearest_time_priv(const A& t_a, const Rest&... t_rest) const;

    template<typename A, typename B, typename... Rest>
    const T&    nearest_time_helper(const A& t_a, const B& t_b, const Rest&... t_rest) const;

    template<typename A, typename B>
    const T& nearest_of(const A& t_a, const B& t_b) const;

    /// @brief Only use inside PerformStep (so that if the state here is updated, all derived classes are aware). Move the current state to a new time between the current time and the most recently adapted state. This is a lowlevel operation, so use carefully or the intended bahavior might break.
    void                    move_state(const T& time);
    
    OdeType         ode_;
    T               t_ics_, habs_ics_, t_old_, t_new_, t_true_, habs_new_, min_step_, max_step_;
    State           q_ics_, q_old_, q_new_;
    RealState       rtol_, atol_;
    detail::StepperFactors<T> factors_;
    size_t          step_count_ = 0;
    int             direction_ = 1;
    StepperStatus   status_ = StepperStatus::NewStep;
};


namespace traits{

template<typename cls, typename derived>
using GetDerived = std::conditional_t<(std::is_same_v<derived, void>), cls, derived>;

} // namespace ode::traits


} // namespace ode
