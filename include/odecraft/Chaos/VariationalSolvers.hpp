#ifndef ODECRAFT_VARIATIONAL_SOLVERS_HPP
#define ODECRAFT_VARIATIONAL_SOLVERS_HPP

#include <odecraft/Core/Events.hpp>
#include <odecraft/DenseOde/OdeInt.hpp>
#include <odecraft/Core/VirtualBase.hpp>
#include <odecraft/Core/VirtualTraits.hpp>
#include <odecraft/Steppers/Steppers.hpp>


namespace ode::chaos {

// ============================================================================
// DECLARATIONS
// ============================================================================


namespace detail{

// out (size 2*nsys) and in (size 2*nsys) can be the same pointer
template<typename T>
void normalized(T* out, const T* src, size_t nsys);

} // namespace ode::chaos::detail


template<typename T, size_t N, UtilPolicy UP>
class ChaoticSolver;

template<UtilPolicy UP, typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
pbox::Box<ChaoticSolver<T, 2*N, UP>> make_variational_solver(Stepper method, OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, Args&&... args);

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class VariationalSolver;


template<typename T, size_t N, UtilPolicy UP>
class ChaoticSolver : public std::conditional_t<UP==UtilPolicy::RichVirtual, OdeRichSolver<T, 2*N>, OdeSolver<T, 2*N>>{

public:

    // ACCESSORS
    virtual void    get_rhs_main(T* out, const T& t, const T* q) const = 0;

    virtual void    get_jac_main(T* out, const T& t, const T* q) const = 0;

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
template<typename T, size_t N, typename OdeType, size_t Order>
inline constexpr bool MainRhsSupportsDuals = supportsDualRhs<OdeType, T, N, Order>;

template<typename T, size_t N, typename OdeType, size_t Order>
inline constexpr bool MainJacSupportsDuals = supportsDualJac<OdeType, T, N, Order>;

// Duals of the *augmented* system's width (2*N). These guard the templated Rhs/Jac:
// those receive duals differentiated with respect to all 2*N augmented variables, so
// the main Rhs/Jac must accept that width, not N. Keeping the two families apart
// matters for an OdeType whose Rhs is templated on the order but fixed at width N:
// it satisfies the Main* traits yet cannot be called from the templated overloads.
template<typename T, size_t N, typename OdeType, size_t Order>
inline constexpr bool MainRhsSupportsAugDuals = supportsDualRhs<OdeType, T, 2*N, Order>;

template<typename T, size_t N, typename OdeType, size_t Order>
inline constexpr bool MainJacSupportsAugDuals = supportsDualJac<OdeType, T, 2*N, Order>;

// First condition (... && ...) preferred, but if not, the second case is utilized
template<typename T, size_t N, typename OdeType, size_t Order>
inline constexpr bool FullRhsSupportsDuals =
    (MainRhsSupportsAugDuals<T, N, OdeType, Order> && MainJacSupportsAugDuals<T, N, OdeType, Order>)
    || MainRhsSupportsAugDuals<T, N, OdeType, Order+1>; // 2 conditions in order of preference

// Two conditions in order of preference.
// If the first condition is satisfied, see the scheme in the definition for how
// it will be used. The second condition is a fallback, which also uses that scheme.
template<typename T, size_t N, typename OdeType, size_t Order>
inline constexpr bool FullJacSupportsDuals =
    (MainJacSupportsAugDuals<T, N, OdeType, Order+1> || FullRhsSupportsDuals<T, N, OdeType, Order+2>);

} // namespace ode::chaos::detail


template<typename T, size_t N, hasRhsFunc<T> OdeType>
struct VariationalOdeSys{

public:

    // same as in BaseSolver, but we need to redefine it here for the variational system
    static constexpr JacPolicy JP_MAIN = getJacPolicy<T, N, OdeType>();

    VariationalOdeSys(OdeType ode, size_t ode_nsys, T atol);

    // Readable aliases for the namespace-scope traits above. The constraints below
    // must spell out the detail:: form (see the note there), but the function bodies
    // are free to use these.
    template<size_t Order>
    static constexpr bool MainRhsSupportsDuals = detail::MainRhsSupportsDuals<T, N, OdeType, Order>;
    template<size_t Order>
    static constexpr bool MainJacSupportsDuals = detail::MainJacSupportsDuals<T, N, OdeType, Order>;
    template<size_t Order>
    static constexpr bool MainRhsSupportsAugDuals = detail::MainRhsSupportsAugDuals<T, N, OdeType, Order>;
    template<size_t Order>
    static constexpr bool MainJacSupportsAugDuals = detail::MainJacSupportsAugDuals<T, N, OdeType, Order>;
    template<size_t Order>
    static constexpr bool FullRhsSupportsDuals = detail::FullRhsSupportsDuals<T, N, OdeType, Order>;
    template<size_t Order>
    static constexpr bool FullJacSupportsDuals = detail::FullJacSupportsDuals<T, N, OdeType, Order>;


    // This Dual-supporting Rhs should not make the Jac provided below unutilized by the solver.
    // In contrast, the appropriate Jac overload of this struct should be used by the solver
    // no matter the availability of this Rhs overload.
    //
    // Passes n-order duals in the main Rhs and the main Jac
    // or if Jac does not support that, passes (n+1)-order duals in the main Rhs
    // Can be used to compute the Jacobian of the *full, augmented* system
    // No point to call it to compute the Jacobian of the main system,
    // since if this is unlocked, then OdeType::Rhs can be called with duals
    //
    // N > 0 is required so that the scratch arrays are allocated on the stack,
    // since we cannot preallocate for every possible `Order`
    template<size_t Order> // Not important feature, the Stepper will choose the Jac function anyway, not autodiff
    requires (detail::FullRhsSupportsDuals<T, N, OdeType, Order> && N > 0)
    void    Rhs(DualType<T, 2*N, Order>* out, const T& t, SeedVec<T, 2*N, Order> q) const;

    /*
    For the variational part:
        If (JP_MAIN == Exact) : use it
        else if (JP_MAIN == Nullable) use that,
            with possible fallback to finite differences
        else if (JP_MAIN == Autodiff) : Just pass 1st order duals in the main Rhs
        else : use finite differences
    */
    void    Rhs(T* out, const T& t, const T* q) const;

    // Not ever required by the solver, the untemplated Jac is only (for some steppers) used
    template<size_t Order>
    requires (detail::FullJacSupportsDuals<T, N, OdeType, Order> && N > 0)
    void    Jac(DualType<T, 2*N, Order>* out, const T& t, SeedVec<T, 2*N, Order> q) const;

    // Providing a Jacobian and not letting the solver use finite differences,
    // only using them here as a final fallback.
    void    Jac(T* out, const T& t, const T* q) const;

    const OdeType& ode() const;

    constexpr size_t nsys_main() const;

private:

    struct ScratchDynamic{

        ScratchDynamic(size_t nsys_main);

        Array1D<T, 3*N>& findiffs() const;
        Array1D<T, N*N>& jacmat() const;
        Array1D<DualType<T, N, 1>, N>& duals() const;
        Array1D<DualType<T, N, 2>, N>& dduals() const;

    private:
        size_t nsys = N;
        mutable Array1D<T, 3*N> findiffs_;
        mutable Array1D<T, N*N> jacmat_;
        mutable Array1D<DualType<T, N, 1>, N> duals_;
        mutable Array1D<DualType<T, N, 2>, N> dduals_;
    };

    struct ScratchStatic{

        ScratchStatic(size_t nsys_main);

        // Returned by value: a fresh automatic buffer per call. See scratch_is_static.
        Array1D<T, 3*N> findiffs() const;
        Array1D<T, N*N> jacmat() const;
        Array1D<DualType<T, N, 1>, N> duals() const;
        Array1D<DualType<T, N, 2>, N> dduals() const;

    };

    using ScratchType = std::conditional_t<::ode::detail::scratch_is_static<T, N>, ScratchStatic, ScratchDynamic>;

    void    jacmat_findiffs(T* mat, const T& t, const T* q) const;

    // Computes the bottom left matrix of the full jacobian matrix,
    // which is d(J)/dq * delta_q, of dims = (N x N)
    // by performing central finite differences of 2nd diff order
    void    delta_J(T* mat, const T& t, const T* q, const T* dt) const;

    OdeType ode_;
    ScratchType scratch;
    size_t nsys_ = N; // Size of the original system, without the variational equations (not the augmented system)
    T atol_;
};


template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
class VariationalSolver : public ::ode::detail::SolverTypeGetter<S, T, 2*N, SP, VariationalOdeSys<T, N, OdeType>, GetDerived<VariationalSolver<S, T, N, SP, OdeType, Derived>, Derived>>::type {

    /**
    This solvers integrates the coupled / augmented ODE system consisting of the original system and the variational equations. The state vector is of size 2*N, where the first N entries correspond to the state of the original system and the last N entries correspond to the deviation vector. The deviation vector is renormalized every "period" time units, and log_ksi keeps track of the logarithm of the stretching factor since the last renormalization. The user can retrieve the current value of the largest Lyapunov exponent using lyapunov_exponent(), which is computed as log_ksi divided by the elapsed time since the beginning of the integration.

    The augmented system is:

    dq/dt = f(q, t)
    d(delta_q)/dt = J(q, t) * delta_q

    The Jacobian of the original system is required to construct the variational equations.
    For implicit solvers, the Jacobian of the full system (original + variational) is also required.
    If the solver is provided with a templated Rhs function, then all jacobians are computed using autodiff. Otherwise, the solver must provide an exact jacobian for the original system, and finite differences are used to compute the jacobian of the full system, which schematically looks like:

    [           J            0  ]
    [ d(J)/dq * delta_q      J  ]

    */

    using Base = typename ::ode::detail::SolverTypeGetter<S, T, 2*N, SP, VariationalOdeSys<T, N, OdeType>, GetDerived<VariationalSolver<S, T, N, SP, OdeType, Derived>, Derived>>::type;

public:

    template<typename... Args>
    VariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir = 1, Args&&... extra);

    T       elapsed_time() const;

    T       stretching_number() const;

    T       kick() const;

    T       period() const;

    T       log_ksi() const;

    T       lyapunov_exponent() const;

    void    Reset();

    void    RhsMain(T* out, const T& t, const T* q) const;

    void    JacMain(T* out, const T& t, const T* q) const;

    // VIRTUAL INTERFACE ALIASES (overrides that forward to the non-virtual accessors)
    void    get_rhs_main(T* out, const T& t, const T* q) const;
    void    get_jac_main(T* out, const T& t, const T* q) const;
    T       get_elapsed_time() const;
    T       get_kick() const;
    T       get_period() const;
    T       get_log_ksi() const;
    T       get_lyapunov_exponent() const;
    T       get_stretching_number() const;

protected:

    void    ReAdjust(const T* new_vector);

    template<typename... Args>
    bool Adv_Impl(Args&&... args);

private:
    struct private_tag{};

    static Array1D<T, 2*N> join_arrays(View1D<T, N> q0, View1D<T, N> delta_q0);

    mutable Array1D<T, 4*N> worker;
    Array1D<T, 2*N> tmp_state_;
    T period_;
    T t_next_; // the next time at which to renormalize
    T t_last_;
    T logksi_ = 0; // log of the norm of the deviation vector at the last renormalization, used for computing the Lyapunov exponent
    T logksi_last_ = 0; // log of the norm of the deviation vector at the previous renormalization, used for computing the Lyapunov exponent
    size_t np = 0;
    bool flagged = false;
};





template<typename T, size_t N>
class VariationalODE : public ODE<T, N>{

public:

    using Base = ODE<T, N>;

    template<hasRhsFunc<T> OdeType>
    VariationalODE(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir = 1, EventList<T> events = {}, Stepper method = Stepper::RK45);

    std::unique_ptr<ODE<T, N>> clone() const override;

    const std::vector<T>& renorm_times() const;

    const std::vector<T>& lyap_values() const;

    const std::vector<T>& kick_values() const;

    DEFAULT_RULE_OF_FOUR(VariationalODE)

    void clear() override;

    void reset() override;

    const ChaoticSolver<T, N, UtilPolicy::RichVirtual>* solver() const;

protected:

    void register_state() override;

private:

    std::vector<T> renorm_times_ = {};
    std::vector<T> lyap_values_ = {};
    std::vector<T> kick_values_ = {};

};



template<UtilPolicy UP, typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
pbox::Box<ChaoticSolver<T, 2*N, UP>> make_variational_solver(Stepper method, OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, Args&&... args);


template<SolverPolicy SP, Stepper S, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (!is_rich<SP>)
auto getVariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1);


template<SolverPolicy SP, Stepper S, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (is_rich<SP>)
auto getVariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T> events = {});

} // namespace ode::chaos


namespace ode::traits{

template<Stepper S, typename T, size_t N, hasRhsFunc<T> OdeType, typename DerivedVS, size_t NBase>
struct SolverVirtualTypeTraits<::ode::chaos::VariationalSolver<S, T, N, SolverPolicy::Virtual, OdeType, DerivedVS>, T, NBase, SolverPolicy::Virtual> {
    using type = chaos::ChaoticSolver<T, N, UtilPolicy::Virtual>;
};


template<Stepper S, typename T, size_t N, hasRhsFunc<T> OdeType, typename DerivedVS, size_t NBase>
struct SolverVirtualTypeTraits<::ode::chaos::VariationalSolver<S, T, N, SolverPolicy::RichVirtual, OdeType, DerivedVS>, T, NBase, SolverPolicy::RichVirtual> {
    using type = chaos::ChaoticSolver<T, N, UtilPolicy::RichVirtual>;
};

} // namespace ode

#endif // ODECRAFT_VARIATIONAL_SOLVERS_HPP
