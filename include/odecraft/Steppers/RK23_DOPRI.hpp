#ifndef ODECRAFT_RK23_DOPRI_HPP
#define ODECRAFT_RK23_DOPRI_HPP

/**
 * @file RK23_DOPRI.hpp
 * @brief Bogacki-Shampine 3(2) adaptive Runge-Kutta stepper.
 *
 * Builds on the shared adaptive Runge-Kutta machinery in DOPRI.hpp.
 */

#include <odecraft/Steppers/DOPRI.hpp>


namespace ode {

namespace detail{

/// @brief One Bogacki-Shampine 3(2) stage sweep. Same contract as rk45_step_impl: on entry K[0]
/// holds the derivative at the start of the step; the sweep fills K[1..3], K[3] being the final
/// (FSAL) stage. Returns the scaled error norm.
template<typename T, typename SD, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
T rk23_step_impl(T& t_new, state_t<T, SD>& q_new, cache_t<T> t, cache_t<T> h, const state_t<T, SD>& q,
                 std::array<state_t<T, SD>, 4>& K, state_t<T, SD>& r, real_state_t<T, SD>& worker,
                 const real_state_t<T, SD>& rtol, const real_state_t<T, SD>& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs);

} // namespace ode::detail

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived = void>
class RK23 : public detail::BaseDispatcher<GetDerived<RK23<T, SD, JD, OdeType, SP, Derived>, Derived>, T, SD, JD, OdeType, SP>{

    using Base = detail::BaseDispatcher<GetDerived<RK23<T, SD, JD, OdeType, SP, Derived>, Derived>, T, SD, JD, OdeType, SP>;

public:
    using State = state_t<T, SD>;
    using RealState = Base::RealState;

    static constexpr size_t Nstages       = 3;
    static constexpr size_t Norder        = 3;
    static constexpr size_t INTERP_ORDER  = 3;
    static constexpr int    ERR_EST_ORDER = 2;
    static constexpr bool   IS_IMPLICIT   = false;

    RK23(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T, State> events = {});

    DEFAULT_RULE_OF_FOUR(RK23)

    Stepper method() const;

    void        Reset();

protected:

    void        ReAdjust(const State& new_vector);
    StepResult  adapt_impl(T& t_new, T& habs_new, State& q_new, const T& t, const T& habs, const State& q);
    void        interp_impl(State& result, const T& t) const;
    auto        local_interp() const;

private:

    using Atype = CArray2D<T, Nstages, Nstages>;
    using Btype = Array1D<T, Nstages>;
    using Ctype = Array1D<T, Nstages>;
    using Etype = Array1D<T, Nstages+1>;
    using Ptype = CArray2D<T, Nstages+1, INTERP_ORDER>;

    static constexpr Atype Amatrix();
    static constexpr Btype Bmatrix();
    static constexpr Ctype Cmatrix();
    static constexpr Etype Ematrix();
    static constexpr Ptype Pmatrix();

    T           step_impl(T& t_new, State& q_new, const T& t, const T& h, const State& q);

    void        set_coef_matrix() const;

    detail::CoefTable<T, &Amatrix> A;
    detail::CoefTable<T, &Bmatrix> B;
    detail::CoefTable<T, &Cmatrix> C;
    detail::CoefTable<T, &Ematrix> E;
    detail::CoefTable<T, &Pmatrix> P;

    // the stages of the last step: K_[0] the derivative at its start, K_[Nstages] at its end (FSAL)
    mutable std::array<State, Nstages+1>    K_;
    mutable std::array<State, INTERP_ORDER> coef_mat;

    // the stage input and the scaled errors of a sweep
    worker_t<State>                         r_scratch_;
    worker_t<RealState>                     real_scratch;

    mutable bool                            mat_is_set = false;

    T ERR_EXP = T(-1)/T(ERR_EST_ORDER+1); // Boost uses -1/(error_order+1) for both increase and decrease
    T INC_EXP = T(-1)/T(Norder);
    T MIN_ERR = T(1)/pow(T(5), Norder);
};

// ============================================================================
// Butcher tableau. It lives with the declarations because the solver holds its tables as
// static constexpr members, so the class cannot be completed without them.
// ============================================================================

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
constexpr RK23<T, SD, JD, OdeType, SP, Derived>::Atype RK23<T, SD, JD, OdeType, SP, Derived>::Amatrix() {
    return {T(0),   T(0),      T(0),
            T(1)/2, T(0),      T(0),
            T(0),   T(3)/T(4), T(0),};
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
constexpr RK23<T, SD, JD, OdeType, SP, Derived>::Btype RK23<T, SD, JD, OdeType, SP, Derived>::Bmatrix(){
    return {T(2)/9,
            T(1)/3,
            T(4)/9,};
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
constexpr RK23<T, SD, JD, OdeType, SP, Derived>::Ctype RK23<T, SD, JD, OdeType, SP, Derived>::Cmatrix(){
    return {T(0),
            T(1)/T(2),
            T(3)/T(4),};
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
constexpr RK23<T, SD, JD, OdeType, SP, Derived>::Etype RK23<T, SD, JD, OdeType, SP, Derived>::Ematrix() {
    return {T(5)/T(72),
            T(-1)/T(12),
            T(-1)/T(9),
            T(1)/T(8),};
}

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
constexpr RK23<T, SD, JD, OdeType, SP, Derived>::Ptype RK23<T, SD, JD, OdeType, SP, Derived>::Pmatrix() {
    return {T(1),  -T(4)/T(3),  T(5)/T(9),
            T(0),   T(1),      -T(2)/T(3),
            T(0),   T(4)/T(3), -T(8)/T(9),
            T(0),  -T(1),       T(1),};
}

namespace detail{

template<typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP, typename Derived>
struct SolverTypeGetter<Stepper::RK23, SP, T, SD, JD, OdeType, Derived>{
    using type = RK23<T, SD, JD, OdeType, SP, Derived>;
};

} // namespace ode::detail

} // namespace ode

#endif // ODECRAFT_RK23_DOPRI_HPP
