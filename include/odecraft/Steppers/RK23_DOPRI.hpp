#ifndef ODECRAFT_RK23_DOPRI_HPP
#define ODECRAFT_RK23_DOPRI_HPP

/**
 * @file RK23_DOPRI.hpp
 * @brief Bogacki-Shampine 3(2) adaptive Runge-Kutta stepper.
 *
 * Builds on the shared explicit Runge-Kutta machinery in DOPRI.hpp.
 */

#include <odecraft/Steppers/DOPRI.hpp>


namespace ode {

namespace detail{

/// @brief One Bogacki-Shampine 3(2) stage sweep. Same contract as rk45_step_impl: `K` holds
/// stages K1..K2 (Nstages-1 rows of n), `KF` the final FSAL stage, `K0` the derivative at the
/// start of the step. Returns the scaled error norm.
template<size_t NSYS, typename T, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
T rk23_step_impl(T* result, const T* state, const T& h, size_t nsys,
                 const T* K0, T* K, T* KF, T* r,
                 const T& rtol, const T& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs);

} // namespace ode::detail

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class RK23 : public detail::BaseDispatcher<GetDerived<RK23<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

    using Base = detail::BaseDispatcher<GetDerived<RK23<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;

public:

    static constexpr size_t Nstages       = 3;
    static constexpr size_t Norder        = 3;
    static constexpr size_t INTERP_ORDER  = 3;
    static constexpr int    ERR_EST_ORDER = 2;
    static constexpr bool   IS_IMPLICIT   = false;


    RK23(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T> events = {});

    DEFAULT_RULE_OF_FOUR(RK23)

    Stepper  method() const;

    void        Reset();

protected:

    void        ReAdjust(const T* new_vector);
    StepResult  adapt_impl(T* res, const T* state);
    void        interp_impl(T* result, const T& t) const;
    auto        local_interp() const;

private:

    using Atype = Array2D<T, Nstages, Nstages, Allocation::Stack>;
    using Btype = Array1D<T, Nstages, Allocation::Stack>;
    using Ctype = Array1D<T, Nstages, Allocation::Stack>;
    using Etype = Array1D<T, Nstages+1, Allocation::Stack>;
    using Ptype = Array2D<T, Nstages+1, INTERP_ORDER, Allocation::Stack>;

    static constexpr Atype Amatrix();
    static constexpr Btype Bmatrix();
    static constexpr Ctype Cmatrix();
    static constexpr Etype Ematrix();
    static constexpr Ptype Pmatrix();

    T           step_impl(T* result, const T* state, const T& h);

    void        set_coef_matrix() const;

    detail::CoefTable<T, &Amatrix> A;
    detail::CoefTable<T, &Bmatrix> B;
    detail::CoefTable<T, &Cmatrix> C;
    detail::CoefTable<T, &Ematrix> E;
    detail::CoefTable<T, &Pmatrix> P;

    detail::RKScratchSpace<T, N, Nstages-1>              scratch_space;
    T                                                   h_last_ = 0; // replayed by set_coef_matrix
    mutable Array1D<T, N>                               K0_;  // derivative at the start of the step
    mutable Array1D<T, N>                               KF_;  // final (FSAL) stage
    mutable Array2D<T, N, 0>                            coef_mat_;
    mutable bool                                        mat_is_set_ = false;

    T ERR_EXP = T(-1)/T(ERR_EST_ORDER+1); // Boost uses -1/(error_order+1) for both increase and decrease
    T INC_EXP = T(-1)/T(Norder);
    T MIN_ERR = T(1)/pow(T(5), Norder);
};

// ============================================================================
// Butcher tableau. It lives with the declarations because the solver holds its tables as
// static constexpr members, so the class cannot be completed without them.
// ============================================================================

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK23<T, N, SP, OdeType, Derived>::Atype RK23<T, N, SP, OdeType, Derived>::Amatrix() {
    return {T(0),   T(0),      T(0),
            T(1)/2, T(0),      T(0),
            T(0),   T(3)/T(4), T(0)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK23<T, N, SP, OdeType, Derived>::Btype RK23<T, N, SP, OdeType, Derived>::Bmatrix(){
    return {T(2)/9,
            T(1)/3,
            T(4)/9};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK23<T, N, SP, OdeType, Derived>::Ctype RK23<T, N, SP, OdeType, Derived>::Cmatrix(){
    return {T(0),
            T(1)/T(2),
            T(3)/T(4)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK23<T, N, SP, OdeType, Derived>::Etype RK23<T, N, SP, OdeType, Derived>::Ematrix() {
    return {T(5)/T(72),
            T(-1)/T(12),
            T(-1)/T(9),
            T(1)/T(8)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK23<T, N, SP, OdeType, Derived>::Ptype RK23<T, N, SP, OdeType, Derived>::Pmatrix() {
    return {T(1),  -T(4)/T(3),  T(5)/T(9),
            T(0),   T(1),      -T(2)/T(3),
            T(0),   T(4)/T(3), -T(8)/T(9),
            T(0),  -T(1),       T(1)};
}

namespace detail{

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
struct SolverTypeGetter<Stepper::RK23, T, N, SP, OdeType, Derived>{
    using type = RK23<T, N, SP, OdeType, Derived>;
};

} // namespace ode::detail

} // namespace ode

#endif // ODECRAFT_RK23_DOPRI_HPP
