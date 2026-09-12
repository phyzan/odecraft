#ifndef ODECRAFT_RK45_DOPRI_HPP
#define ODECRAFT_RK45_DOPRI_HPP

/**
 * @file RK45_DOPRI.hpp
 * @brief Dormand-Prince 5(4) adaptive Runge-Kutta stepper.
 *
 * Builds on the shared explicit Runge-Kutta machinery in DOPRI.hpp.
 */

#include <odecraft/Steppers/DOPRI.hpp>


namespace ode {

namespace detail{

template<size_t NSYS, typename T, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
T rk45_step_impl(T* result, const T* state, const T& h, size_t nsys,
                 const T* K0, T* K, T* KF, T* r,
                 const T& rtol, const T& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs);

} // namespace ode::detail

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class RK45 : public detail::BaseDispatcher<GetDerived<RK45<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

    using Base = detail::BaseDispatcher<GetDerived<RK45<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;

public:

    static constexpr size_t Nstages       = 6;
    static constexpr size_t Norder        = 5;
    static constexpr size_t INTERP_ORDER  = 4;
    static constexpr int    ERR_EST_ORDER = 4;
    static constexpr bool   IS_IMPLICIT   = false;

    RK45(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T> events = {});

    DEFAULT_RULE_OF_FOUR(RK45)

    Stepper method() const;

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

    detail::RKScratchSpace<T, N, Nstages-1> scratch_space;
    T                                       h_last_ = 0; // step size of the last sweep, replayed by set_coef_matrix
    mutable Array1D<T, N>                   K0_;
    mutable Array1D<T, N>                   KF_;
    mutable Array2D<T, N, 0>                coef_mat;
    mutable bool                            mat_is_set = false;

    T ERR_EXP = T(-1)/T(ERR_EST_ORDER+1); // Boost uses -1/(error_order+1) for both increase and decrease
    T INC_EXP = T(-1)/T(Norder);
    T MIN_ERR = T(1)/pow(T(5), Norder);
};

// ============================================================================
// Butcher tableau. It lives with the declarations because the solver holds its tables as
// static constexpr members, so the class cannot be completed without them.
// ============================================================================

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK45<T, N, SP, OdeType, Derived>::Atype RK45<T, N, SP, OdeType, Derived>::Amatrix() {
    return {T(0),        T(0),        T(0),        T(0),        T(0), T(0),
            T(1)/T(5),  T(0),        T(0),        T(0),        T(0), T(0),
            T(3)/T(40), T(9)/T(40), T(0),        T(0),        T(0), T(0),
            T(44)/T(45), T(-56)/T(15), T(32)/T(9), T(0),      T(0), T(0),
            T(19372)/T(6561), T(-25360)/T(2187), T(64448)/T(6561), T(-212)/T(729), T(0), T(0),
            T(9017)/T(3168), T(-355)/T(33), T(46732)/T(5247), T(49)/T(176), T(-5103)/T(18656), T(0)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK45<T, N, SP, OdeType, Derived>::Btype RK45<T, N, SP, OdeType, Derived>::Bmatrix(){
    return {T(35)/T(384),
            T(0),
            T(500)/T(1113),
            T(125)/T(192),
            T(-2187)/T(6784),
            T(11)/T(84)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK45<T, N, SP, OdeType, Derived>::Ctype RK45<T, N, SP, OdeType, Derived>::Cmatrix(){
    return {T(0),
            T(1)/T(5),
            T(3)/T(10),
            T(4)/T(5),
            T(8)/T(9),
            T(1)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK45<T, N, SP, OdeType, Derived>::Etype RK45<T, N, SP, OdeType, Derived>::Ematrix() {
    return {T(-71)/T(57600),
            T(0),
            T(71)/T(16695),
            T(-71)/T(1920),
            T(17253)/T(339200),
            T(-22)/T(525),
            T(1)/T(40)};
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
constexpr typename RK45<T, N, SP, OdeType, Derived>::Ptype RK45<T, N, SP, OdeType, Derived>::Pmatrix() {
    return {T(1),   -T(8048581381)/T(2820520608),   T(8663915743)/T(2820520608),   -T(12715105075)/T(11282082432),
            T(0),    T(0),                          T(0),                          T(0),
            T(0),    T(131558114200)/T(32700410799), -T(68118460800)/T(10900136933), T(87487479700)/T(32700410799),
            T(0),   -T(1754552775)/T(470086768),     T(14199869525)/T(1410260304),  -T(10690763975)/T(1880347072),
            T(0),    T(127303824393)/T(49829197408), -T(318862633887)/T(49829197408), T(701980252875)/T(199316789632),
            T(0),   -T(282668133)/T(205662961),       T(2019193451)/T(616988883),   -T(1453857185)/T(822651844),
            T(0),    T(40617522)/T(29380423),        -T(110615467)/T(29380423),     T(69997945)/T(29380423)};
}

namespace detail{


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
struct SolverTypeGetter<Stepper::RK45, T, N, SP, OdeType, Derived>{
    using type = RK45<T, N, SP, OdeType, Derived>;
};

} // namespace ode::detail

} // namespace ode

#endif // ODECRAFT_RK45_DOPRI_HPP
