#ifndef ODECRAFT_DOPRI_HPP
#define ODECRAFT_DOPRI_HPP

//https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method


#include <odecraft/Core/RichBase.hpp>


namespace ode {

namespace detail{

template<typename T, size_t NSYS, size_t NCOUNT> class StaticRKScratch;
template<typename T, size_t NSYS, size_t NCOUNT> class DynamicRKScratch;

// Same rule as the solver's own scratch (see scratch_is_static in BaseSolver.hpp): automatic
// storage only for a compile-time size and a trivial scalar; everything else is heap-backed
// and allocated once.
template<typename T, size_t NSYS, size_t NCOUNT>
using RKScratchSpace = std::conditional_t<scratch_is_static<T, NSYS>,
                                          StaticRKScratch<T, NSYS, NCOUNT>,
                                          DynamicRKScratch<T, NSYS, NCOUNT>>;

// A Butcher-tableau constant (A, B, C, E, P, ...) as a class member: a static constexpr
// table (shared, zero per-instance storage) when T is arithmetic, since it is then a
// compile-time constant; otherwise (e.g. T = mpfr::mpreal, which cannot be constant-evaluated)
// a plain instance member computed once at construction. Either way it's used the same:
// Coef(1,0), Coef[i], Coef.data().
template<typename T, auto Generator>
struct StaticCoefTable{
    using Matrix = decltype(Generator());
    static constexpr Matrix table = Generator();

    template<typename... Idx>
    inline constexpr const auto& operator()(Idx... idx) const { return table(idx...); }
    inline constexpr const auto& operator[](size_t i) const { return table[i]; }
    inline constexpr const auto* data() const { return table.data(); }
};

template<typename T, auto Generator>
struct DynamicCoefTable{
    using Matrix = decltype(Generator());
    Matrix table = Generator();

    template<typename... Idx>
    inline const auto& operator()(Idx... idx) const { return table(idx...); }
    inline const auto& operator[](size_t i) const { return table[i]; }
    inline const auto* data() const { return table.data(); }
};

template<typename T, auto Generator>
using CoefTable = std::conditional_t<std::is_arithmetic_v<T>, StaticCoefTable<T, Generator>, DynamicCoefTable<T, Generator>>;



// ============================================================================
// Shared explicit Runge-Kutta building blocks, used by RK23, RK45 and DOP853.
// The per-stage arithmetic itself (e.g. h * (a21*K0 + ...)) stays hardcoded in each
// solver's step_impl for performance; only the surrounding, method-agnostic machinery
// (step-size control, dense-output coefficient assembly) is shared here.
// ============================================================================

/// @brief Build the dense-output polynomial coefficient matrix (n x order) from stage
/// derivatives K (Nstages+1 rows) and interpolation weights P (Nstages+1 x order).
template<typename T>
void rk_interp_matrix(T* coef_mat, const T* K, const T* K0, const T* KF, const T* P, size_t Nstages, size_t order, size_t n);


template<size_t NSYS, typename T, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
T rk45_step_impl(T* result, const T* state, const T& h, size_t nsys,
                 const T* K0, T* K, T* KF, T* r,
                 const T& rtol, const T& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs);

/// @brief One Bogacki-Shampine 3(2) stage sweep. Same contract as rk45_step_impl: `K` holds
/// stages K1..K2 (Nstages-1 rows of n), `KF` the final FSAL stage, `K0` the derivative at the
/// start of the step. Returns the scaled error norm.
template<size_t NSYS, typename T, typename Atab, typename Btab, typename Ctab, typename Etab, typename RhsFn>
T rk23_step_impl(T* result, const T* state, const T& h, size_t nsys,
                 const T* K0, T* K, T* KF, T* r,
                 const T& rtol, const T& atol,
                 const Atab& A, const Btab& B, const Ctab& C, const Etab& E, RhsFn&& rhs);

/// @brief Shared step-size control loop: repeatedly calls step_fn(res, state, h) -> err_norm,
/// halving/growing habs until the local error is accepted (mirrors scipy/boost step control).
template<typename T, typename StepFn>
StepResult rk_adapt_step(T* res, const T* state, size_t n,
                          const T& min_step, const T& max_step, const T& min_step_abs,
                          const T& safety, const T& max_factor, const T& min_factor,
                          const T& err_exp, const T& inc_exp, const T& min_err,
                          int direction, StepFn&& step_fn);

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

    RK23(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1) requires (!is_rich<SP>);

    RK23(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T> events = {}) requires (is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(RK23)

    Integrator  method() const;

    auto        local_interp() const;

    void        Reset();

protected:

    void        ReAdjust(const T* new_vector);

    StepResult  adapt_impl(T* res, const T* state);

    void        interp_impl(T* result, const T& t) const;

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


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived = void>
class RK45 : public detail::BaseDispatcher<GetDerived<RK45<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>{

    using Base = detail::BaseDispatcher<GetDerived<RK45<T, N, SP, OdeType, Derived>, Derived>, T, N, SP, OdeType>;

public:

    static constexpr size_t Nstages       = 6;
    static constexpr size_t Norder        = 5;
    static constexpr size_t INTERP_ORDER  = 4;
    static constexpr int    ERR_EST_ORDER = 4;
    static constexpr bool   IS_IMPLICIT   = false;

    RK45(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1) requires (!is_rich<SP>);

    RK45(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T> events = {}) requires (is_rich<SP>);

    DEFAULT_RULE_OF_FOUR(RK45)

    Integrator method() const;

    auto local_interp() const;

    void        Reset();

protected:

    void        ReAdjust(const T* new_vector);

    StepResult  adapt_impl(T* res, const T* state);

    void        interp_impl(T* result, const T& t) const;

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

namespace detail{

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
struct SolverTypeGetter<Integrator::RK23, T, N, SP, OdeType, Derived>{
    using type = RK23<T, N, SP, OdeType, Derived>;
};


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
struct SolverTypeGetter<Integrator::RK45, T, N, SP, OdeType, Derived>{
    using type = RK45<T, N, SP, OdeType, Derived>;
};

} // namespace ode::detail

} // namespace ode

#endif // ODECRAFT_DOPRI_HPP
