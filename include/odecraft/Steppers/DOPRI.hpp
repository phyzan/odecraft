#ifndef ODECRAFT_DOPRI_HPP
#define ODECRAFT_DOPRI_HPP

/**
 * @file DOPRI.hpp
 * @brief Machinery shared by the adaptive Runge-Kutta steppers.
 *
 * Butcher-tableau holder, and the method-agnostic step-size control and
 * dense-output assembly used by RK23, RK45 and DOP853. The solvers themselves live in
 * RK23_DOPRI.hpp and RK45_DOPRI.hpp; this header carries nothing specific to either.
 */

#include <odecraft/Core/RichSolver/RichBase.hpp>


namespace ode::detail{


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
// Shared adaptive Runge-Kutta building blocks, used by RK23, RK45 and DOP853.
// The per-stage arithmetic itself (e.g. h * (a21*K0 + ...)) stays hardcoded in each
// solver's step_impl for performance; only the surrounding, method-agnostic machinery
// (step-size control, dense-output coefficient assembly) is shared here.
// ============================================================================

/// @brief Distance, in elements, between consecutive stage rows of n elements (K1, K2, ...).
/// This padding is more cache-friendly and can increase performance
template<typename T>
constexpr size_t rk_stage_stride(size_t n){
    constexpr size_t line = 64;
    if constexpr (sizeof(T) > line || line % sizeof(T) != 0){
        return n;
    } else {
        constexpr size_t per_line = line / sizeof(T);
        const size_t lines = (n + per_line - 1) / per_line;
        return n == 0 ? 0 : (lines | 1) * per_line;
    }
}

/// @brief Dense-output coefficients from the stage derivatives - K0, the Nstages-1 rows of K
/// (`ld` elements apart) and the final stage KF - and the interpolation weights P
/// (Nstages+1 x order).
template<typename T>
void rk_interp_matrix(T* coef_mat, const T* K, const T* K0, const T* KF, const T* P, size_t Nstages, size_t order, size_t n, size_t ld);

/// @brief Shared step-size control loop: repeatedly calls step_fn(t_new, q_new, t, h, q) -> err_norm,
/// halving/growing habs until the local error is accepted (mirrors scipy/boost step control).
/// Same in/out contract as BaseSolver::adapt_impl.
template<size_t N, typename T, typename StepFn>
StepResult rk_adapt_step(T& t_new, T& habs_new, T* q_new, const T& t, const T& habs, const T* q, size_t n,
                          const T& min_step, const T& max_step, const T& min_step_abs,
                          const T& safety, const T& max_factor, const T& min_factor,
                          const T& err_exp, const T& inc_exp, const T& min_err,
                          int direction, StepFn&& step_fn);

} // namespace ode::detail



#endif // ODECRAFT_DOPRI_HPP
