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


/// @brief The stages that a weighted sum of stages runs over, e.g. stages<0, 3, 4>: those with a
/// nonzero weight, so that the zeros of a Butcher tableau cost nothing.
template<size_t... j>
inline constexpr std::index_sequence<j...> stages{};

/// @brief r = q + h * sum_j A(i, j) * K[j], over the stages j listed.
template<typename SD, typename State, size_t NK, typename T, typename Tab, size_t... j>
void rk_stage(State& r, const State& q, const T& h, const std::array<State, NK>& K, const Tab& A, size_t i, std::index_sequence<j...>);

/// @brief r = q + h * sum_j b(j) * K[j], over the stages j listed.
template<typename SD, typename State, size_t NK, typename T, typename Tab, size_t... j>
void rk_stage(State& r, const State& q, const T& h, const std::array<State, NK>& K, const Tab& b, std::index_sequence<j...>);

/// @brief out = h * sum_j D(i, j) * K[j], over the stages j listed.
template<typename SD, typename State, size_t NK, typename T, typename Tab, size_t... j>
void rk_dot(State& out, const T& h, const std::array<State, NK>& K, const Tab& D, size_t i, std::index_sequence<j...>);

/// @brief Dense-output coefficients from the stage derivatives K of a step: K[0] at its start,
/// K[NS] at its end. coef_mat[j] = sum_k P(k, j) * K[k], over the stages k listed (those whose
/// row of P is not zero), added in the order listed.
template<typename T, typename SD, size_t NS, size_t ORD, size_t... k>
void rk_interp_matrix(
    std::array<state_t<T, SD>, ORD>& coef_mat, // mat[i][j] = j-th elem of i-th state
    const std::array<state_t<T, SD>, NS+1>& K,
    const CArray2D<T, NS+1, ORD>& P,
    std::index_sequence<k...>);

/// @brief Shared step-size control loop: repeatedly calls step_fn(t_new, q_new, t, h, q) -> err_norm,
/// halving/growing habs until the local error is accepted (mirrors scipy/boost step control).
/// Same in/out contract as BaseSolver::adapt_impl.
template<typename T, typename SD, typename StepFn>
StepResult rk_adapt_step(
    T& t_new, T& habs_new, state_t<T, SD>& q_new,
    cache_t<T> t, cache_t<T> habs, cache_t<state_t<T, SD>> q,
    cache_t<T> min_step, cache_t<T> max_step, cache_t<T> min_step_abs,
    cache_t<T> safety, cache_t<T> max_factor, cache_t<T> min_factor,
    cache_t<T> err_exp, cache_t<T> inc_exp, cache_t<T> min_err,
    int direction, StepFn&& step_fn);

} // namespace ode::detail



#endif // ODECRAFT_DOPRI_HPP
