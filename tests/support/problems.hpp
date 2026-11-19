#pragma once

/**
 * @file problems.hpp
 * @brief Reference ODE systems with known solutions or Jacobians.
 *
 * A system written with `auto` parameters can be called on dual numbers, which is what makes
 * the solvers differentiate it automatically (JacPolicy::Autodiff). The Plain* and *WithJac
 * variants pin the state type, so they take the finite-difference and the explicit Jacobian
 * paths instead.
 */

#include "states.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>

namespace odetest {


/**
 * @brief The harmonic oscillator x'' = -x, as q = (x, v), q' = (v, -x).
 *
 * The reference trajectory used throughout is the one through q(0) = (1, 0): q(t) = (cos t, -sin t).
 * Its first component crosses zero at t = pi/2 + k*pi, decreasing for even k and increasing for odd k.
 */
struct Oscillator {

    static void Rhs(auto& out, const auto& /*t*/, const auto& q){
        out[0] = q[1];
        out[1] = -q[0];
    }

    /// @brief The reference trajectory at time t
    template<typename SD, typename T>
    static state_t<T, SD> at(const T& t){
        const T c = cos(t);
        const T s = sin(t);
        return make_state<SD, T>({c, T(-s)});
    }

    /// @brief The constant Jacobian [[0, 1], [-1, 0]]
    template<typename SD, typename T>
    static void jacobian(jacmat_t<T, SD>& J){
        J(0, 0) = 0; J(0, 1) = 1;
        J(1, 0) = -1; J(1, 1) = 0;
    }
};

/// @brief The oscillator for states of SD only: no autodiff, so the solvers use finite differences.
template<typename T, typename SD>
struct PlainOscillator {
    void Rhs(state_t<T, SD>& out, const T& t, const state_t<T, SD>& q) const { Oscillator::Rhs(out, t, q); }
};

/// @brief The oscillator with its exact Jacobian (JacPolicy::Exact).
template<typename T, typename SD>
struct OscillatorWithJac : PlainOscillator<T, SD> {
    void Jac(jacmat_t<T, SD>& J, const T& /*t*/, const state_t<T, SD>& /*q*/) const { Oscillator::jacobian<SD, T>(J); }
};


/**
 * @brief The oscillator, until its right-hand side turns into NaN past t_fail.
 *
 * Integrating past t_fail (in the direction `dir`) makes every stepper fail its step with a
 * non-finite state, which is how the tests stop a solver at a known point.
 */
struct PoisonedOscillator {
    double t_fail;
    int dir = 1;

    void Rhs(auto& out, const auto& t, const auto& q) const {
        const bool poisoned = dir * (to_double(t) - t_fail) > 0;
        const double poison = poisoned ? std::numeric_limits<double>::quiet_NaN() : 0.0;
        out[0] = q[1] + poison;
        out[1] = -q[0];
    }
};


/**
 * @brief Uniform motion, q' = (1, -1): q(t) = q(t0) + (t - t0) * (1, -1).
 *
 * Linear in t, so every stepper integrates it exactly, whatever its step, and so does every
 * dense output, but for rounding (see uniform_motion_tolerance). That makes event times and
 * dense output exactly predictable even across huge steps, which is what the tests of several
 * events within one step rely on.
 */
struct UniformMotion {
    static void Rhs(auto& out, const auto& /*t*/, const auto& q){
        out[0] = q[0]*0 + 1;
        out[1] = q[1]*0 - 1;
    }
};

/**
 * @brief How closely a stepper's states and dense output follow uniform motion, in unit steps.
 *
 * 20 epsilon, but 100 for DOP853. Its dense output adds up coefficients of several hundred,
 * which cancel exactly on a constant right-hand side but are rounded, and summed, in T: that
 * leaves up to tens of epsilon in the interpolated state, and so in an event time located on it.
 */
template<typename Stepper, typename T>
double uniform_motion_tolerance(){
    return (std::is_same_v<Stepper, ode::tags::DOP853> ? 100 : 20) * epsilon<T>();
}


/**
 * @brief The stiff Prothero-Robinson problem, q' = lambda*(q - cos t) - sin t.
 *
 * Its solution through q(0) = 1 is cos t for any lambda, and lambda = -1000 makes it stiff:
 * an explicit stepper is held to |h| < 2/|lambda| by stability alone, an implicit one is not.
 */
struct StiffDecay {
    static constexpr double lambda = -1000;

    static void Rhs(auto& out, const auto& t, const auto& q){
        out[0] = lambda * (q[0] - cos(t)) - sin(t);
    }
};


/**
 * @brief A nonlinear system with a dense, state- and time-dependent Jacobian:
 *     f = (sin(q0) + q0*q1,  exp(-q1)*q0^2 + t*q1)
 */
struct Nonlinear {

    static void Rhs(auto& out, const auto& t, const auto& q){
        out[0] = sin(q[0]) + q[0]*q[1];
        out[1] = exp(-q[1])*q[0]*q[0] + t*q[1];
    }

    /// @brief J(i, j) = df_i/dq_j
    template<typename SD, typename T>
    static void jacobian(jacmat_t<T, SD>& J, const T& t, const state_t<T, SD>& q){
        const T e = exp(-q[1]);
        J(0, 0) = T(cos(q[0])) + q[1];  J(0, 1) = q[0];
        J(1, 0) = 2*q[0]*e;             J(1, 1) = T(-q[0]*q[0]*e) + t;
    }

    /// @brief B(i, j) = sum_k d^2 f_i / (dq_j dq_k) * dq_k, the derivative of J along dq
    template<typename SD, typename T>
    static void jacobian_derivative(jacmat_t<T, SD>& B, const state_t<T, SD>& q, const state_t<T, SD>& dq){
        const T e = exp(-q[1]);
        B(0, 0) = T(-sin(q[0])*dq[0]) + dq[1];
        B(0, 1) = dq[0];
        B(1, 0) = 2*e*dq[0] - 2*q[0]*e*dq[1];
        B(1, 1) = -2*q[0]*e*dq[0] + q[0]*q[0]*e*dq[1];
    }
};

/// @brief The nonlinear system for states of SD only (JacPolicy::Approx).
template<typename T, typename SD>
struct PlainNonlinear {
    void Rhs(state_t<T, SD>& out, const T& t, const state_t<T, SD>& q) const { Nonlinear::Rhs(out, t, q); }
};

/// @brief The nonlinear system with its exact Jacobian (JacPolicy::Exact).
template<typename T, typename SD>
struct NonlinearWithJac : PlainNonlinear<T, SD> {
    void Jac(jacmat_t<T, SD>& J, const T& t, const state_t<T, SD>& q) const { Nonlinear::jacobian<SD, T>(J, t, q); }
};

/// @brief The nonlinear system as type-erased callables, with a Jacobian that may be null (JacPolicy::Nullable).
template<typename T, typename SD>
using NullableNonlinear = ode::OdeData<
    std::function<void(state_t<T, SD>&, T, const state_t<T, SD>&)>,
    std::function<void(jacmat_t<T, SD>&, T, const state_t<T, SD>&)>>;

template<typename T, typename SD>
NullableNonlinear<T, SD> nullable_nonlinear(bool with_jacobian){
    NullableNonlinear<T, SD> ode{.Rhs = [](state_t<T, SD>& out, T t, const state_t<T, SD>& q){ Nonlinear::Rhs(out, t, q); }};
    if (with_jacobian){
        ode.Jac = [](jacmat_t<T, SD>& J, T t, const state_t<T, SD>& q){ Nonlinear::jacobian<SD, T>(J, t, q); };
    }
    return ode;
}


/**
 * @brief The linear saddle q' = (q0, -q1), whose finite-time Lyapunov exponent is known exactly.
 *
 * A deviation vector d obeys d' = diag(1, -1) d, so |d(t)| = |(e^t d0, e^-t d1)|, and the
 * largest Lyapunov exponent estimated over [t0, t0 + T] is log(|d(T)|/|d(0)|)/T, for either sign of T.
 */
struct Saddle {

    static void Rhs(auto& out, const auto& /*t*/, const auto& q){
        out[0] = q[0];
        out[1] = -q[1];
    }

    template<typename SD, typename T>
    static void jacobian(jacmat_t<T, SD>& J){
        J(0, 0) = 1; J(0, 1) = 0;
        J(1, 0) = 0; J(1, 1) = -1;
    }

    /// @brief log(|d(T)|/|d(0)|)/T for the deviation d(0) = (d0, d1), after the time T (of either sign)
    template<typename T>
    static T lyapunov_exponent(const T& T_elapsed, const T& d0, const T& d1){
        const T grown = exp(2*T_elapsed)*d0*d0 + exp(-2*T_elapsed)*d1*d1;
        const T initial = d0*d0 + d1*d1;
        return T(log(grown/initial)) / (2*T_elapsed);
    }
};

template<typename T, typename SD>
struct PlainSaddle {
    void Rhs(state_t<T, SD>& out, const T& t, const state_t<T, SD>& q) const { Saddle::Rhs(out, t, q); }
};

template<typename T, typename SD>
struct SaddleWithJac : PlainSaddle<T, SD> {
    void Jac(jacmat_t<T, SD>& J, const T& /*t*/, const state_t<T, SD>& /*q*/) const { Saddle::jacobian<SD, T>(J); }
};

} // namespace odetest
