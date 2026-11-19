#pragma once

/**
 * @file crafted_test.hpp
 * @brief The one include of the tests of the pre-compiled interface, ode::crafted.
 *
 * The library comes in through <odecraft/Crafted.hpp> only, which declares what the compiled
 * library defines. So every solver these tests run comes out of the library, and one it does not
 * ship fails to link rather than being compiled here.
 *
 * The conventions are those of odecraft_test.hpp, over the scalar types the library ships,
 * ODECRAFT_CRAFTED_SCALARS. A stepper is chosen at runtime here, by its name in the catalog.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <odecraft/Crafted.hpp>

#include "scalars.hpp"
#include "settings.hpp" // IWYU pragma: keep
#include "output.hpp" // IWYU pragma: keep

#include <limits>
#include <string>
#include <vector>

/// @brief The scalar types the compiled library ships, for TEMPLATE_TEST_CASE.
#define ODECRAFT_CRAFTED_SCALARS float, double, long double, ode::crafted::mpreal_t

namespace odetest {

namespace oc = ode::crafted;

/// @brief The names of the compiled steppers in their catalog, ode::crafted::Steppers.
inline const std::vector<std::string> stepper_names = {"Euler", "RK4", "RK23", "RK45", "DOP853", "BDF"};

/// @brief A state shaped like another, with every component set to value.
template<typename T>
oc::State<T> filled_like(const oc::State<T>& like, const T& value){
    oc::State<T> out = like;
    for (size_t i = 0; i < out.size(); i++){
        out[i] = value;
    }
    return out;
}

/// @brief max_i |a_i - b_i| over the components of two states, NaN if they differ in size.
template<typename T>
Magnitude max_error(const oc::State<T>& a, const oc::State<T>& b){
    if (a.size() != b.size()){
        return {std::numeric_limits<double>::quiet_NaN()};
    }
    Magnitude err{0};
    for (size_t i = 0; i < a.size(); i++){
        err = worst(abs_error(T(a[i]), T(b[i])), err);
    }
    return err;
}

/// @brief The current state of a solver behind the virtual interface.
template<typename Solver>
auto current_vector(const Solver& solver){
    auto q = solver.get_new_vector();
    solver.get_current_vector(q);
    return q;
}


/// @brief The harmonic oscillator x' = v, v' = -x, from (1, 0) at t = 0: (cos t, -sin t).
namespace oscillator {

template<typename T>
oc::State<T> at(const T& t){
    return {T(cos(t)), T(-sin(t))};
}

/// @brief Its system, with its Jacobian or without one, which leaves it to finite differences.
template<typename T>
oc::ode_t<T> system(bool with_jacobian = true){
    oc::ode_t<T> ode{.Rhs = [](oc::State<T>& out, T /*t*/, const oc::State<T>& q){
        out[0] = q[1];
        out[1] = -q[0];
    }};
    if (with_jacobian){
        ode.Jac = [](oc::JacMat<T>& J, T /*t*/, const oc::State<T>& /*q*/){
            J(0, 0) = 0;
            J(0, 1) = 1;
            J(1, 0) = -1;
            J(1, 1) = 0;
        };
    }
    return ode;
}

} // namespace oscillator


/// @brief The linear saddle x' = x, y' = -y, whose largest Lyapunov exponent is known exactly.
namespace saddle {

template<typename T>
oc::ode_t<T> system(bool with_jacobian = true){
    oc::ode_t<T> ode{.Rhs = [](oc::State<T>& out, T /*t*/, const oc::State<T>& q){
        out[0] = q[0];
        out[1] = -q[1];
    }};
    if (with_jacobian){
        ode.Jac = [](oc::JacMat<T>& J, T /*t*/, const oc::State<T>& /*q*/){
            J(0, 0) = 1;
            J(0, 1) = 0;
            J(1, 0) = 0;
            J(1, 1) = -1;
        };
    }
    return ode;
}

/// @brief The finite-time exponent after a time `elapsed` (negative backward) from the deviation
/// (d0, d1): log(|d(t)| / |d(0)|) / elapsed.
template<typename T>
T lyapunov_exponent(const T& elapsed, const T& d0, const T& d1){
    const T grown = exp(T(2 * elapsed)) * d0 * d0 + exp(T(-2 * elapsed)) * d1 * d1;
    return T(log(T(grown / T(d0 * d0 + d1 * d1))) / T(2 * elapsed));
}

} // namespace saddle

} // namespace odetest
