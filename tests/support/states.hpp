#pragma once

/**
 * @file states.hpp
 * @brief The state descriptors the tests run on, and a few ways to build their states.
 */

#include <algorithm>
#include <initializer_list>
#include <string_view>
#include <tuple>

#include <odecraft/odecraft.hpp>
#include "scalars.hpp" // IWYU pragma: keep

namespace odetest {

using ode::state_t, ode::real_state_t, ode::jacmat_t, ode::value_t;

/// @brief States of N components, held in a std::array.
template<size_t N>
using StaticSize = ode::contracts::VectorDescriptor<N>;

/// @brief States whose size is a runtime value, held in a std::vector.
using RuntimeSize = ode::contracts::VectorDescriptor<0>;

/// @brief Both kinds of state of N components, for the tests that run on each.
template<size_t N>
using StateKinds = std::tuple<StaticSize<N>, RuntimeSize>;

/// @brief Whether the size of the states of SD is only known at runtime.
template<typename SD>
inline constexpr bool is_runtime_sized = !std::is_trivially_copyable_v<state_t<double, SD>>;

/// @brief A section name for the kind of state of SD.
template<typename SD>
inline constexpr std::string_view kind_name = is_runtime_sized<SD> ? "runtime-sized state" : "static-sized state";


/// @brief The state of SD with the given components.
template<typename SD, typename T>
state_t<T, SD> make_state(std::initializer_list<T> values){
    using State = state_t<T, SD>;
    if constexpr (std::is_constructible_v<State, std::initializer_list<T>>){
        return State(values);
    } else {
        State state{};
        std::copy(values.begin(), values.end(), state.begin());
        return state;
    }
}

/// @brief A state of SD with every component equal to `value`, shaped like `like`.
template<typename SD, typename T>
real_state_t<T, SD> uniform_like(const state_t<T, SD>& like, const T& value){
    real_state_t<T, SD> out{};
    SD::resize_like(out, like);
    SD::for_each([&value](T& out_i){ out_i = value; }, out);
    return out;
}

/// @brief The state of a variational solver over SD: the main state, and its deviation vector.
template<typename SD, typename T>
using var_state_t = state_t<T, ode::contracts::VarSD<SD>>;

template<typename SD, typename T>
var_state_t<SD, T> make_var_state(std::initializer_list<T> main, std::initializer_list<T> variation){
    return var_state_t<SD, T>{.main = make_state<SD, T>(main), .variation = make_state<SD, T>(variation)};
}

/// @brief A Jacobian matrix of SD, shaped for states like `like`.
template<typename SD, typename T>
jacmat_t<T, SD> make_jacmat(const state_t<T, SD>& like){
    jacmat_t<T, SD> J{};
    SD::JD::resize_like(J, like);
    return J;
}

/// @brief The current state of a solver, which may lie between its last two steps.
template<typename Solver>
typename Solver::State current_state(const Solver& solver){
    typename Solver::State q = solver.vector_new();
    solver.FillCurrentState(q);
    return q;
}

} // namespace odetest
