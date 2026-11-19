#pragma once

/**
 * @file checks.hpp
 * @brief Comparisons of states and Jacobians, as plain numbers a CHECK can print.
 */

#include "output.hpp"
#include "states.hpp"

#include <cmath>
#include <limits>
#include <string>

namespace odetest {

/// @brief The number of components of a state of SD.
template<typename SD, typename State>
size_t size_of(const State& state){
    size_t n = 0;
    SD::for_each([&n](const auto& /*x*/){ ++n; }, state);
    return n;
}

/// @brief max_i |actual_i - expected_i| over two states of SD. NaN if any component is NaN, or if
/// the two differ in size, which a state that was moved from or never sized does.
template<typename SD, typename State>
Magnitude max_error(const State& actual, const State& expected){
    if (size_of<SD>(actual) != size_of<SD>(expected)){
        return {std::numeric_limits<double>::quiet_NaN()};
    }
    Magnitude err{0};
    SD::for_each([&err](const auto& a, const auto& e){
        err = worst(abs_error(a, e), err);
    }, actual, expected);
    return err;
}

/// @brief Whether two states of SD have the same size and hold exactly the same values.
template<typename SD, typename State>
bool same_state(const State& a, const State& b){
    if (size_of<SD>(a) != size_of<SD>(b)){
        return false;
    }
    bool same = true;
    SD::for_each([&same](const auto& x, const auto& y){
        same = same && (x == y);
    }, a, b);
    return same;
}

/// @brief max |A(i, j) - B(i, j)| over two n x n matrices.
template<typename Matrix>
Magnitude max_matrix_error(const Matrix& A, const Matrix& B, size_t n){
    Magnitude err{0};
    for (size_t i = 0; i < n; i++){
        for (size_t j = 0; j < n; j++){
            err = worst(abs_error(A(i, j), B(i, j)), err);
        }
    }
    return err;
}

/// @brief A state written out, for INFO messages.
template<typename SD, typename State>
std::string to_string(const State& state){
    std::string out = "{";
    SD::for_each([&out](const auto& x){
        out += (out.size() > 1 ? ", " : "") + to_string(x);
    }, state);
    return out + "}";
}

} // namespace odetest
