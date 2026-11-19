#pragma once

/**
 * @file settings.hpp
 * @brief The built-in steppers, how the tests set each of them up, and loops over types.
 *
 * Shared by both test executables: it needs only the stepper tags and their catalog, which
 * both interfaces declare.
 */

#include <odecraft/Steppers/Steppers.hpp>

#include <tuple>
#include <type_traits>

namespace odetest {

namespace tags = ode::tags;

/// @brief Every stepper the library ships.
using AllSteppers = std::tuple<tags::Euler, tags::RK4, tags::RK23, tags::RK45, tags::DOP853, tags::BDF>;

/// @brief Calls f.template operator()<E>() for every type E of the tuple, in order.
template<typename Tuple, typename F>
void for_each_type(F&& f){
    [&]<typename... E>(std::type_identity<std::tuple<E...>>){
        (f.template operator()<E>(), ...);
    }(std::type_identity<Tuple>{});
}

template<typename Stepper>
inline constexpr bool is_fixed_step = std::is_same_v<Stepper, tags::Euler> || std::is_same_v<Stepper, tags::RK4>;


/**
 * @brief How the tests run a stepper, and how accurate it then is.
 *
 * Fixed-step steppers are given a step, adaptive ones a tolerance. `error` bounds the distance
 * to the exact solution of the oscillator over |t - t0| <= 10 with these settings, a few times
 * above what the stepper actually achieves. Since the oscillator moves at unit speed, it also
 * bounds the error of a time located on it, such as an event's.
 */
template<typename T>
struct Settings {
    T rtol;
    T atol;
    T stepsize;
    double error;
};

template<typename Stepper, typename T>
Settings<T> settings(){
    // Single precision cannot hold the tolerances of the others: its own are bounded by roundoff
    constexpr bool single = std::is_same_v<T, float>;
    // Fixed-step steppers only use their tolerances to pick a step when they are not given one
    if constexpr (std::is_same_v<Stepper, tags::Euler>){
        return single ? Settings<T>{T(1e-6), T(1e-6), T(1e-3), 3e-2} : Settings<T>{T(1e-6), T(1e-6), T(1e-4), 2e-3};
    } else if constexpr (std::is_same_v<Stepper, tags::RK4>){
        return single ? Settings<T>{T(1e-6), T(1e-6), T(1e-2), 1e-3} : Settings<T>{T(1e-6), T(1e-6), T(1e-2), 1e-8};
    } else if constexpr (std::is_same_v<Stepper, tags::RK23>){
        return single ? Settings<T>{T(1e-6), T(1e-6), T(0), 3e-4} : Settings<T>{T(1e-9), T(1e-9), T(0), 2e-7};
    } else if constexpr (std::is_same_v<Stepper, tags::RK45>){
        return single ? Settings<T>{T(1e-6), T(1e-6), T(0), 4e-5} : Settings<T>{T(1e-10), T(1e-10), T(0), 3e-9};
    } else if constexpr (std::is_same_v<Stepper, tags::DOP853>){
        return single ? Settings<T>{T(1e-6), T(1e-6), T(0), 6e-5} : Settings<T>{T(1e-12), T(1e-12), T(0), 2e-11};
    } else {
        static_assert(std::is_same_v<Stepper, tags::BDF>, "settings: unknown stepper");
        return single ? Settings<T>{T(1e-6), T(1e-6), T(0), 4e-4} : Settings<T>{T(1e-10), T(1e-10), T(0), 3e-7};
    }
}

/**
 * @brief The orders of a stepper's first step from initial conditions, and of its dense output.
 *
 * A step of size h is then off by O(h^(step + 1)), and its dense output by O(h^(dense + 1)).
 * BDF starts at order 1 (backward Euler), and RK4 interpolates with cubic Hermite polynomials.
 */
template<typename Stepper>
struct Orders {
    static constexpr int step = std::is_same_v<Stepper, tags::Euler> ? 1
                              : std::is_same_v<Stepper, tags::RK4> ? 4
                              : std::is_same_v<Stepper, tags::RK23> ? 3
                              : std::is_same_v<Stepper, tags::RK45> ? 5
                              : std::is_same_v<Stepper, tags::DOP853> ? 8
                              : 1;
    static constexpr int dense = std::is_same_v<Stepper, tags::RK4> ? 3
                               : std::is_same_v<Stepper, tags::RK23> ? 3
                               : std::is_same_v<Stepper, tags::RK45> ? 4
                               : std::is_same_v<Stepper, tags::DOP853> ? 7
                               : 1;
};


/// @brief settings<Stepper, T>() of the stepper with an id in the catalog of the built-in ones,
/// for the interfaces that choose a stepper at runtime.
template<typename T>
Settings<T> settings_of(ode::Steppers::Id id){
    return ode::Steppers::from_stepper<Settings<T>>(id, []<typename Stepper>(){ return settings<Stepper, T>(); });
}

} // namespace odetest
