#pragma once

/**
 * @file steppers.hpp
 * @brief How the tests build a solver of a built-in stepper.
 */

#include "settings.hpp"
#include "states.hpp"

namespace odetest {

/**
 * @brief A solver of the stepper, as the tests usually build one.
 *
 * The tolerances, step and direction come from `settings`, unless the arguments after `q0` say
 * otherwise. Any policy works, the default being the static one with events.
 */
template<typename Stepper, typename SD, ode::SolverPolicy SP = ode::SolverPolicy::RichStatic, typename T, typename OdeType>
auto make_solver(OdeType ode, const T& t0, const state_t<T, SD>& q0, int dir = 1, ode::EventList<T, state_t<T, SD>> events = {}, T max_step = 0){
    const Settings<T> s = settings<Stepper, T>();
    return ode::getSolver<SD, SP>(Stepper{}, std::move(ode), t0, q0,
        uniform_like<SD, T>(q0, s.rtol), uniform_like<SD, T>(q0, s.atol), T(0), max_step, s.stepsize, dir, std::move(events));
}

} // namespace odetest
