#pragma once

/**
 * @file DenseOde/CraftedODE.hpp
 * @brief The high-level ODE driver of the compiled interface. Compiled by src/ODE.cpp.
 *
 * Links after Steppers and OdeHistory. This is the entry point most users want: it owns a
 * solver, runs it to a target time, and hands back an OdeResult or OdeSolution.
 */

#include <odecraft/Steppers/CraftedSteppers.hpp>
#include <odecraft/OdeResult/CraftedOdeResult.hpp>
#include "OdeInt.hpp"
#include "OdeInt_mem_impl.hpp" // IWYU pragma: keep


namespace ode::crafted{

using ::ode::EventCounter;

/**
 * @brief The polymorphic driver base: what every driver in the compiled interface *is*.
 *
 * ODE and VariationalODE are both one of these, and they are siblings rather than one
 * deriving from the other -- both are constructor wrappers over their own ode:: base. So a
 * handle that may hold either, `pbox::owner<OdeDriver<T>>`, has to be declared on this.
 * Declaring it on ODE<T> instead would compile, but ode::ODE<T, SD>::clone() returns the base,
 * and pbox::owner would static_cast that back down to the wrapper -- slicing a
 * VariationalODE on every copy.
 *
 * D is SD for an ODE, and VarSD for a VariationalODE. The two no longer share a base, since
 * their states are different types.
 *
 * Construct through ODE or VariationalODE; hold and pass around an OdeDriver.
 */
template<typename T, typename D = SD>
using OdeDriver = ::ode::ODE<T, D>;

/**
 * @brief Owns a solver and records its trajectory.
 *
 * A thin wrapper over ode::ODE<T, SD> that exists only to pin the system type. The underlying
 * constructor is a template over any type with an Rhs, but the compiled library ships just the
 * ode_t<T> instantiation, so anything else would compile here and fail at link with a mangled
 * error. Taking ode_t<T> by value turns that into a plain conversion error at the call site,
 * and lets you write the system inline: ODE<double> ode({.Rhs = f}, t0, q0, rtol, atol).
 */
template<typename T>
class ODE : public ::ode::ODE<T, SD>{

public:

    using Base = ::ode::ODE<T, SD>;

    ODE(
        ode_t<T> ode,
        T t0,
        State<T> q0,
        State<T> rtol,
        State<T> atol,
        T min_step = 0,
        T max_step = 0,
        T stepsize = 0,
        int dir = 1,
        EventList<T> events = {},
        Stepper method = Stepper::RK45
    ) : Base(std::move(ode), t0, std::move(q0), std::move(rtol), std::move(atol), min_step, max_step, stepsize, dir, std::move(events), method) {}

    DEFAULT_RULE_OF_FOUR(ODE)
};

} // namespace ode::crafted


namespace ode{

// ODE's constructor and init() are member templates over the system type. Only the
// ode_t<T> instantiation is compiled -- that is the one that would otherwise drag every
// stepper's implementation into the caller's translation unit, since init() dispatches
// over all of them.
#define ODECRAFT_EXTERN_ODE(T)                                                                      \
    extern template class EventCounter<T, crafted::SD>;                                             \
    extern template class EventCounter<T, crafted::VarSD>;                                          \
    extern template class ODE<T, crafted::SD>;                                                      \
    extern template class ODE<T, crafted::VarSD>;                                                   \
    extern template void ODE<T, crafted::SD>::init<crafted::ode_t<T>>(                              \
        crafted::ode_t<T>, T, crafted::State<T>, crafted::State<T>, crafted::State<T>, T, T, T, int, crafted::EventList<T>, Stepper);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_ODE)

#undef ODECRAFT_EXTERN_ODE

} // namespace ode

