#pragma once

/**
 * @file Steppers/CraftedSteppers.hpp
 * @brief Every compiled stepper, plus the runtime factories that pick one.
 *
 * Links after the individual stepper units: the factories only need those classes to be
 * complete, since each one's constructor and vtable already live in its own unit.
 */

#include <odecraft/Core/CraftedBase.hpp>
#include "Euler/CraftedEuler.hpp" // IWYU pragma: keep
#include "RK4/CraftedRK4.hpp" // IWYU pragma: keep
#include "DOPRI/RK23/CraftedRK23.hpp" // IWYU pragma: keep
#include "DOPRI/RK45/CraftedRK45.hpp" // IWYU pragma: keep
#include "DOPRI/DOP853/CraftedDOP853.hpp" // IWYU pragma: keep
#include "BDF/CraftedBDF.hpp" // IWYU pragma: keep
#include "Steppers.hpp" // the catalog ode::Steppers


namespace ode::crafted{

/// @brief The catalog of the compiled steppers. The compiled interface chooses a stepper at
/// runtime, by its id here: e.g. Steppers::id<tags::RK45>, or Steppers::getID("RK45").
using ::ode::Steppers;

/// @brief The stepper tags, e.g. tags::RK45.
namespace tags = ::ode::tags;

/**
 * @brief Construct a solver of the requested method, behind the OdeRichSolver interface.
 *
 * There is deliberately only one factory. Every compiled solver is RichVirtual, and a
 * RichVirtual solver already is an OdeSolver, so callers who do not want events can simply
 * pass no events and hold the result by OdeSolver<T>.
 *
 * @param method    Which stepper to use: its id in Steppers.
 * @param ode       Rhs and optional Jacobian.
 * @param t0        Initial time.
 * @param q0        Initial state; its shape fixes the shape of the system.
 * @param rtol      Relative error tolerance, per component of the state.
 * @param atol      Absolute error tolerance, per component of the state.
 * @param min_step  Lower bound on the step size; 0 for none.
 * @param max_step  Upper bound on the step size; 0 for none.
 * @param stepsize  Initial step size; 0 to pick one automatically.
 * @param dir       Integration direction, +1 forward or -1 backward.
 * @param events    Events to monitor during integration.
 */
template<typename T>
BoxedRichSolver<T> make_rich_vsolver(
    Steppers::Id method,
    ode_t<T> ode,
    T t0,
    State<T> q0,
    State<T> rtol,
    State<T> atol,
    T min_step = 0,
    T max_step = 0,
    T stepsize = 0,
    int dir = 1,
    EventList<T> events = {}
);


#define ODECRAFT_EXTERN_SOLVER_FACTORY(T)                                       \
    extern template BoxedRichSolver<T> make_rich_vsolver<T>(                    \
        Steppers::Id, ode_t<T>, T, State<T>, State<T>, State<T>, T, T, T, int, EventList<T>);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_SOLVER_FACTORY)

#undef ODECRAFT_EXTERN_SOLVER_FACTORY

} // namespace ode::crafted
