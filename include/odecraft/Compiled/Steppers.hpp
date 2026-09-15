#ifndef ODECRAFT_COMPILED_STEPPERS_HPP
#define ODECRAFT_COMPILED_STEPPERS_HPP

/**
 * @file Compiled/Steppers.hpp
 * @brief Every compiled stepper, plus the runtime factories that pick one.
 *
 * Links after the individual stepper units: the factories only need those classes to be
 * complete, since each one's constructor and vtable already live in its own unit.
 */

#include <odecraft/Compiled/SolverBase.hpp>
#include <odecraft/Compiled/Euler.hpp>
#include <odecraft/Compiled/RK4.hpp>
#include <odecraft/Compiled/RK23_DOPRI.hpp>
#include <odecraft/Compiled/RK45_DOPRI.hpp>
#include <odecraft/Compiled/DOP853.hpp>
#include <odecraft/Compiled/BDF.hpp>


namespace ode::crafted{

/**
 * @brief Construct a solver of the requested method, behind the OdeRichSolver interface.
 *
 * There is deliberately only one factory. Every compiled solver is RichVirtual, and a
 * RichVirtual solver already is an OdeSolver, so callers who do not want events can simply
 * pass no events and hold the result by OdeSolver<T>.
 *
 * @param method    Which stepper to use.
 * @param ode       Rhs and optional Jacobian.
 * @param t0        Initial time.
 * @param q0        Initial state; its size fixes the system size.
 * @param rtol      Relative error tolerance.
 * @param atol      Absolute error tolerance.
 * @param min_step  Lower bound on the step size; 0 for none.
 * @param max_step  Upper bound on the step size; 0 for none.
 * @param stepsize  Initial step size; 0 to pick one automatically.
 * @param dir       Integration direction, +1 forward or -1 backward.
 * @param events    Events to monitor during integration.
 */
template<typename T>
BoxedRichSolver<T> make_rich_vsolver(
    Stepper method,
    ode_t<T> ode,
    T t0,
    View1D<T> q0,
    T rtol,
    T atol,
    T min_step = 0,
    T max_step = 0,
    T stepsize = 0,
    int dir = 1,
    EventList<T> events = {}
);


#define ODECRAFT_EXTERN_SOLVER_FACTORY(T)                                       \
    extern template BoxedRichSolver<T> make_rich_vsolver<T>(                    \
        Stepper, ode_t<T>, T, View1D<T>, T, T, T, T, T, int, EventList<T>);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_SOLVER_FACTORY)

#undef ODECRAFT_EXTERN_SOLVER_FACTORY

} // namespace ode::crafted

#endif // ODECRAFT_COMPILED_STEPPERS_HPP
