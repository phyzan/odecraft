#pragma once

/**
 * @file Chaos/CraftedChaos.hpp
 * @brief Variational integration and Lyapunov exponents. Compiled by src/Chaos.cpp.
 *
 * Links last: builds on ODE and on every stepper.
 *
 * A variational solver integrates the original system alongside its variational equations,
 * renormalising the deviation vector every `period` time units, which yields the largest
 * Lyapunov exponent. The state it exposes is a VarState<T>: `.main`, the system state, and
 * `.variation`, the deviation vector.
 *
 * @note The variational equations need the Jacobian of the original system. With the
 *       type-erased ode_t<T> there is no autodiff to fall back on, so supplying `.Jac` is
 *       strongly recommended here; otherwise everything is finite-differenced.
 */

#include <odecraft/DenseOde/CraftedODE.hpp>
#include "VariationalSolvers.hpp"
#include "VariationalSolvers_mem_impl.hpp" // IWYU pragma: keep


namespace ode::crafted{

/// @brief Abstract variational solver: an OdeRichSolver that also reports Lyapunov data.
template<typename T>
using ChaoticSolver = ::ode::chaos::ChaoticSolver<T, SD, ::ode::UtilPolicy::RichVirtual>;

/// @brief Owning handle to a ChaoticSolver.
template<typename T>
using BoxedChaoticSolver = pbox::Box<ChaoticSolver<T>>;

/**
 * @brief ODE driver that additionally records renormalisation times and Lyapunov values.
 *
 * Pinned to ode_t<T> for the same reason as crafted::ODE -- see the note there.
 *
 * @param ode      Rhs and (strongly recommended) Jacobian of the *original* system.
 * @param t0       Initial time.
 * @param q0       Initial state: `.main` of the original system, and `.variation` the initial
 *                 deviation vector, of the same shape.
 * @param period   Time between renormalisations of the deviation vector.
 * @param rtol     Relative error tolerance, per component of @p q0.
 * @param atol     Absolute error tolerance, per component of @p q0.
 * @param method   The stepper for the augmented system: its id in Steppers.
 * @see crafted::ODE for the remaining parameters.
 */
template<typename T>
class VariationalODE : public ::ode::chaos::VariationalODE<T, SD>{

public:

    using Base = ::ode::chaos::VariationalODE<T, SD>;

    // Same constructor as the base, but only for ode_t<T>, and with the stepper chosen at runtime.
    VariationalODE(
        ode_t<T> ode,
        T t0,
        VarState<T> q0,
        T period,
        VarState<T> rtol,
        VarState<T> atol,
        T min_step = 0,
        T max_step = 0,
        T stepsize = 0,
        int dir = 1,
        EventList<T, VarState<T>> events = {},
        Steppers::Id method = Steppers::id<tags::RK45>
    ) : Base(Steppers::from_stepper<Base>(method, [&]<typename S>(){
            return Base(S{}, std::move(ode), t0, std::move(q0), period, std::move(rtol), std::move(atol), min_step, max_step, stepsize, dir, std::move(events));
        })) {}

    DEFAULT_RULE_OF_FOUR(VariationalODE)
};


/**
 * @brief Construct a variational solver of the requested method.
 * @param method   Which stepper to use for the augmented system: its id in Steppers.
 * @param ode      Rhs and (strongly recommended) Jacobian of the *original* system.
 * @param t0       Initial time.
 * @param q0       Initial state: `.main` of the original system, and `.variation` the initial
 *                 deviation vector, of the same shape.
 * @param period   Time between renormalisations of the deviation vector.
 * @param rtol     Relative error tolerance, per component of @p q0.
 * @param atol     Absolute error tolerance, per component of @p q0.
 * @see make_rich_vsolver for the step-size parameters.
 */
template<typename T>
BoxedChaoticSolver<T> make_variational_solver(
    Steppers::Id method,
    ode_t<T> ode,
    T t0,
    VarState<T> q0,
    T period,
    VarState<T> rtol,
    VarState<T> atol,
    T min_step = 0,
    T max_step = 0,
    T stepsize = 0,
    int dir = 1,
    EventList<T, VarState<T>> events = {}
);

#define ODECRAFT_EXTERN_CHAOS_FACTORY(T)                                        \
    extern template BoxedChaoticSolver<T> make_variational_solver<T>(           \
        Steppers::Id, ode_t<T>, T, VarState<T>, T, VarState<T>, VarState<T>, T, T, T, int, EventList<T, VarState<T>>);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_CHAOS_FACTORY)

#undef ODECRAFT_EXTERN_CHAOS_FACTORY

} // namespace ode::crafted


/**
 * @brief Every instantiation one variational stepper needs, for one scalar type.
 *
 * @param SPEC `extern template` (header) or `template` (src unit).
 * @param CLS  The stepper class driving the augmented system, e.g. RK45, whose tag is tags::CLS.
 * @param T    Scalar type.
 *
*/
#define ODECRAFT_VARIATIONAL_SOLVER(CLS, T)                                                                 \
    ::ode::chaos::VariationalSolver<::ode::tags::CLS, T, ::ode::crafted::SD, ::ode::crafted::ode_t<T>, ::ode::SolverPolicy::RichVirtual>

#define ODECRAFT_VARIATIONAL_SYS(T)                                                                         \
    ode::chaos::VariationalOdeSys<T, ::ode::crafted::SD, ::ode::crafted::ode_t<T>>

#define ODECRAFT_VARIATIONAL_STEPPER(CLS, T)                                                                \
    ode::CLS<T, ::ode::crafted::VarSD, ODECRAFT_VARIATIONAL_SYS(T), ::ode::SolverPolicy::RichVirtual,       \
             ODECRAFT_VARIATIONAL_SOLVER(CLS, T)>

#define ODECRAFT_VARIATIONAL_BASE(CLS, T)                                                                   \
    ::ode::BaseSolver<ODECRAFT_VARIATIONAL_SOLVER(CLS, T), T, ::ode::crafted::VarSD,                        \
                      ODECRAFT_VARIATIONAL_SYS(T), ::ode::SolverPolicy::RichVirtual>

#define ODECRAFT_VARIATIONAL_RICH(CLS, T)                                                                   \
    ::ode::RichSolver<ODECRAFT_VARIATIONAL_SOLVER(CLS, T), T, ::ode::crafted::VarSD,                        \
                      ODECRAFT_VARIATIONAL_SYS(T), ::ode::SolverPolicy::RichVirtual>

#define ODECRAFT_VARIATIONAL_SET(SPEC, CLS, T)                                                              \
    SPEC class ODECRAFT_VARIATIONAL_BASE(CLS, T);                                                           \
    SPEC class ODECRAFT_VARIATIONAL_RICH(CLS, T);                                                           \
    SPEC class ODECRAFT_VARIATIONAL_STEPPER(CLS, T);                                                        \
    SPEC class ODECRAFT_VARIATIONAL_SOLVER(CLS, T);                                                         


#define ODECRAFT_DECLARE_VARIATIONAL(T, CLS) ODECRAFT_VARIATIONAL_SET(extern template, CLS, T)
#define ODECRAFT_DEFINE_VARIATIONAL(T, CLS)  ODECRAFT_VARIATIONAL_SET(template, CLS, T)

// ODECRAFT_FOR_EACH_STEPPER passes the class first
#define ODECRAFT_DECLARE_VARIATIONAL_OF(CLS, T) ODECRAFT_DECLARE_VARIATIONAL(T, CLS)

/// @brief Every stepper's variational set for one scalar, plus the driver.
#define ODECRAFT_VARIATIONAL_ALL(T)                                             \
    ODECRAFT_FOR_EACH_STEPPER(ODECRAFT_DECLARE_VARIATIONAL_OF, T)               \
    extern template class ::ode::chaos::VariationalODE<T, ::ode::crafted::SD>;


ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_VARIATIONAL_ALL)



namespace ode::chaos{


// The constructors crafted::VariationalODE dispatches over, one per compiled stepper
#define ODECRAFT_VARIATIONAL_ODE_CTOR(CLS, SPEC, T)                                                 \
    SPEC VariationalODE<T, crafted::SD>::VariationalODE(                                            \
        ::ode::tags::CLS, crafted::ode_t<T>, T, crafted::VarState<T>, T, crafted::VarState<T>, crafted::VarState<T>, T, T, T, int, crafted::EventList<T, crafted::VarState<T>>);

#define ODECRAFT_EXTERN_VARIATIONAL_ODE(T) ODECRAFT_FOR_EACH_STEPPER(ODECRAFT_VARIATIONAL_ODE_CTOR, extern template, T)

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_VARIATIONAL_ODE)

#undef ODECRAFT_EXTERN_VARIATIONAL_ODE

} // namespace ode::chaos
