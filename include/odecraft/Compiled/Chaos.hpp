#ifndef ODECRAFT_COMPILED_CHAOS_HPP
#define ODECRAFT_COMPILED_CHAOS_HPP

/**
 * @file Compiled/Chaos.hpp
 * @brief Variational integration and Lyapunov exponents. Compiled by src/Chaos.cpp.
 *
 * Links last: builds on ODE and on every stepper.
 *
 * A variational solver integrates the original system alongside its variational equations,
 * renormalising the deviation vector every `period` time units, which yields the largest
 * Lyapunov exponent. The state vector it exposes is 2*nsys long: the system state followed
 * by the deviation vector.
 *
 * @note The variational equations need the Jacobian of the original system. With the
 *       type-erased ode_t<T> there is no autodiff to fall back on, so supplying `.Jac` is
 *       strongly recommended here; otherwise everything is finite-differenced.
 */

#include <odecraft/Compiled/ODE.hpp>
#include <odecraft/Chaos/VariationalSolvers.hpp>
#include <odecraft/Chaos/VariationalSolvers_mem_impl.hpp>


namespace ode::crafted{

/// @brief Abstract variational solver: an OdeRichSolver that also reports Lyapunov data.
template<typename T>
using ChaoticSolver = ::ode::chaos::ChaoticSolver<T, 0, ::ode::UtilPolicy::RichVirtual>;

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
 * @param q0       Initial state of the original system.
 * @param delta_q0 Initial deviation vector; same size as @p q0.
 * @param period   Time between renormalisations of the deviation vector.
 * @see crafted::ODE for the remaining parameters.
 */
template<typename T>
class VariationalODE : public ::ode::chaos::VariationalODE<T, 0>{

public:

    using Base = ::ode::chaos::VariationalODE<T, 0>;

    // Same constructor as the base, but only for ode_t<T>.
    VariationalODE(
        ode_t<T> ode,
        T t0,
        View1D<T> q0,
        View1D<T> delta_q0,
        T period,
        T rtol,
        T atol,
        T min_step = 0,
        T max_step = 0,
        T stepsize = 0,
        int dir = 1,
        EventList<T> events = {},
        Stepper method = Stepper::RK45
    ) : Base(std::move(ode), t0, q0, delta_q0, period, rtol, atol, min_step, max_step, stepsize, dir, std::move(events), method) {}

    DEFAULT_RULE_OF_FOUR(VariationalODE)
};


/**
 * @brief Construct a variational solver of the requested method.
 * @param method   Which stepper to use for the augmented system.
 * @param ode      Rhs and (strongly recommended) Jacobian of the *original* system.
 * @param t0       Initial time.
 * @param q0       Initial state of the original system.
 * @param delta_q0 Initial deviation vector; same size as @p q0.
 * @param period   Time between renormalisations of the deviation vector.
 * @see make_vsolver for the tolerance and step-size parameters.
 */
template<typename T>
BoxedChaoticSolver<T> make_variational_solver(
    Stepper method,
    ode_t<T> ode,
    T t0,
    View1D<T> q0,
    View1D<T> delta_q0,
    T period,
    T rtol,
    T atol,
    T min_step = 0,
    T max_step = 0,
    T stepsize = 0,
    int dir = 1,
    EventList<T> events = {}
);

#define ODECRAFT_EXTERN_CHAOS_FACTORY(T)                                        \
    extern template BoxedChaoticSolver<T> make_variational_solver<T>(           \
        Stepper, ode_t<T>, T, View1D<T>, View1D<T>, T, T, T, T, T, T, int, EventList<T>);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_CHAOS_FACTORY)

#undef ODECRAFT_EXTERN_CHAOS_FACTORY

} // namespace ode::crafted


/**
 * @brief Every instantiation one variational stepper needs, for one scalar type.
 *
 * @param SPEC `extern template` (header) or `template` (src unit).
 * @param S    Stepper enum value driving the augmented system.
 * @param CLS  The stepper class that enum selects, e.g. RK45.
 * @param T    Scalar type.
 *
*/
#define ODECRAFT_VARIATIONAL_SOLVER(S, T)                                                                   \
    ::ode::chaos::VariationalSolver<S, T, 0, ::ode::SolverPolicy::RichVirtual, ::ode::crafted::ode_t<T>>

#define ODECRAFT_VARIATIONAL_SYS(T)                                                                         \
    ode::chaos::VariationalOdeSys<T, 0, ::ode::crafted::ode_t<T>>

#define ODECRAFT_VARIATIONAL_STEPPER(CLS, S, T)                                                             \
    ode::CLS<T, 0, ::ode::SolverPolicy::RichVirtual, ODECRAFT_VARIATIONAL_SYS(T),                           \
             ODECRAFT_VARIATIONAL_SOLVER(S, T)>


#define ODECRAFT_VARIATIONAL_SET(SPEC, S, CLS, T)                                                           \
    SPEC class ODECRAFT_VARIATIONAL_STEPPER(CLS, S, T);                                                     \
    SPEC class ODECRAFT_VARIATIONAL_SOLVER(S, T);                                                           


#define ODECRAFT_DECLARE_VARIATIONAL(T, S, CLS) ODECRAFT_VARIATIONAL_SET(extern template, S, CLS, T)
#define ODECRAFT_DEFINE_VARIATIONAL(T, S, CLS)  ODECRAFT_VARIATIONAL_SET(template, S, CLS, T)

/// @brief Every stepper's variational set for one scalar, plus the driver.
#define ODECRAFT_VARIATIONAL_ALL(T)                                             \
    ODECRAFT_DECLARE_VARIATIONAL(T, ::ode::Stepper::Euler,  Euler)              \
    ODECRAFT_DECLARE_VARIATIONAL(T, ::ode::Stepper::RK4,    RK4)                \
    ODECRAFT_DECLARE_VARIATIONAL(T, ::ode::Stepper::RK23,   RK23)               \
    ODECRAFT_DECLARE_VARIATIONAL(T, ::ode::Stepper::RK45,   RK45)               \
    ODECRAFT_DECLARE_VARIATIONAL(T, ::ode::Stepper::DOP853, DOP853)             \
    ODECRAFT_DECLARE_VARIATIONAL(T, ::ode::Stepper::BDF,    BDF)                \
    extern template class ::ode::chaos::VariationalODE<T, 0>;


ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_VARIATIONAL_ALL)



namespace ode::chaos{


#define ODECRAFT_EXTERN_VARIATIONAL_ODE(T)                                          \
    extern template VariationalODE<T, 0>::VariationalODE(                                   \
        crafted::ode_t<T>, T, View1D<T, 0>, View1D<T, 0>, T, T, T, T, T, T, int, EventList<T>, Stepper);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_VARIATIONAL_ODE)

#undef ODECRAFT_EXTERN_VARIATIONAL_ODE

} // namespace ode::chaos

#endif // ODECRAFT_COMPILED_CHAOS_HPP
