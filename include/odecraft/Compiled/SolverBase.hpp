#ifndef ODECRAFT_COMPILED_SOLVER_BASE_HPP
#define ODECRAFT_COMPILED_SOLVER_BASE_HPP

/**
 * @file Compiled/SolverBase.hpp
 * @brief The abstract solver interfaces every compiled stepper is reached through.
 *
 * Header-only: OdeSolver and OdeRichSolver are pure interfaces with nothing to compile.
 * The CRTP bases (BaseSolver, RichSolver) are likewise never instantiated on their own --
 * they are parameterised on their Derived stepper, so no two translation units could share
 * an instantiation. Their templated member functions do come in here, so that a stepper
 * held by its concrete type still accepts an arbitrary observer or checkpoint container.
 */

#include <odecraft/Compiled/Events.hpp>
#include <odecraft/Core/RichSolver/RichBase.hpp>
#include <odecraft/Core/BaseSolver/BaseSolver_mem_impl.hpp>
#include <odecraft/Core/VirtualBase.hpp>
#include <odecraft/Core/SolverFactory.hpp>


/**
 * @brief Every instantiation one stepper class needs, for one scalar type.
 *
 * @param SPEC `extern template` in a Compiled/ header, `template` in the matching src/ unit.
 * @param CLS  Unqualified stepper class name in namespace ode, e.g. RK45.
 * @param T    Scalar type.
 *
 * Only SolverPolicy::RichVirtual is shipped. The compiled interface deliberately does not
 * offer a policy choice.
 */
#define ODECRAFT_STEPPER_ARGS(T) T, 0, ::ode::SolverPolicy::RichVirtual, ::ode::crafted::ode_t<T>

#define ODECRAFT_STEPPER_TYPE(CLS, T) ode::CLS<ODECRAFT_STEPPER_ARGS(T)>
#define ODECRAFT_STEPPER_BASE(CLS, T) ode::BaseSolver<ODECRAFT_STEPPER_TYPE(CLS, T), ODECRAFT_STEPPER_ARGS(T)>
#define ODECRAFT_STEPPER_RICH(CLS, T) ode::RichSolver<ODECRAFT_STEPPER_TYPE(CLS, T), ODECRAFT_STEPPER_ARGS(T)>

#define ODECRAFT_STEPPER_SET(SPEC, CLS, T)                                                              \
    SPEC class ODECRAFT_STEPPER_BASE(CLS, T);                                                           \
    SPEC class ODECRAFT_STEPPER_RICH(CLS, T);                                                           \
    SPEC class ODECRAFT_STEPPER_TYPE(CLS, T);                                                           

/// @brief Declare a stepper's instantiations as provided by the library (Compiled/ headers).
/// Scalar first, so it can be driven straight from ODECRAFT_FOR_EACH_SCALAR.
#define ODECRAFT_DECLARE_STEPPER(T, CLS) ODECRAFT_STEPPER_SET(extern template, CLS, T)

/// @brief Define them (src/ units). Must appear after the matching declarations.
#define ODECRAFT_DEFINE_STEPPER(T, CLS)  ODECRAFT_STEPPER_SET(template, CLS, T)


namespace ode::crafted{


using ::ode::Stepper;

template<typename T>
using OdeSolver = ::ode::OdeSolver<T, 0>;

template<typename T>
using OdeRichSolver = ::ode::OdeRichSolver<T, 0>;

/// @brief Owning handle to an OdeRichSolver.
template<typename T>
using BoxedRichSolver = ::ode::BoxedSolver<T, 0, ::ode::UtilPolicy::RichVirtual>;

/// @brief Owning handle to a dense-output interpolant.
template<typename T>
using BoxedInterp = ::ode::BoxedInterp<T, 0>;

} // namespace ode::crafted

#endif // ODECRAFT_COMPILED_SOLVER_BASE_HPP
