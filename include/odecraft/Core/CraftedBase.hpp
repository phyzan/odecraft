#pragma once

/**
 * @file Core/CraftedBase.hpp
 * @brief The abstract solver interfaces every compiled stepper is reached through.
 *
 * Header-only: OdeSolver and OdeRichSolver are pure interfaces with nothing to compile.
 * The CRTP bases (BaseSolver, RichSolver) are likewise never instantiated on their own --
 * they are parameterised on their Derived stepper, so no two translation units could share
 * an instantiation. Their templated member functions do come in here, so that a stepper
 * held by its concrete type still accepts an arbitrary observer or checkpoint container.
 */

#include <odecraft/Events/CraftedEvents.hpp>
#include "RichSolver/RichBase.hpp" // IWYU pragma: keep
#include "BaseSolver/BaseSolver_mem_impl.hpp" // IWYU pragma: keep
#include "VirtualBase.hpp"
#include "SolverFactory.hpp"


/**
 * @brief Every instantiation one stepper class needs, for one scalar type.
 *
 * @param SPEC `extern template` in a Crafted header, `template` in the matching src/ unit.
 * @param CLS  Unqualified stepper class name in namespace ode, e.g. RK45.
 * @param T    Scalar type.
 *
 * Only SolverPolicy::RichVirtual is shipped. The compiled interface deliberately does not
 * offer a policy choice.
 */
#define ODECRAFT_STEPPER_ARGS(T) T, ::ode::crafted::SD, ::ode::crafted::ode_t<T>, ::ode::SolverPolicy::RichVirtual

#define ODECRAFT_STEPPER_TYPE(CLS, T) ode::CLS<ODECRAFT_STEPPER_ARGS(T)>
#define ODECRAFT_STEPPER_BASE(CLS, T) ode::BaseSolver<ODECRAFT_STEPPER_TYPE(CLS, T), ODECRAFT_STEPPER_ARGS(T)>
#define ODECRAFT_STEPPER_RICH(CLS, T) ode::RichSolver<ODECRAFT_STEPPER_TYPE(CLS, T), ODECRAFT_STEPPER_ARGS(T)>

#define ODECRAFT_STEPPER_SET(SPEC, CLS, T)                                                              \
    SPEC class ODECRAFT_STEPPER_BASE(CLS, T);                                                           \
    SPEC class ODECRAFT_STEPPER_RICH(CLS, T);                                                           \
    SPEC class ODECRAFT_STEPPER_TYPE(CLS, T);                                                           

/// @brief Declare a stepper's instantiations as provided by the library (Crafted headers).
/// Scalar first, so it can be driven straight from ODECRAFT_FOR_EACH_SCALAR.
#define ODECRAFT_EXTERN_STEPPER(T, CLS) ODECRAFT_STEPPER_SET(extern template, CLS, T)

/// @brief Define them (src/ units). Must appear after the matching declarations.
#define ODECRAFT_DEFINE_STEPPER(T, CLS)  ODECRAFT_STEPPER_SET(template, CLS, T)


namespace ode::crafted{


using ::ode::Stepper;

template<typename T>
using OdeSolver = ::ode::OdeSolver<T, SD>;

template<typename T>
using OdeRichSolver = ::ode::OdeRichSolver<T, SD>;

/// @brief Owning handle to an OdeRichSolver.
template<typename T>
using BoxedRichSolver = ::ode::BoxedSolver<T, SD, ::ode::UtilPolicy::RichVirtual>;

/// @brief Owning handle to a dense-output interpolant.
template<typename T, typename D = SD>
using BoxedInterp = ::ode::BoxedInterp<T, D>;

} // namespace ode::crafted
