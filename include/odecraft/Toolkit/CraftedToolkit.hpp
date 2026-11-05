#pragma once

/**
 * @file Toolkit/CraftedToolkit.hpp
 * @brief Shared vocabulary of the pre-compiled `ode::crafted` interface.
 *
 * Everything in `ode::crafted` is the header-only `ode` interface with every template
 * parameter but the scalar type `T` pinned down: the state has a dynamic shape and dims
 * (contracts::common::SD<>) and every callable is type-erased into a std::function (rhs_t,
 * objfun_t, observer_t, ...).
 * That leaves `T` as the only parameter left to instantiate, and the library ships those
 * instantiations for specific scalar types, so a translation unit that only uses
 * `ode::crafted` never has to compile a solver itself.
 *
 * The price is the usual one for type erasure: no compile-time state shape, no autodiff
 * Jacobians (a std::function cannot be called on dual numbers), and an indirect call per
 * Rhs evaluation. Code that wants any of those keeps using the header-only `ode`
 * interface via <odecraft/odecraft.hpp>, which is unaffected by this header.
 */

#include "Tools.hpp"
#include <odecraft/Contracts/DenseDescriptor.hpp>
#include <odecraft/Chaos/VariationalContract.hpp>
#include <lazex/apps/lazex_mpreal.hpp>



#define ODECRAFT_FOR_EACH_SCALAR(MACRO, ...)                    \
    MACRO(float __VA_OPT__(,) __VA_ARGS__)                      \
    MACRO(double __VA_OPT__(,) __VA_ARGS__)                     \
    MACRO(long double __VA_OPT__(,) __VA_ARGS__)                \
    MACRO(::ode::crafted::mpreal_t __VA_OPT__(,) __VA_ARGS__)


namespace ode::crafted{

#ifdef ODECRAFT_USE_LAZY_MPREAL
using mpreal_t = lazex::LazyType<mpfr::mpreal>;
#else
using mpreal_t = mpfr::mpreal;
#endif


/**
 * @brief Set the working precision, in bits, for every subsequent mpreal_t computation.
 * @param prec Precision in bits (e.g. 256 for roughly quad-and-a-half precision)
 */
void set_mpreal_prec(mpfr_prec_t prec);


/**
 * @brief The precision, in bits, that new mpreal_t values are created with.
 */
mpfr_prec_t get_default_prec();

// ---------------------------------------------------------------------------
// The states every compiled template is fixed to.
// ---------------------------------------------------------------------------

/// @brief The State Descriptor of the compiled interface: dynamic shape and dims.
using SD = ::ode::contracts::common::SD<>;

/// @brief The State Descriptor of the variational solvers built on SD.
using VarSD = ::ode::chaos::contracts::VarSD<SD>;

/// @brief A state: contiguous, and both its shape and its number of dims are runtime values.
template<typename T>
using State = ::ode::state_t<T, SD>;

/// @brief The state of a variational solver: `.main` and `.variation`, both a State<T>.
template<typename T>
using VarState = ::ode::state_t<T, VarSD>;

/// @brief The Jacobian: J(i, j) = df_i/dq_j, with i and j running over the flattened state.
template<typename T>
using JacMat = ::ode::jacmat_t<T, SD>;


// ---------------------------------------------------------------------------
// Type-erased callables. These are the signatures every compiled template is fixed to.
// S is the state they act on: State<T>, or VarState<T> around a variational solver.
// ---------------------------------------------------------------------------

template<typename T, typename S = State<T>>
using rhs_t = ::ode::rhs_t<T, S>;           // void(S& out, T t, const S& q)

template<typename T>
using jac_t = std::function<void(JacMat<T>& out, T t, const State<T>& q)>;

template<typename T, typename S = State<T>>
using objfun_t = ::ode::objfun_t<T, S>;     // T   (const T& t, const S& q)

template<typename T, typename S = State<T>>
using interp_t = ::ode::interp_t<T, S>;     // void(S& out, const T& t)

template<typename T, typename S = State<T>>
using observer_t = ::ode::observer_t<T, S>; // bool(const T& t, const S& q, const T* t_ptr)


/**
 * @brief The one ODE system type the compiled interface is built around.
 *
 * Holds an Rhs and an optional Jacobian, both type-erased. Leaving `.Jac` null is fine:
 * that selects JacPolicy::Nullable, so the implicit steppers fall back to a finite
 * difference Jacobian at runtime. Provide one when you have it -- BDF in particular is
 * considerably happier with an exact Jacobian.
 *
 * @note The Jacobian is written as `out(i, j) = df_i/dq_j`.
 *       See ode::OdeData for the full explanation.
 */
template<typename T>
using ode_t = ::ode::OdeData<rhs_t<T>, jac_t<T>>;


// ---------------------------------------------------------------------------
// Plain vocabulary types, re-exported unchanged.
// ---------------------------------------------------------------------------

using ::ode::OdeData,
      ::ode::StepResult,
      ::ode::JacPolicy,
      ::ode::RootPolicy;

using ::ndspan::Array,
      ::ndspan::Array1D,
      ::ndspan::Array2D,
      ::ndspan::View,
      ::ndspan::View1D,
      ::ndspan::View2D,
      ::ndspan::Allocation,
      ::ndspan::Layout;

using ::ode::Vector,
      ::ode::make_vector;

} // namespace ode::crafted
