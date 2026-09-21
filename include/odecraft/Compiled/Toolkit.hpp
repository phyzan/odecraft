#ifndef ODECRAFT_COMPILED_TOOLKIT_HPP
#define ODECRAFT_COMPILED_TOOLKIT_HPP

/**
 * @file Compiled/Toolkit.hpp
 * @brief Shared vocabulary of the pre-compiled `ode::crafted` interface.
 *
 * Everything in `ode::crafted` is the header-only `ode` interface with every template
 * parameter but the scalar type `T` pinned down: the system size is dynamic (N = 0) and
 * every callable is type-erased into a std::function (rhs_t, objfun_t, observer_t, ...).
 * That leaves `T` as the only parameter left to instantiate, and the library ships those
 * instantiations for specific scalar types, so a translation unit that only uses
 * `ode::crafted` never has to compile a solver itself.
 *
 * The price is the usual one for type erasure: no compile-time system size, no autodiff
 * Jacobians (a std::function cannot be called on dual numbers), and an indirect call per
 * Rhs evaluation. Code that wants any of those keeps using the header-only `ode`
 * interface via <odecraft/odecraft.hpp>, which is unaffected by this header.
 */

#include <odecraft/Toolkit/Tools.hpp>
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
// Type-erased callables. These are the signatures every compiled template is fixed to.
// ---------------------------------------------------------------------------

using ::ode::rhs_t,         // void(T* out, const T& t, const T* q)
      ::ode::objfun_t,      // T   (const T& t, const T* q)
      ::ode::interp_t,      // void(T* out, const T& t)
      ::ode::observer_t;    // bool(const T& t, const T* q, const T* t_ptr)


/**
 * @brief The one ODE system type the compiled interface is built around.
 *
 * Holds an Rhs and an optional Jacobian, both type-erased. Leaving `.Jac` null is fine:
 * that selects JacPolicy::Nullable, so the implicit steppers fall back to a finite
 * difference Jacobian at runtime. Provide one when you have it -- BDF in particular is
 * considerably happier with an exact Jacobian.
 *
 * @note The Jacobian is expected in column-major order, i.e. `out[j + i*n] = df_i/dq_j`.
 *       See ode::OdeData for the full explanation.
 */
template<typename T>
using ode_t = ::ode::OdeData<rhs_t<T>, rhs_t<T>>;


// ---------------------------------------------------------------------------
// Plain vocabulary types, re-exported unchanged.
// ---------------------------------------------------------------------------

using ::ode::State,
      ::ode::OdeData,
      ::ode::StepResult,
      ::ode::JacPolicy,
      ::ode::RootPolicy,
      ::ode::Clock;

using ::ode::Array,
      ::ode::Array1D,
      ::ode::Array2D,
      ::ode::View,
      ::ode::MutView,
      ::ode::View1D,
      ::ode::View2D,
      ::ode::Allocation,
      ::ode::Layout;

using ::ode::Vector,
      ::ode::make_vector;

} // namespace ode::crafted

#endif // ODECRAFT_COMPILED_TOOLKIT_HPP
