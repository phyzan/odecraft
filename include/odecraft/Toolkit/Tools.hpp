#pragma once


#include <chrono>
#include <functional>
#include <cmath>
#include <type_traits>
#include <utility>
#include <xdiff/xdiff.hpp>
#include <polybox/polybox.hpp>
#include <ndspan/ndspan.hpp>
#include <lazex/lazex.hpp>


#define ODECRAFT_LAMBDA(out, t, q) [=](auto& out, const auto& t, const auto& q) -> void 

#define ODECRAFT_USING_MATH XDIFF_USING_MATH

namespace ode {

// Standard math overloads, so that an unqualified abs()/sin()/pow()/... on a bare scalar
// resolves to std's while ADL still carries xdiff Duals and lazex expressions to their own.
// Without it, a scalar call reaches whatever <cstdlib> left in the global namespace, where
// ::abs(int) silently truncates. Callers need the same line in their own namespace -
// see odecraft/math.hpp.
ODECRAFT_USING_MATH;

template<typename cls, typename derived>
using GetDerived = std::conditional_t<(std::is_same_v<derived, void>), cls, derived>;

// USEFUL ALIASES

using ndspan::Array1D, ndspan::CArray2D, ndspan::FArray2D;

template<typename F>
using cache_t = std::conditional_t<
    std::is_trivially_copyable_v<F>,
    const F,
    const F&>;

template<typename T, typename SD>
using value_t = SD::template scalar<T>;

template<typename T, typename SD, int Order>
using dual_t = xdiff::Dual<value_t<T, SD>, SD, Order>;

template<typename T, typename SD, int Order>
using seed_t = xdiff::Seed<value_t<T, SD>, SD, Order>;

template<typename T, typename SD, int Order>
using seed_state_t = xdiff::SeedState<value_t<T, SD>, SD, Order>;

template<typename T, typename SD, int Order>
using dual_state_t = xdiff::dual_state_t<value_t<T, SD>, SD, Order>;

template<typename T, typename SD>
using real_state_t = SD::template state_type<T>;

template<typename T, typename SD>
using state_t = SD::template state_type<value_t<T, SD>>;

template<typename T, typename SD>
using jacmat_t = SD::template jacobian_matrix<value_t<T, SD>>;

template<typename T, typename SD, typename JD, int Order>
using dual_jacmat_t = SD::template jacobian_matrix<dual_t<T, SD, Order>>;


namespace detail{

// Whether user callbacks (Rhs, Jac, event functions, masks) may throw: define ODECRAFT_RHS_THROWS
// to allow it. Otherwise the solvers' Rhs wrappers and their steps are noexcept, so a callback
// that throws terminates the program, and the compiler can drop the unwinding paths around every
// call into it.
#ifdef ODECRAFT_RHS_THROWS
inline constexpr bool rhs_may_throw = true;
#else
inline constexpr bool rhs_may_throw = false;
#endif


template<typename F, typename T, typename State>
concept supportsRhs =
    requires(F f, State& out, T t, const State& q) {
        { f.Rhs(out, t, q) } -> std::same_as<void>;
    };

template<typename F, int Order, typename T, typename SD>
concept supportsDualRhsAt =
    requires(F f, dual_state_t<T, SD, Order>& out, T t, const seed_state_t<T, SD, Order>& q) {
        { f.Rhs(out, t, q) } -> std::same_as<void>;
    };

template<typename F, int Order, typename T, typename SD, typename JD>
concept supportsDualJacAt =
    requires(F f, dual_jacmat_t<T, SD, JD, Order>& out, T t, const seed_state_t<T, SD, Order>& q) {
        { f.Jac(out, t, q) } -> std::same_as<void>;
    };

template<typename F, typename T, typename SD, typename Seq>
inline constexpr bool supportsDualRhsAll = false;

template<typename F, typename T, typename SD, typename JD, typename Seq>
inline constexpr bool supportsDualJacAll = false;

template<typename F, typename T, typename SD, int... Is>
inline constexpr bool supportsDualRhsAll<F, T, SD, std::integer_sequence<int, Is...>> =
    (supportsDualRhsAt<F, Is+1, T, SD> && ...);

template<typename F, typename T, typename SD, typename JD, int... Is>
inline constexpr bool supportsDualJacAll<F, T, SD, JD, std::integer_sequence<int, Is...>> =
    (supportsDualJacAt<F, Is+1, T, SD, JD> && ...);
} // namespace ode::detail


template<typename F, typename T, typename State>
concept isRhsFunc = 
requires(F f, State& out, T t, const State& q){
    { f(out, t, q) } -> std::same_as<void>;
};

// Check if F has a callable Rhs (static or non-static, non-template)
template<typename F, typename T, typename State>
concept hasRhsFunc =
    requires(F f, State& out, T t, const State& q) {
        { f.Rhs(out, t, q) } -> std::same_as<void>;
    };

// Check if F has a callable Jac (static or non-static, non-template)
template<typename F, typename T, typename State, typename JM>
concept hasJacFunc =
    requires(F f, JM& out , T t, const State& q) {
        { f.Jac(out, t, q) } -> std::same_as<void>;
    };

template<typename F>
concept isNullable = 
    requires(F f) {
        f = nullptr;
        { f == nullptr } -> std::convertible_to<bool>;
    };

template<typename F>
concept hasNullableJac =
    requires(F f) {
        f.Jac = nullptr;
        { f.Jac == nullptr } -> std::convertible_to<bool>;
    };

template<typename F, typename T, typename State>
concept isObjFun =
requires(F f, T t, const State& q){
    { f(t, q) } -> std::convertible_to<T>;
};

template<typename F, typename T, typename State>
concept isObserver =
requires(F f, T t, const State& q, const T* t_ptr){
    { f(t, q, t_ptr) } -> std::convertible_to<bool>;
};

template<typename F, typename T>
concept isArray =
requires(const F& f, size_t i){
    { f.size() } -> std::same_as<size_t>;
    { f[i] } -> std::same_as<const T&>;
};


// Satisfied when f.Rhs(out, t, q) allows for `out` and `q`
// to be Duals of order 1 <= order <= MaxOrder
template<typename F, int MaxOrder, typename T, typename SD>
concept supportsDualRhs =
    MaxOrder > 0 &&
    detail::supportsDualRhsAll<F, T, SD, std::make_integer_sequence<int, MaxOrder>>;

// Satisfied when f.Jac(out, t, q) allows for `out` and `q`
// to be Duals of order 1 <= order <= MaxOrder
template<typename F, int MaxOrder, typename T, typename SD, typename JD>
concept supportsDualJac =
    MaxOrder > 0 &&
    detail::supportsDualJacAll<F, T, SD, JD, std::make_integer_sequence<int, MaxOrder>>;


template<typename F, typename T, typename State>
concept isStateInterp = requires(F f, State& out, T t){
    { f(out, t) } -> std::same_as<void>;
};

// f(time, State) -> T
template<typename T, typename State>
using objfun_t = std::function<T(const T&, const State&)>;

// f(State& out, const T& t, const State& q) -> void
template<typename T, typename State>
using rhs_t = std::function<void(State& out, T t, const State& q)>;

// f(State& out, const T& t)
template<typename T, typename State>
using interp_t = std::function<void(State&, const T&)>;

// f(time, state, optional_address)
template<typename T, typename State>
using observer_t = std::function<bool(const T&, const State&, const T*)>;


template<typename RHS, typename JAC = std::nullptr_t>
struct OdeData {

    using RhsType = RHS;
    using JacType = JAC;

    RHS Rhs;
    JAC Jac = nullptr;

    /*
    IMPORTANT
    ================

    The jacobian function takes as first input the output matrix (just like the rhs function takes as input the output array).
    However, the output matrix stores its data in a column-major order, but is passed as the flattened array.
    Since the output matrix must behave as J(i, j) = df_i / dx_j, the output array must be accessed
    as m[j + i*n] = df_i/dx_j, and not m[i + j*n].

    In other words, if the jacobian matrix is

    J_ij = [[a  b],
            [c  d]]
    
    Then the output array must be set as

    m[0] = a, m[1] = c, m[2] = b, m[3] = d
    and NOT
    m[0] = a, m[1] = b, m[2] = c, m[3] = d;
    */
};

// Clang fails to synthesize the implicit aggregate deduction guide for
// OdeData{.Rhs=...} when the initialization occurs inside a template (even if
// unrelated to the enclosing template parameters); GCC handles it fine. These
// explicit guides make designated-initializer CTAD work portably everywhere,
// for both the Rhs-only and the Rhs+Jac aggregate-init forms.
template<typename RHS>
OdeData(RHS) -> OdeData<RHS>;

template<typename RHS, typename JAC>
OdeData(RHS, JAC) -> OdeData<RHS, JAC>;


enum class JacPolicy : std::uint8_t{
    Autodiff,
    Exact,
    Nullable,
    Approx,
};


template<typename T, typename SD, typename F>
constexpr JacPolicy getJacPolicy(){
    static_assert(hasRhsFunc<F, T, state_t<T, SD>>, "F::Rhs has invalid signature");
    using JM = SD::template jacobian_matrix<value_t<T, SD>>;
    if constexpr (hasJacFunc<F, T, state_t<T, SD>, JM>){
        // An explicitly provided Jacobian should
        // always be utilized, even when the Rhs supports Duals
        // If the explicitly provided Jacobian is nullable,
        // and at runtime it is indeed null, then Jac computation
        // falls backs to numerical approximation for now.
        // So, either provide a non-Nullable Jac, otherwise
        // an Rhs that supports Duals (or both)
        // But it's better *NOT* to provide a nullable Jac
        // that is indeed null alongside an Rhs that supports Duals,
        // because the autodiff capability will not be utilized
        // and that might raise confusion as to the loss of accuracy
        if constexpr (hasNullableJac<F>){
            return JacPolicy::Nullable;
        } else {
            return JacPolicy::Exact;
        }
    } else if (supportsDualRhs<F, 1, T, SD>) {
        return JacPolicy::Autodiff;
    } else {
        return JacPolicy::Approx;
    }
}


enum class StepResult : std::uint8_t {
    Success, // Successful step
    NonFiniteError, // Non-finite value encountered (NaN or Inf)
    TinyStepError, // Step size became too small (below machine epsilon)
    MinStepError, // Step size reached minimum set by user
    MaxStepError, // Step size reached maximum set by user
    SingularMatrixError, // The linear system of an implicit step is singular
};


template<typename Callable, typename... F>
class LightFunctor{

    // The data is referenced, so it must outlive the functor. A temporary cannot, so it is copied.
    template<typename U>
    using held_t = std::conditional_t<std::is_lvalue_reference_v<U>, const std::remove_reference_t<U>&, std::remove_cvref_t<U>>;

public:
    LightFunctor(Callable func, F&&... data) : f_(std::move(func)), data_{std::forward<F>(data)...} {}

    template<typename... Args>
    decltype(auto) operator()(Args&&... args) const {
        return this->apply_impl(std::make_index_sequence<sizeof...(F)>(), std::forward<Args>(args)...);
    }

    auto detach() const {
        return this->detach_impl(std::make_index_sequence<sizeof...(F)>());
    }

private:

    template<size_t... I, typename... Args>
    decltype(auto) apply_impl(std::index_sequence<I...> /**/, Args&&... args) const {
        return f_(std::forward<Args>(args)..., std::get<I>(data_)...);
    }

    template<size_t... I>
    auto detach_impl(std::index_sequence<I...> /**/) const {
        return [func=this->f_, ...data = std::remove_cvref_t<F>(std::get<I>(data_))]<typename... Args>(Args&&... args){
            return func(std::forward<Args>(args)..., data...);
        };
    }

    Callable f_;
    std::tuple<held_t<F>...> data_;
};

template<typename Callable, typename... F>
LightFunctor<Callable, F...> make_light_functor(Callable func, F&&... data){
    return LightFunctor<Callable, F...>(std::move(func), std::forward<F>(data)...);
}


template <typename T>
inline T inf() {
    // When using -ffast-math, infinity() may cause issues or segfaults
    // Use a very large finite number instead that's safe with -ffast-math
    #ifdef __FAST_MATH__
    return std::numeric_limits<T>::max();
    #else
    return std::numeric_limits<T>::infinity();
    #endif
}

template<typename T>
bool resize_step(T& factor, T& habs, const T& min_step, const T& max_step){
    bool res = false;
    if (habs*factor < min_step){
        factor = min_step/habs;
        habs = min_step;
    } else if (habs*factor > max_step){
        factor = max_step/habs;
        habs = max_step;
    } else{
        habs *= factor;
        res = true;
    }
    return res;
}

template <typename T>
inline bool isfinite(const T& value) {
#ifndef ODECRAFT_NO_NAN_CHECK
    if constexpr (!std::is_integral_v<T>) {
        #ifdef __FAST_MATH__
        // When -ffast-math is enabled, std::isfinite may not work correctly
        // Use range check instead: value is finite if it's within representable range
        return (value >= std::numeric_limits<T>::lowest() &&
                value <= std::numeric_limits<T>::max());
        #else
        return std::isfinite(value);
        #endif
    } else {
        return true; // Integral types are always finite
    }
#else
    return true; // If ODECRAFT_NO_NAN_CHECK is defined, assume all values are finite
#endif
}

template<typename T, typename SD>
bool all_are_finite(const state_t<T, SD>& state){
#ifndef ODECRAFT_NO_NAN_CHECK
    // TODO optimize similar to allEqual
    bool res = true;
    SD::for_each([&res](const value_t<T, SD>& x){
        res = res ? isfinite(x) : false;
    }, state);
    return res;
#else
    return true;
#endif
}

template<typename T, typename SD>
bool allEqual(const state_t<T, SD>& a, const state_t<T, SD>& b){
    // TODO optimize so that SD::for_each does not unnecessarily call all
    // items when the first `false` is encountered.
    bool res;
    SD::for_each([&res](const value_t<T, SD>& x, const value_t<T, SD>& y){
        res = res ? x == y : false;
    }, a, b);
    return res;
}

template<typename T, typename SD>
T max_elem(const real_state_t<T, SD>& err){
    T res{0};
    SD::for_each([&res](const T& xi){
        if (xi > res){
            res = xi;
        }
    }, err);
    return res;
}


template<typename... Args>
std::string GetStr(Args&&... args) {
    std::ostringstream out;
    (out << ... << std::forward<Args>(args));
    return out.str();
}


template<typename SD, typename U>
void copy_state(typename SD::template state_type<U>& dst, const typename SD::template state_type<U>& src){
    SD::for_each([](U& xi, const U& si){
        xi = si;
    }, dst, src);
}

template<typename F>
double timeit_ms(F&& action) {
    const auto t_start = std::chrono::steady_clock::now();

    std::forward<F>(action)();

    const auto t_end = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::milli>(
        t_end - t_start
    ).count();
}

template<typename T>
inline int sgn(const T& x){
    return ( x > 0) ? 1 : ( (x < 0) ? -1 : 0);
}

template<typename T>
inline int sgn(const T& t1, const T& t2){
    //same as sgn(t2-t1), but avoids roundoff error
    return (t1 < t2 ? 1 : (t1 > t2 ? -1 : 0));
}

template<typename T, typename A, typename B>
NDSPAN_INLINE void set_min(T& out, const A& a, const B& b){
    (a < b) ? (out = a) : (out = b);
}

template<typename T, typename A, typename B>
NDSPAN_INLINE void set_max(T& out, const A& a, const B& b){
    (a > b) ? (out = a) : (out = b);
}

template<typename T>
NDSPAN_INLINE const T& max_ref(const T& a, const T& b){
    return (a > b) ? a : b;
}

template<typename T>
NDSPAN_INLINE const T& min_ref(const T& a, const T& b){
    return (a < b) ? a : b;
}

} // namespace ode
