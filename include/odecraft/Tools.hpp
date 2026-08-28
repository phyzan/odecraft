#ifndef ODECRAFT_TOOLS_HPP
#define ODECRAFT_TOOLS_HPP


#include <complex>
#include <chrono>
#include <functional>
#include <omp.h>
#include <cmath>
#include <sstream>
#include <xdiff/xdiff.hpp>
#include <polybox/polybox.hpp>
#include <xdiff/tools.hpp>
#include <ndspan/ndspan.hpp>


#define ODE_LAMBDA(out, t, q) [=](auto* out, const auto& t, const auto* q) -> void 

namespace ode {

using std::pow, std::sin, std::cos, std::exp, std::real, std::imag, ndspan::min, ndspan::max, std::complex;

using ndspan::Array, ndspan::Array1D, ndspan::Array2D, ndspan::View, ndspan::MutView, ndspan::View1D, ndspan::View2D, ndspan::View3D, ndspan::Allocation, ndspan::Layout, ndspan::prod;

using xdiff::Vector, xdiff::make_vector;

template<typename cls, typename derived>
using GetDerived = std::conditional_t<(std::is_same_v<derived, void>), cls, derived>;

// USEFUL ALIASES

// The layout is the only part of DualType that varies with N, and it does not depend
// on Order. Keeping it in its own constant lets DualType stay a direct alias to a class
// template specialisation, which leaves `Order` in a *deducible* position.
//
// Spelling DualType as std::conditional_t<...> instead would expand it to
// `typename std::conditional<...>::type`, a dependent qualified name and therefore a
// non-deduced context: no function template taking a DualType<T, N, Order>* parameter
// could then deduce Order from its argument, and every such overload would silently
// drop out of overload resolution (and out of the supportsDualRhs/Jac concepts).
template<size_t N>
inline constexpr xdiff::Layout dual_layout = (N > 0) ? xdiff::Layout::Flat : xdiff::Layout::Nested;

template<typename T, size_t N, size_t Order>
using DualType = xdiff::Dual<T, N, Order, dual_layout<N>>;


template<typename T, size_t N>
using JacMat = Array2D<T, N, N, Allocation::Auto, Layout::F>;


using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

namespace detail {

template<typename F, typename T, size_t N, size_t Order>
concept supportsDualRhsAt =
    requires(F f, DualType<T, N, Order>* out, T t, const DualType<T, N, Order>* q) {
        { f.Rhs(out, t, q) } -> std::same_as<void>;
    };

template<typename F, typename T, size_t N, size_t Order>
concept supportsDualJacAt =
    requires(F f, DualType<T, N, Order>* out, T t, const DualType<T, N, Order>* q) {
        { f.Jac(out, t, q) } -> std::same_as<void>;
    };

template<typename F, typename T, size_t N, typename Seq>
inline constexpr bool supportsDualRhsAll = false;

template<typename F, typename T, size_t N, typename Seq>
inline constexpr bool supportsDualJacAll = false;

template<typename F, typename T, size_t N, size_t... Is>
inline constexpr bool supportsDualRhsAll<F, T, N, std::index_sequence<Is...>> =
    (supportsDualRhsAt<F, T, N, Is + 1> && ...);

template<typename F, typename T, size_t N, size_t... Is>
inline constexpr bool supportsDualJacAll<F, T, N, std::index_sequence<Is...>> =
    (supportsDualJacAt<F, T, N, Is + 1> && ...);

} // namespace detail

template<typename F, typename T>
concept isRhsFunc = 
requires(F f, T* out, T t, const T* q){
    { f(out, t, q) } -> std::same_as<void>;
};

// Check if F has a callable Rhs (static or non-static, non-template)
template<typename F, typename T>
concept hasRhsFunc =
    requires(F f, T* out, T t, const T* q) {
        { f.Rhs(out, t, q) } -> std::same_as<void>;
    };

// Check if F has a callable Jac (static or non-static, non-template)
template<typename F, typename T>
concept hasJacFunc =
    requires(F f, T* out, T t, const T* q) {
        { f.Jac(out, t, q) } -> std::same_as<void>;
    };

template<typename F>
concept hasNullableJac =
    requires(F f) {
        f.Jac = nullptr;
        { f.Jac == nullptr } -> std::convertible_to<bool>;
    };

template<typename F, typename T>
concept hasRhsOnly = hasRhsFunc<F, T> && !hasJacFunc<F, T>;

template<typename F, typename T>
concept hasRhsAndJac = hasRhsFunc<F, T> && hasJacFunc<F, T>;

template<typename F, typename T>
concept isObjFun =
requires(F f, T t, const T* q){
    { f(t, q) } -> std::convertible_to<T>;
};

template<typename F, typename T>
concept isObserver =
requires(F f, T t, const T* q, const T* t_ptr){
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
template<typename F, typename T, size_t N, size_t MaxOrder>
concept supportsDualRhs =
    MaxOrder > 0 &&
    detail::supportsDualRhsAll<F, T, N, std::make_index_sequence<MaxOrder>>;

// Satisfied when f.Jac(out, t, q) allows for `out` and `q`
// to be Duals of order 1 <= order <= MaxOrder
template<typename F, typename T, size_t N, size_t MaxOrder>
concept supportsDualJac =
    MaxOrder > 0 &&
    detail::supportsDualJacAll<F, T, N, std::make_index_sequence<MaxOrder>>;


template<typename F, typename T>
concept isStateInterp = requires(F f, T* out, T t){
    { f(out, t) } -> std::same_as<void>;
};

// f(T t, const T* q) -> T
template<typename T>
using objfun_t = std::function<T(const T&, const T*)>;

// f(T* out, const T& t, const T* q) -> void
template<typename T>
using rhs_t = std::function<void(T*, const T&, const T*)>;

// f(time, state_vector, optional_address)
template<typename T>
using observer_t = std::function<bool(const T&, const T*, const T*)>;

enum class RootPolicy : std::uint8_t { Left, Middle, Right};


template<typename T>
inline T abs(const T& x){
    return x >= 0 ? x : T{-x};
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

template<typename T, typename U>
NDSPAN_INLINE void set_abs(T& out, const U& x){
    (x > 0) ? (out = x) : (out = -x);
}

template<typename T>
NDSPAN_INLINE const T& max_ref(const T& a, const T& b){
    return (a > b) ? a : b;
}

template<typename T>
NDSPAN_INLINE const T& min_ref(const T& a, const T& b){
    return (a < b) ? a : b;
}

template<typename T>
bool allEqual(const T* a, const T* b, size_t n){
    for (size_t i=0; i<n; i++){
        if (a[i] != b[i]){
            return false;
        }
    }
    return true;
}

template<typename T, RootPolicy RP, typename Callable>
T bisect(Callable&& f, T a, T b, const T& ftol){
    T err = 2*ftol+1;
    T m = a;
    T fa = f(a);
    T fm;

    assert((fa * f(b) <= 0) && "Root not bracketed" );
    
    while (err > ftol){
        m = (a + b) / 2;
        if (m == a || m == b){
            // reached machine precision limit, return the best guess so far
            break;
        } else {
            fm = f(m);
        }

        if (fa * fm  > 0){
            a = m;
            fa = fm;
        } else {
            b = m;
        }
        set_abs(err, fm);
    }

    if constexpr (RP == RootPolicy::Left) {
        return a;
    } else if constexpr (RP == RootPolicy::Middle) {
        return m;
    } else {
        return b;
    }
}

template<typename T>
void inv_mat_row_major(T* out, const T* mat, size_t N, T* work, size_t* pivot) {
    // out: size N*N, row-major order
    // mat: size N*N, row-major order
    // work: size N
    // pivot: size N
    if (N == 0){
        return;
    }

    std::vector<T> lu(N * N);
    for (size_t i = 0; i < N * N; ++i){
        lu[i] = mat[i];
    }

    for (size_t i = 0; i < N; ++i){
        pivot[i] = i;
    }

    for (size_t i = 0; i < N; ++i) {
        size_t max_row = i;
        T max_val = abs<T>(lu[i*N + i]);
        for (size_t j = i + 1; j < N; ++j) {
            T val = abs<T>(lu[j*N + i]);
            if (val > max_val) { max_val = val; max_row = j; }
        }
        // assert(max_val != 0 && "Matrix is singular");

        if (max_row != i) {
            for (size_t k = 0; k < N; ++k){
                std::swap(lu[i*N + k], lu[max_row*N + k]);
            }
            std::swap(pivot[i], pivot[max_row]);
        }

        for (size_t j = i + 1; j < N; ++j) {
            lu[j*N + i] /= lu[i*N + i];
            for (size_t k = i + 1; k < N; ++k){
                lu[j*N + k] -= lu[j*N + i] * lu[i*N + k];
            }
        }
    }

    for (size_t col = 0; col < N; ++col) {
        for (size_t i = 0; i < N; ++i){
            work[i] = (pivot[i] == col) ? 1 : 0;
        }

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < i; ++j){
                work[i] -= lu[i*N + j] * work[j];
            }
        }

        for (size_t i = N; i-- > 0;) {
            for (size_t j = i + 1; j < N; ++j){
                work[i] -= lu[i*N + j] * work[j];
            }
            work[i] /= lu[i*N + i];
        }

        for (size_t i = 0; i < N; ++i){
            out[i*N + col] = work[i];
        }

    }
}

template<typename T>
T choose_step(const T& habs, const T& hmin, const T& hmax) {
    return std::max<T>(std::min<T>(habs, hmax), hmin);
}

template<typename T>
T detLU_row_major(T* mat, size_t N) {
    if (N == 0) {
        return 1;
    }

    T det = 1;
    int sign = 1;

    for (size_t i = 0; i < N; ++i) {
        // Partial pivoting
        size_t pivot = i;
        T max_val = abs<T>(mat[i * N + i]);
        for (size_t j = i + 1; j < N; ++j) {
            T val = abs<T>(mat[j * N + i]);
            if (val > max_val) {
                pivot = j;
                max_val = val;
            }
        }

        if (max_val == T(0)) {
            return T(0);
        }

        if (pivot != i) {
            // Swap rows
            for (size_t k = 0; k < N; ++k) {
                T tmp = mat[i * N + k];
                mat[i * N + k] = mat[pivot * N + k];
                mat[pivot * N + k] = tmp;
            }
            sign = -sign;
        }

        det *= mat[i * N + i];

        // Eliminate below pivot
        for (size_t j = i + 1; j < N; ++j) {
            T factor = mat[j * N + i] / mat[i * N + i];
            for (size_t k = i; k < N; ++k) {
                mat[j * N + k] = mat[j * N + k] - factor * mat[i * N + k];
            }
        }
    }

    return det*sign;
}


template<typename T, size_t N>
struct SolverState{
    
    ndspan::Array1D<T, N> vector;
    std::string msg;
    size_t nt;
    T time;
    T stepsize;
    bool diverging;
    bool running;

    SolverState(const T* q, T t, T habs, size_t nsys, bool diverges, bool is_running, size_t updates, std::string message) : vector(q, nsys), msg(std::move(message)), nt(updates), time(t), stepsize(habs), diverging(diverges), running(is_running){}

    void show(int precision = 15) const {
        std::cout << "\n" << std::setprecision(precision) << 
        "OdeSolver current state:\n---------------------------\n"
        "\ttime       : " << time << "\n" <<
        "\tq          : ";
        
        array_repr(std::cout, vector);
        std::cout << "\n" <<
        "\tstepsize   : " << stepsize << "\n" <<
        "\tDiverges   : " << (diverging ? "true" : "false") << "\n" << 
        "\tRunning    : " << (running ? "true" : "false") << "\n" <<
        "\tUpdates    : " << nt << "\n" <<
        "\tState      : " << msg << std::endl;
    }

};


template<typename T, size_t N>
struct SolverRichState : public SolverState<T, N>{

    std::string event_name;

    SolverRichState(const T* q, T t, T habs, size_t Nsys, bool diverges, bool is_running, size_t Nt, std::string message, std::string event) : SolverState<T, N>(q, t, habs, Nsys, diverges, is_running, Nt, std::move(message)), event_name(std::move(event)) {}
    
    void show(int precision = 15) const {
        SolverState<T, N>::show(precision);
        std::cout << "\tEvent     : " << event_name << "\n" << std::endl;
    }

};


template<typename T>
class State{

    //provides a view of data, does not own it. The lifetime of a State object must be shorter
    //than that of the underlying data, otherwise the program will crash or behave incorrectly

public:

    State(const T* data, size_t Nsys) : _data(data), _nsys(Nsys) {}

    const T& t() const {return _data[0];}

    const T& habs() const {return _data[1];}

    const T* vector() const {return _data + 2;}

    size_t nsys() const {return _nsys;}

protected:

    const T* _data;
    size_t _nsys;
};


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
    Since the output matrix, must behave as J(i, j) = df_i / dx_j, the output array must be accessed
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
    Approx
};


template<typename T, size_t N, hasRhsFunc<T> F>
constexpr JacPolicy getJacPolicy(){
    if constexpr (hasJacFunc<F, T>){
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
    } else if (supportsDualRhs<F, T, N, 1>) {
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
};


class Clock{

public:

    Clock() = default;

    inline static TimePoint now(){
        return std::chrono::high_resolution_clock::now();
    }

    inline static double as_duration(const TimePoint& t1, const TimePoint& t2){
        std::chrono::duration<double> duration = t2-t1;
        return duration.count();
    }


    inline static std::string format_duration(double t){
        int h = int(t/3600);
        int m = int((t - h*3600)/60);
        int s = int(t - h*3600 - m*60);

        return std::to_string(h) + " h, " + std::to_string(m) + " m, " + std::to_string(s) + " s";  
    }

    inline void start(){
        _start = now();
    }

    inline double seconds() const{
        return as_duration(_start, now());
    }

    inline std::string message() const{
        return format_duration(seconds());
    }

private:

    TimePoint _start;
};


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
T norm_squared(const T* x, size_t size){
    //optimize
    T res = 0;
    // #pragma omp simd reduction(+:res)
    for (size_t i=0; i<size; i++){
        res += x[i]*x[i];
    }
    return res;
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


template<typename T>
T rms_norm(const T* x, size_t size){
    return sqrt(norm_squared(x, size)/size);
}

template<typename T>
T rms_norm(const T* x, const T* scale, size_t size){
    T norm_sq = 0;
    // #pragma omp simd reduction(+:norm_sq)
    for (size_t i=0; i<size; i++){
        norm_sq += x[i]*x[i]/(scale[i]*scale[i]);
    }
    return sqrt(norm_sq/size);
}


template<typename T>
T norm(const T* x, size_t size){
    return sqrt(norm_squared(x, size));
}

template<typename T>
NDSPAN_INLINE bool all_are_finite(const T* data, size_t n){
#ifndef ODECRAFT_NO_NAN_CHECK
    for (size_t i=0; i<n; i++){
        if (!isfinite(data[i])){
            return false;
        }
    }
#endif
    return true;
}


template<typename... Args>
std::string GetStr(Args&&... args) {
    std::ostringstream out;
    (out << ... << std::forward<Args>(args));
    return out.str();
}



inline void show_progress(int n, int target, const Clock& clock){
    std::cout << "\033[2K\rProgress: " << std::setprecision(2) << n*100./target << "%" <<   " : " << n << "/" << target << "  Time elapsed : " << clock.message() << "      Estimated duration: " << Clock::format_duration(target*clock.seconds()/n) << std::flush;
}

template<typename... Arg>
NDSPAN_INLINE void print(Arg&&... x){
    ((std::cout << x << ' '), ...);
    std::cout << "\n";
}


} // namespace ode

#endif // ODECRAFT_TOOLS_HPP