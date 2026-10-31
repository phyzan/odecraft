#pragma once


#include <cstdint>
#include <utility>


enum class RootPolicy : std::uint8_t { Left, Middle, Right};

// Returned after a bisection algorithm
enum class RootStatus : uint8_t {
    Success,      // Converged within the iteration budget
    MaxIter,      // Root detected within an interval but ran out of iterations
    NotBracketed, // No root found (for bisection algorithms)
    Failure,      // Did not find a root (general)
};

/**
 * @brief Find an equation root using the bisection method
 *
 * @tparam T : Scalar type
 * @tparam Callable f(x)
 * @param out On success, the converged right bracket of the solution
 * @param f_out On success, f(out)
 * @param f The function to find the root of
 * @param a lower root bound
 * @param b upper root bound
 * @param fa f(a)
 * @param fb f(b)
 * @param ftol convergence tolerance: Find x such that |f(x)| <= ftol
 * @return RootStatus::Success, or RootStatus::NotBracketed if f(a) * f(b) > 0
 */
template<typename T, typename Callable>
RootStatus bisect_root(T& out, T* f_out, Callable&& f, T a, T b, T fa, T fb, const T& ftol){
    // It is important that `out` is not affected unless
    // the loop converged to a solution.
    // f(a) and f(b) are passed in, so the endpoints are never re-evaluated.
    T m = a;
    T fm;

    if (fa * fb > 0){
        return RootStatus::NotBracketed;
    }
    
    while (abs(fb) > ftol){
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
            fb = fm;
        }
    }

    out = b;
    if (f_out != nullptr){
        *f_out = fb;
    }
    return RootStatus::Success;
}

/**
 * @brief Find an equation root using the bisection method.
 * Same as above, but evaluates f(a) and f(b) itself.
 */
template<typename T, typename Callable>
RootStatus bisect_root(T& out, T* f_out, Callable&& f, T a, T b, const T& ftol){
    T fa = f(a);
    T fb = f(b);
    return bisect_root(
        out,
        f_out,
        std::forward<Callable>(f),
        std::move(a), std::move(b),
        std::move(fa), std::move(fb),
        ftol
    );
}