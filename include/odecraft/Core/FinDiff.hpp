#ifndef ODECRAFT_FINDIFF_HPP
#define ODECRAFT_FINDIFF_HPP

#include <odecraft/Tools.hpp>
#include <limits>

namespace ode{

/**
 * @brief Approximates the Jacobian matrix using central finite differences.
 *  Performs (2 x n) function evaluations in the process, where n is the size of the state vector.
 * 
 * @tparam T The type of the elements.
 * @tparam Callable The type of the callable object representing the function.
 * @param f The callable object representing the function.
 * @param out The output (F-storage) array for the Jacobian matrix, of size (n x n)
 * @param worker A temporary array for intermediate computations, of size (3 x n)
 * @param t The current time.
 * @param q The current state vector (size n).
 * @param dt The time step or perturbation vector (size n) or nullptr.
 * @param threshold A maximum threshold for determining the step size when dt is nullptr.
 * @param n The size of the state vector (size of the ode system)
 * @note Ideally, we want the threshold parameter to be an array of values, defining a scale value for each
 * component of the scale vector. A component-wise tolerance has not been implemented yet in this library for simplicity.
 */
template<typename T, typename Callable>
constexpr void jac_approx(Callable&& func, T* out, T* worker, const T& t, const T* q, const T* dt, const T& threshold, size_t n){

    const T EPS_SQRT = sqrt(std::numeric_limits<T>::epsilon());

    T* x = worker;
    T* y1 = worker + n;
    T* y2 = worker + 2*n;

    std::copy(q, q + n, x);

    for (size_t i = 0; i < n; i++) {
        // Compute step size: use provided dt or compute
        const T abs_qi = abs<T>(q[i]);
        T h_i;
        if (dt != nullptr) {
            h_i = dt[i];
        } else {
            h_i = EPS_SQRT * max_ref(threshold, abs_qi);
        }

        x[i] = q[i] - h_i;
        func(y1, t, x);

        x[i] = q[i] + h_i;
        func(y2, t, x);

        x[i] = q[i]; // restore it for the next iteration

        // Compute Jacobian column using central differences
        T* col = out + i * n;
        T two_h = 2 * h_i;
        for (size_t j = 0; j < n; j++) {
            col[j] = (y2[j] - y1[j]) / two_h;
        }
    }
}

}; // namespace ode


#endif // ODECRAFT_FINDIFF_HPP