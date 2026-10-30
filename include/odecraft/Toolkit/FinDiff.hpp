#ifndef ODECRAFT_FINDIFF_HPP
#define ODECRAFT_FINDIFF_HPP

#include "Tools.hpp"


namespace ode{

namespace detail{

/**
 * @brief Approximates the Jacobian matrix using central finite differences.
 *  Performs (2 x n) function evaluations in the process, where n is the size of the state vector.
 * 
 * @tparam T The type of the elements.
 * @tparam SD The State Descriptor.
 * @tparam Callable The type of the callable object representing the function.
 * @param f The callable object representing the function.
 * @param out The output (F-storage) array for the Jacobian matrix, of size (n x n)
 * @param x, y1, y2 Three temporary scratch states for intermediate computations
 * @param t The current time.
 * @param q The current state vector (size n).
 * @param h The perturbation vector to use for finite differences.
 */
template<typename T, typename SD, typename U, typename Callable, typename StepCaller>
constexpr void jac_approx_impl(Callable&& func, StepCaller&& step_elem, jacmat_t<T, SD>& out, state_t<T, SD>& x, state_t<T, SD>& y1, state_t<T, SD>& y2, const T& t, const state_t<T, SD>& q, const typename SD::template state_type<U>& args){

    using S = value_t<T, SD>;

    copy_state<SD>(x, q);

    SD::for_each([&, i = size_t{0}](S& xi, const S& qi, const U& arg) mutable {

        decltype(auto) vi = step_elem(arg, abs(qi));

        xi = qi - vi; //vi could be T or Scalar. The operation should work regardless
        func(y1, t, x);

        xi = qi + vi;
        func(y2, t, x);

        xi = qi; // restore it for the next iteration

        // Compute Jacobian element using central differences
        auto two_h = 2 * vi;
        SD::for_each([&, j = size_t{0}](const S& y1_j, const S& y2_j) mutable {
            out(j, i) = (y2_j - y1_j) / two_h;
            ++j;
        }, y1, y2);

        ++i;
    }, x, q, args);
}

} // namespace ode::detail



template<typename T, typename SD, typename Callable>
constexpr void jac_approx_autostep(Callable&& func, jacmat_t<T, SD>& out, state_t<T, SD>& x, state_t<T, SD>& y1, state_t<T, SD>& y2, const T& t, const state_t<T, SD>& q, const real_state_t<T, SD>& ftol){

    const T EPS_SQRT = sqrt(std::numeric_limits<T>::epsilon());

    detail::jac_approx_impl<T, SD>(
        std::forward<Callable>(func),
        [&](const T& ftol_i, const T& abs_qi){
            return EPS_SQRT * max_ref(ftol_i, abs_qi);
        },
        out, x, y1, y2, t, q, ftol
    );
}

template<typename T, typename SD, typename Callable>
constexpr void jac_approx_explicit(Callable&& func, jacmat_t<T, SD>& out, state_t<T, SD>& x, state_t<T, SD>& y1, state_t<T, SD>& y2, const T& t, const state_t<T, SD>& q, const state_t<T, SD>& delta_q){

    detail::jac_approx_impl<T, SD>(
        std::forward<Callable>(func),
        [&](const value_t<T, SD>& dq_i, const T& /*abs_qi*/){
            return dq_i;
        },
        out, x, y1, y2, t, q, delta_q
    );
}

}; // namespace ode


#endif // ODECRAFT_FINDIFF_HPP