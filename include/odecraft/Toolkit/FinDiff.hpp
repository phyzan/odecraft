#pragma once

#include "Tools.hpp"
#include <tuple>


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
 * @param out The Jacobian, written one column at a time with SD::JD::linalg::for_each_column
 * @param x, y1, y2 Three temporary scratch states for intermediate computations
 * @param t The current time.
 * @param q The current state vector (size n).
 * @param h The perturbation vector to use for finite differences.
 */
template<typename T, typename SD, typename U, typename Callable, typename StepCaller>
constexpr void jac_approx_impl(Callable&& func, StepCaller&& step_elem, jacmat_t<T, SD>& out, state_t<T, SD>& x, state_t<T, SD>& y1, state_t<T, SD>& y2, const T& t, const state_t<T, SD>& q, const typename SD::template state_type<U>& args){

    using S = value_t<T, SD>;

    copy_state<SD>(x, q);

    // Column j of the Jacobian, by central differences around q_j, into y2
    SD::JD::linalg::for_each_column([](S& J_ij, const S& df_i){
        J_ij = df_i;
    }, [&](S& xj, const S& qj, const U& arg){

        decltype(auto) vj = step_elem(arg, abs(qj));

        xj = qj - vj; //vj could be T or Scalar. The operation should work regardless
        func(y1, t, x);

        xj = qj + vj;
        func(y2, t, x);

        xj = qj; // restore it for the next column

        auto two_h = 2 * vj;
        SD::for_each([&two_h](S& y2_i, const S& y1_i){
            y2_i = (y2_i - y1_i) / two_h;
        }, y2, y1);
        return std::tie(y2);
    }, std::tie(out), x, q, args);
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