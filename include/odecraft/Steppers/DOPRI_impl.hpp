#pragma once


#include <odecraft/Steppers/DOPRI.hpp>


namespace ode::detail{


// h is read once, before the loop: through a reference the compiler could not keep it in a
// register, since a store to r or out might change it.

template<typename SD, typename State, size_t NK, typename T, typename Tab, size_t... j>
XDIFF_FORCEINLINE void rk_stage(State& r, const State& q, const T& h_, const std::array<State, NK>& K, const Tab& A, size_t i, std::index_sequence<j...>){
    const cache_t<T> h = h_;
    SD::for_each([&](auto& r_k, const auto& q_k, const auto&... K_jk){
        r_k = q_k + h * (... + (A(i, j) * K_jk));
    }, r, q, K[j]...);
}

template<typename SD, typename State, size_t NK, typename T, typename Tab, size_t... j>
XDIFF_FORCEINLINE void rk_stage(State& r, const State& q, const T& h_, const std::array<State, NK>& K, const Tab& b, std::index_sequence<j...>){
    const cache_t<T> h = h_;
    SD::for_each([&](auto& r_k, const auto& q_k, const auto&... K_jk){
        r_k = q_k + h * (... + (b(j) * K_jk));
    }, r, q, K[j]...);
}

template<typename SD, typename State, size_t NK, typename T, typename Tab, size_t... j>
XDIFF_FORCEINLINE void rk_dot(State& out, const T& h_, const std::array<State, NK>& K, const Tab& D, size_t i, std::index_sequence<j...>){
    const cache_t<T> h = h_;
    SD::for_each([&](auto& out_k, const auto&... K_jk){
        out_k = h * (... + (D(i, j) * K_jk));
    }, out, K[j]...);
}


template<typename T, typename SD, size_t NS, size_t ORD, size_t... k>
void rk_interp_matrix(
    std::array<state_t<T, SD>, ORD>& coef_mat,
    const std::array<state_t<T, SD>, NS+1>& K,
    const CArray2D<T, NS+1, ORD>& P,
    std::index_sequence<k...>){

    // One pass per coefficient, which sums in a register rather than through coef_mat[j]
    for (size_t j=0; j < ORD; ++j){
        SD::for_each([&P, &j](auto& coef_ji, const auto&... K_ki){
            coef_ji = (... + (P(k, j) * K_ki));
        }, coef_mat[j], K[k]...);
    }

}


template<typename T, typename SD, typename StepFn>
StepResult rk_adapt_step(
    T& t_new_, T& habs_new_, state_t<T, SD>& q_new_,
    cache_t<T> t, cache_t<T> habs, cache_t<state_t<T, SD>> q,
    cache_t<T> min_step, cache_t<T> max_step, cache_t<T> min_step_abs,
    cache_t<T> safety, cache_t<T> max_factor, cache_t<T> min_factor,
    cache_t<T> err_exp, cache_t<T> inc_exp, cache_t<T> min_err,
    const int direction, StepFn&& step_fn){
    T factor;

    // ----------- Scratch pad ---------------
    auto  scratch    = make_scratch(t_new_, habs_new_, q_new_);
    auto& [t_new, habs_new, q_new] = scratch;
    auto  write_back = make_write_back(scratch, t_new_, habs_new_, q_new_);
    // ---------------------------------------

    habs_new = habs;

    bool step_accepted = false;
    while (!step_accepted){
        const T h = habs_new * direction;
        const T err_norm = step_fn(t_new, q_new, t, h, q);

        if (err_norm <= 1){
            step_accepted = true;
            if (2*err_norm < 1){
                const auto& err_clamped = max_ref(err_norm, min_err);
                set_min(factor, max_factor, safety * pow(err_clamped, inc_exp));
            } else {
                factor = 1;
            }
        } else {
            set_max(factor, min_factor, safety * pow(err_norm, err_exp));
        }

        if (!all_are_finite<T, SD>(q_new)){
            return StepResult::NonFiniteError;
        } else if (habs_new < min_step_abs){
            return StepResult::TinyStepError;
        } else if (!resize_step<T>(factor, habs_new, min_step, max_step)){
            break;
        }
    }
    return StepResult::Success;
}

} // namespace ode::detail
