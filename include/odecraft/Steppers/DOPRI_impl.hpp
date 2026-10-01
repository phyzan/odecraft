#ifndef ODECRAFT_DOPRI_IMPL_HPP
#define ODECRAFT_DOPRI_IMPL_HPP

#include "odecraft/Toolkit/Cache.hpp"
#include "odecraft/Toolkit/Tools.hpp"
#include <odecraft/Steppers/DOPRI.hpp>


namespace ode::detail{

template<typename T>
XDIFF_FORCEINLINE void rk_interp_matrix(T* coef_mat, const T* K, const T* K0, const T* KF, const T* P, size_t Nstages, size_t order, size_t n){
    // The Nstages+1 stage rows are no longer one contiguous block: row 0 is K0, rows
    // 1..Nstages-1 live in K, and the final (FSAL) row is KF.
    for (size_t i = 0; i < n; i++){
        for (size_t j = 0; j < order; j++){
            T sum = K0[i] * P[j] + KF[i] * P[Nstages*order + j];
            for (size_t k = 1; k < Nstages; k++){
                sum += K[(k-1)*n + i] * P[k*order + j];
            }
            coef_mat[i*order + j] = sum;
        }
    }
}





template<size_t N, typename T, typename StepFn>
XDIFF_FORCEINLINE StepResult rk_adapt_step(T* res, const T* state, size_t n,
                          const T& hmin, const T& hmax, const T& hmin_abs,
                          const T& safety_, const T& max_factor_, const T& min_factor_,
                          const T& err_exp_, const T& inc_exp_, const T& min_err_,
                          int direction, StepFn&& step_fn){
    T factor;
    // ----- Const cache ------
    const auto& [
        min_step,
        max_step,
        min_step_abs,
        safety,
        max_factor,
        min_factor,
        err_exp,
        inc_exp,
        min_err,
        time,
        habs_old
    ] = make_scratch_view(
        hmin, hmax, hmin_abs, safety_, max_factor_,
        min_factor_, err_exp_, inc_exp_, min_err_, state[0], state[1]
    );

    auto q = cache_array<N>(state+2, n);
    // ------------------------

    // ----------- Scratch pad ---------------
    auto  scratch = make_scratch(res[0], res[1]);
    auto& [t_new, habs] = scratch;

    auto q_out = cache_mut_array<N>(res+2, n);

    [[maybe_unused]] auto t_new_write_back = make_write_back(t_new, res[0]);
    [[maybe_unused]] auto habs_write_back  = make_write_back(habs, res[1]);
    [[maybe_unused]] auto q_out_write_back  = make_array_write_back<N>(q_out, res+2);
    // ---------------------------------------

    habs = habs_old;

    bool step_accepted = false;
    while (!step_accepted){
        const T h = habs * direction;
        const T err_norm = step_fn(t_new, q_out, time, h, q);

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

        if (!all_are_finite<T>(q_out, n)){
            return StepResult::NonFiniteError;
        } else if (habs < min_step_abs){
            return StepResult::TinyStepError;
        } else if (!resize_step<T>(factor, habs, min_step, max_step)){
            break;
        }
    }
    return StepResult::Success;
}

} // namespace ode::detail



#endif // ODECRAFT_DOPRI_IMPL_HPP
