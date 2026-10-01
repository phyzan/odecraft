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

    decltype(auto) min_step = const_cache(hmin);
    decltype(auto) max_step = const_cache(hmax);
    decltype(auto) min_step_abs = const_cache(hmin_abs);
    decltype(auto) safety = const_cache(safety_);
    decltype(auto) max_factor = const_cache(max_factor_);
    decltype(auto) min_factor = const_cache(min_factor_);
    decltype(auto) err_exp = const_cache(err_exp_);
    decltype(auto) inc_exp = const_cache(inc_exp_);
    decltype(auto) min_err = const_cache(min_err_);
    decltype(auto) time = const_cache(state[0]);
    decltype(auto) habs_old = const_cache(state[1]);
    decltype(auto) q = const_cached_array<T, N>(state+2, n);

    // Each cache is followed immediately by its write-back guard; see Toolkit/Cache.hpp for why
    // the two are separate objects.
    decltype(auto) t_new = mut_cache(res[0]);
    decltype(auto) habs = mut_cache(res[1]);
    decltype(auto) q_out = mut_cached_array<T, N>(res+2, n);
    [[maybe_unused]] decltype(auto) t_new_sync = cache_write_back(t_new, res[0]);
    [[maybe_unused]] decltype(auto) habs_sync = cache_write_back(habs, res[1]);
    [[maybe_unused]] decltype(auto) q_out_sync = cache_write_back<T, N>(q_out, res+2);

    habs = habs_old;
    // T* q_out = res + 2;

    T factor;
    bool step_accepted = false;
    while (!step_accepted){
        const T h = habs * direction;
        const T err_norm = step_fn(t_new, q_out.data(), time, h, q.data());

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

        if (!all_are_finite<T>(q_out.data(), n)){
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
