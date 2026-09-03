#ifndef ODECRAFT_BASESOLVER_MEM_IMPL_HPP
#define ODECRAFT_BASESOLVER_MEM_IMPL_HPP

#include "BaseSolver.hpp"


namespace ode{

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<isObserver<T> Callable>
bool BaseSolver<Derived, T, N, SP, OdeType>::advance_until(const T& time, Callable&& observer){
    return this->generic_advance_until(time, std::forward<Callable>(observer), nullptr);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<isObserver<T> Callable, isArray<T> ArrayType>
bool BaseSolver<Derived, T, N, SP, OdeType>::advance_until(const T& time, Callable&& observer, ArrayType&& checkpoints){
    return this->generic_advance_until(
        time,
        std::forward<Callable>(observer),
        std::forward<ArrayType>(checkpoints)
    );
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<isObserver<T> Callable>
BoxedInterp<T, N> BaseSolver<Derived, T, N, SP, OdeType>::interpolate_until(const T& time, Callable&& observer){
    return this->generic_interpolate_until(time, std::forward<Callable>(observer));
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename... Args>
bool BaseSolver<Derived, T, N, SP, OdeType>::Adv_Impl(Args&&... args){
    const int d = this->direction();
    if constexpr (sizeof...(Args) > 0){
        T time_floor = nearest_time(args...);
        if (this->is_at_new_state() && (time_floor*d <= t_new()*d)){
            return false;
        } else if (this->is_at_new_state()){
            decltype(auto) scratch_state = this->scratch_.state();
            StepResult result = this->adapt_impl(scratch_state.data(), this->new_state_ptr());
            if (validate_it(result, scratch_state.data())){
                // ======== update internal states ========
                std::swap(old_state_, new_state_);
                std::swap(new_state_, scratch_state);
                for (size_t i=0; i<scratch_state.size(); i++){
                    true_state_[i] = new_state_[i];
                }
                use_new_state_ = true;
                step_count_++;
                // ==========================================
                T new_floor;
                if (ODECRAFT_CALL_DERIVED(RequestTimeFloor, new_floor)){
                    assert((new_floor*d > t_old()*d && new_floor*d <= t_new()*d) && "Invalid floor requested, with additional requests");
                    time_floor = nearest_time(new_floor, time_floor);
                }
                
                if (time_floor*d < t_new()*d){
                    this->move_state(time_floor);
                }
                return true;
            }else{
                return false;
            }
        } else if (time_floor*d < t_new()*d){
            this->move_state(time_floor);
            return true;
        } else {
            this->move_state(t_new());
            return true;
        }
    } else if (this->is_at_new_state()){
        decltype(auto) scratch_state = this->scratch_.state();
        StepResult result = this->adapt_impl(scratch_state.data(), this->new_state_ptr());
        if (validate_it(result, scratch_state.data())){
            // ======== update internal states ========
                std::swap(old_state_, new_state_);
                std::swap(new_state_, scratch_state);
                for (size_t i=0; i<scratch_state.size(); i++){
                    true_state_[i] = new_state_[i];
                }
                use_new_state_ = true;
                step_count_++;
            // ==========================================
            T new_floor;
            if (ODECRAFT_CALL_DERIVED(RequestTimeFloor, new_floor) && new_floor*d < t_new()*d){
                assert((new_floor*d > t_old()*d && new_floor*d <= t_new()*d) && "Invalid floor requested without additional requests.");
                this->move_state(new_floor);
            }
            return true;
        } else {
            return false;
        }
    } else {
        this->move_state(t_new());
        return true;
    }
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename... U>
T BaseSolver<Derived, T, N, SP, OdeType>::nearest_time(const U&... t) const{
    static_assert(sizeof...(U) > 0, "BaseSolver::nearest_time requires at least one argument");
    return nearest_time_priv(t...);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename Callable, typename ArrayLike>
bool BaseSolver<Derived, T, N, SP, OdeType>::generic_advance_until(
    const T& time,
    Callable&& observer,
    ArrayLike&& checkpoints){

    static_assert(isObserver<Callable, T> || std::is_same_v<std::decay_t<Callable>, std::nullptr_t>, "Callable must be an observer or nullptr");
    static_assert(isArray<ArrayLike, T> || std::is_same_v<std::decay_t<ArrayLike>, std::nullptr_t>, "ArrayLike must be an array or nullptr");

    if (this->is_dead()){
        this->warn_dead();
        return false;
    }

    int d = this->direction();
    if (time == this->t()){
        return false;
    } else if (time*d < this->t()*d) {
        throw std::runtime_error(GetStr("Cannot advance until time ", time, " because it is in the opposite direction of integration. Current time is ", this->t(), " and direction is ", d, "."));
    }

    constexpr bool explicit_steps = !std::is_same_v<std::decay_t<ArrayLike>, std::nullptr_t>;

    const bool has_checkpoints = [&](){
        if constexpr (explicit_steps) {
            return checkpoints.size() > 0;
        } else {
            return false;
        }
    }();

    const T& t_dual = [&]() -> const T& {
        if constexpr (explicit_steps){
            if (has_checkpoints) {
                return checkpoints[checkpoints.size()-1];
            }
        }
        return time;
    }();


    bool success;
    auto evolve = [&]() NDSPAN_LAMBDA_INLINE -> bool {
        bool res;
        while ((res = (this->is_running() && Accessor::call_Adv_Impl(*THIS, time))) && (time != this->t())){
            bool obs_res;
            if constexpr (isObserver<Callable, T>){
                obs_res = observer(this->t(), this->true_state_ptr()+2, nullptr);
            } else{
                obs_res = true;
            }
            if (!obs_res){
                // the observer itself might have advanced the solver to the same target time, so its worth making this check.
                return this->t() * d >= time * d;
            }
        }

        if (res){
            const T* t_ptr = (!explicit_steps || (has_checkpoints && t_dual == time)) ? &t_dual : nullptr;
            if constexpr (isObserver<Callable, T>){
                observer(this->t(), this->true_state_ptr()+2, t_ptr);
            }            
            return true;
        } else {
            return this->t() * d >= time * d;
        }
    };

    if constexpr (!explicit_steps){
        return evolve();
    } else if (!has_checkpoints){
        return evolve();
    } else if (t_dual*d > time*d){
        // a.t.p. t_dual is the last element of `checkpoints`
        throw std::runtime_error(GetStr("Invalid extra steps: last extra step is ", t_dual, " but target time is ", time, ". Extra steps must be in the same direction and between the current time and the target time."));
    }else{
        auto validate_idx = [&](size_t idx) NDSPAN_LAMBDA_INLINE -> size_t {
            if (checkpoints[idx]*d <= this->t()*d){
                throw std::runtime_error(GetStr("Invalid extra step: ", checkpoints[idx], ". Extra steps must be in the same direction and between the current time (", this->t(), ") and the target time (", time, ")."));
            }
            return idx;
        };
        size_t idx = 0;
        while (idx < checkpoints.size() && (success = (this->is_running() && this->generic_advance_until(checkpoints[validate_idx(idx)], observer, nullptr))) && (time != this->t())){
            idx++;
        }

        if (this->t() != time && success){
            return evolve();
        } else{
            return success;
        }
    }

}



template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename Callable>
BoxedInterp<T, N> BaseSolver<Derived, T, N, SP, OdeType>::generic_interpolate_until(const T& time, Callable&& observer){

    pbox::Box<LinkedInterpolator<T, N>> interp = pbox::make_box<LinkedInterpolator<T, N>>(this->t_old(), this->vector_old().data(), this->nsys());
    bool current_state_is_new = false;
    if (!this->is_at_new_state()){
        interp->expand_by_owning(this->state_interpolator(0, -1));
    }else{
        current_state_is_new = true;
    }

    const T t_start = this->t();
    if (this->advance_until(
        time,
        [&](const T& t, const T* q, const T* t_ptr){
            bool obs_res;
            if constexpr (isObserver<Callable, T>){
                obs_res = observer(t, q, t_ptr);
            } else{
                obs_res = true;
            }
            if (obs_res){
                if (this->is_at_new_state()){
                    if (current_state_is_new){
                        interp->expand_by_owning(this->state_interpolator(0, -1));
                    }
                    interp->expand_by_owning(std::make_unique<LocalInterpolator<T, N>>(this->t(), this->vector().data(), this->nsys()));
                    current_state_is_new = true;
                } else if (current_state_is_new) {
                    interp->expand_by_owning(this->state_interpolator(0, -1));
                    current_state_is_new = false;
                }
                return true;
            } else {
                return false;
            }
        })
    ){
        if (t_start != interp->t_start()){
            interp->adjust_start(t_start);
        }
        if (time != interp->t_end()){
            interp->adjust_end(time);
        }
        interp->close_end();
        return interp;
    } else {
        return BoxedInterp<T, N>();
    }
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename A, typename... Rest>
const T& BaseSolver<Derived, T, N, SP, OdeType>::nearest_time_priv(const A& t_a, const Rest&... t_rest) const{
    if constexpr (sizeof...(Rest) > 0){
        return nearest_time_helper(t_a, t_rest...);
    } else {
        return t_a;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename A, typename B, typename... Rest>
const T& BaseSolver<Derived, T, N, SP, OdeType>::nearest_time_helper(const A& t_a, const B& t_b, const Rest&... t_rest) const{
    if constexpr (sizeof...(Rest) > 0){
        return nearest_time_helper(nearest_of(t_a, t_b), t_rest...);
    }else{
        return nearest_of(t_a, t_b);
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename A, typename B>
const T& BaseSolver<Derived, T, N, SP, OdeType>::nearest_of(const A& t_a, const B& t_b) const{
    if (this->direction() == 1){
        return (t_a < t_b ? t_a : t_b);
    }else{
        return (t_a > t_b ? t_a : t_b);
    }
}

} // namespace ode

#endif // ODECRAFT_BASESOLVER_MEM_IMPL_HPP