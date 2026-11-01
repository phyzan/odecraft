#pragma once

#include "BaseSolver.hpp"


namespace ode{

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
template<isObserver<T, state_t<T, SD>> Callable>
bool BaseSolver<Derived, T, SD, OdeType, SP>::advance_until(const T& time, Callable&& observer){
    return this->generic_advance_until(time, std::forward<Callable>(observer), nullptr);
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
template<isObserver<T, state_t<T, SD>> Callable, isArray<T> ArrayType>
bool BaseSolver<Derived, T, SD, OdeType, SP>::advance_until(const T& time, Callable&& observer, ArrayType&& checkpoints){
    return this->generic_advance_until(
        time,
        std::forward<Callable>(observer),
        std::forward<ArrayType>(checkpoints)
    );
}


template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
template<isObserver<T, state_t<T, SD>> Callable>
BoxedInterp<T, SD> BaseSolver<Derived, T, SD, OdeType, SP>::interpolate_until(const T& time, Callable&& observer){
    return this->generic_interpolate_until(time, std::forward<Callable>(observer));
}


template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
template<typename Callable, typename ArrayLike>
bool BaseSolver<Derived, T, SD, OdeType, SP>::generic_advance_until(
    const T& time,
    Callable&& observer,
    ArrayLike&& checkpoints){

    static_assert(isObserver<Callable, T, State> || std::is_same_v<std::decay_t<Callable>, std::nullptr_t>, "Callable must be an observer or nullptr");
    static_assert(isArray<ArrayLike, T> || std::is_same_v<std::decay_t<ArrayLike>, std::nullptr_t>, "ArrayLike must be an array or nullptr");

    if (! this->is_running()){
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
    auto evolve = [&](State& worker) NDSPAN_LAMBDA_INLINE -> bool {
        bool res;
        while (
            (res = (this->is_running() && THIS->Adv_Impl(time)))
            && (time != this->t())
        ){
            bool obs_res;
            if constexpr (isObserver<Callable, T, State>){
                this->fill_current_vector(worker);
                obs_res = observer(this->t(), worker, nullptr);
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
            if constexpr (isObserver<Callable, T, State>){
                this->fill_current_vector(worker);
                observer(this->t(), worker, t_ptr);
            }            
            return true;
        } else {
            return this->t() * d >= time * d;
        }
    };

    // Safest option is to allocate a temporary here
    // which is expensive for heap-allocated workers.
    // However the item is not discarded immediately
    // after use in evolve(), as it is passed in the observer,
    // so using a pre-allocated scratch could have unexpected results
    // if the observer calls one of the solver's methods.
    State work(this->vector_new());

    if constexpr (!explicit_steps){
        return evolve(work);
    } else if (!has_checkpoints){
        return evolve(work);
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
        while (
            idx < checkpoints.size()
            && (success = (this->is_running() && this->generic_advance_until(checkpoints[validate_idx(idx)], observer, nullptr)))
            && (time != this->t())
        ){
            idx++;
        }

        if (this->t() != time && success){
            return evolve(work);
        } else{
            return success;
        }
    }

}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
template<typename Callable>
BoxedInterp<T, SD> BaseSolver<Derived, T, SD, OdeType, SP>::generic_interpolate_until(const T& time, Callable&& observer){

    // The chain grows as alternating point / open-interval / point entries, and always ends
    // on a point at the last time reached. Each callback bridges from that point to wherever
    // the solver now is, rather than appending the whole [t_old, t_new] step: a step is only
    // partly traversed whenever the solver stops inside it, so appending all of it overshoots.
    // That overshoot used to be reconciled later, when the solver came back round to t_new
    // exactly -- but after a masked event it restarts from the event and never reaches t_new,
    // leaving the chain permanently past its own end.
    pbox::Box<LinkedInterpolator<T, SD>> interp;
    {
        decltype(auto) worker = scratch_0.get();
        this->fill_current_vector(worker);
        interp = pbox::make_box<LinkedInterpolator<T, SD>>(this->t(), worker);
    }

    const T t_start = this->t();
    const bool reached = this->advance_until(
        time,
        [&](const T& t, State q, const T* t_ptr){
            // The solver is already at t, so the chain is extended before the observer is
            // asked: if it stops the integration here, the dense output still reaches t.
            // (covered, t) always lies within the step now held: `covered` is either the
            // previous t_new, which has become t_old, or a stop inside this same step.
            // Narrow the step's interpolator to exactly that span. adjust_* only move
            // the reported interval, bounded by the polynomial's own t1/t2, so the
            // evaluation is untouched.
            const T covered = interp->t_end();
            if (lt(covered, t)){
                BoxedInterp<T, SD> segment = this->state_interpolator(0, -1);
                if (covered != segment->t_start()){
                    segment->adjust_start(covered);
                }
                if (t != segment->t_end()){
                    segment->adjust_end(t);
                }
                interp->expand(std::move(segment));
                // q is the current vector at t, filled by generic_advance_until.
                interp->expand(std::make_unique<LocalInterpolator<T, SD>>(t, std::move(q)));
            }
            if constexpr (isObserver<Callable, T, State>){
                return observer(t, q, t_ptr);
            } else {
                return true;
            }
        }
    );

    // After an early stop (observer, terminating event or a dead solver) the chain ends where
    // the solver stopped, so the result covers exactly the span that was integrated.
    if (t_start != interp->t_start()){
        interp->adjust_start(t_start);
    }
    if (reached && time != interp->t_end()){
        interp->adjust_end(time);
    }
    interp->close_end();
    return interp;
}

} // namespace ode
