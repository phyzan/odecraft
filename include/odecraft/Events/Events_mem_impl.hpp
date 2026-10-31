#pragma once

#include "Events.hpp"


namespace ode{

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
template<isStateInterp<T, State> Callable>
bool EventBase<Derived, EP, T, State, MaskFunc>::locate_state(T& out,
    const T& t_old,
    const T& t_new,
    const State& q_old,
    const State& q_new,
    Callable&& f){

    assert(this->is_setup_ && "Call setup() method before trying to locate an event");
    assert((sgn(t_old, t_new) == this->direction_) && "Invalid direction");
    if (THIS->locate_impl(out, t_old, t_new, q_old, q_new, std::forward<Callable>(f))){
        is_located_ = true;
        return true;
    }else {
        is_located_ = false;
        return false;
    }
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
template<isStateInterp<T, State> Callable>
bool EventBase<Derived, EP, T, State, MaskFunc>::locate_impl(T& /*t*/,
    const T& /*t_old*/,
    const T& /*t_new*/,
    const State& /*q_old*/,
    const State& /*q_new*/,
    Callable&& /*f*/) const{

    // Always false but only fires if the function is instantiated
    static_assert(sizeof(Callable) == 0, "static override");
    return false;
}

template<typename T, typename State, isObjFun<T, State> Target, typename MaskFunc, EventPolicy EP, typename Derived>
template<isStateInterp<T, State> Callable>
bool PreciseEvent<T, State, Target, MaskFunc, EP, Derived>::locate_impl(T& t_out,
    const T& t_old,
    const T& t_new,
    const State& q_old,
    const State& q_new,
    Callable&& f) const{

    T f_old = this->obj_fun(t_old, q_old);

    int dir = this->direction() * this->sign_change_dir();

    // Making sure that the previous step was not a detected event
    if (abs(f_old) > ftol){
        if (dir * f_old > 0){
            return false;
        } else {
            T f_new = this->obj_fun(t_new, q_new);
            decltype(auto) worker = this->worker_.get();

            RootStatus rs = bisect_root(
                t_out,
                static_cast<T*>(nullptr),
                [&](const T& t){
                    // interpolate the state vector at time t, and pass the value on vec
                    f(worker, t);
                    return this->obj_fun(t, worker);
                },
                t_old, t_new,
                std::move(f_old), std::move(f_new),
                this->ftol
            );

            return rs == RootStatus::Success;
        }
    } else {
        return false;
    }
}


template<typename T, typename State, typename MaskFunc, EventPolicy EP, typename Derived>
template<isStateInterp<T, State> Callable>
bool PeriodicEvent<T, State, MaskFunc, EP, Derived>::locate_impl(T& t,
    const T& t_old,
    const T& t_new,
    const State& /*q_old*/,
    const State& /*q_new*/,
    Callable&& /*f*/) const{

    n_aux_ = n_;
    int d = this->direction();

    while (get_t(++n_aux_)*d <= t_old*d){}

    if (get_t(n_aux_)*d <= t_new*d){
        t = get_t(n_aux_);
        return true;
    } else {
        return false;
    }
}



template<typename T, typename State>
template<isStateInterp<T, State> Callable>
bool EventCollection<T, State>::detect_all_between(
    const T& t_old,
    const T& t_new,
    const State& q_old,
    const State& q_new,
    Callable&& f) {

    if (this->size() == 0){
        return false;
    }

    detections = 0;
    has_masked_state = false;
    located.assign(located.size(), false);

    // Locate all events and record which ones were found
    for (size_t i=0; i<this->size(); i++){
        if (events[i]->locate(event_times[i], t_old, t_new, q_old, q_new, std::forward<Callable>(f))){
            located[i] = true;
            detection_order[detections++] = i;
        }
    }

    if (detections == 0){
        return false;
    }

    // Sort detected events by time (respecting integration direction), with index as tiebreaker
    int dir = events[0]->direction();
    std::sort(detection_order.data(), detection_order.data() + detections,
        [&](size_t a, size_t b){
            T ta = dir * event_times[a];
            T tb = dir * event_times[b];
            return (ta < tb) || (ta == tb && a < b);
        });

    // Remove simultaneous events, keeping only the first one (lowest index)
    size_t write = 1;
    for (size_t read = 1; read < detections; read++){
        if (event_times[detection_order[read]] != event_times[detection_order[read - 1]]){
            detection_order[write++] = detection_order[read];
        } else {
            // Discard this event - mark as not located
            located[detection_order[read]] = false;
        }
    }

    detections = write;

    // Find the first masked event and truncate detections after it
    size_t first_masked = detections;
    for (size_t i=0; i<detections; i++){
        size_t idx = detection_order[i];
        if (events[idx]->is_masked()){
            first_masked = i;
            break;
        }
    }

    // If a masked event was found, compute the masked state and truncate
    if (first_masked < detections){
        size_t masked_idx = detection_order[first_masked];
        T t_mask = event_times[masked_idx];

        // Apply the mask transformation
        if (t_mask == t_new){
            events[masked_idx]->apply_mask(masked_data.masked_vector, t_mask, q_new);
        } else {
            // Interpolate state at mask time
            decltype(auto) worker = this->worker_.get();
            f(worker, t_mask);
            events[masked_idx]->apply_mask(masked_data.masked_vector, t_mask, worker);
        }
                
        masked_data.time = t_mask;
        masked_data.idx = masked_idx;
        has_masked_state = true;

        // Mark events after the masked event as not located
        for (size_t i = first_masked + 1; i < detections; i++){
            located[detection_order[i]] = false;
        }

        // Keep only events up to and including the first masked event
        detections = first_masked + 1;
    }

    return true;
}


} // namespace ode
