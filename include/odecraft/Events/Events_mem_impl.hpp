#ifndef ODECRAFT_EVENTS_MEM_IMPL_HPP
#define ODECRAFT_EVENTS_MEM_IMPL_HPP

#include "Events.hpp"


namespace ode{

template<typename Derived, EventPolicy EP, typename T, typename MaskFunc>
template<isStateInterp<T> Callable>
bool EventBase<Derived, EP, T, MaskFunc>::locate_state(T& out, State<T> before, State<T> after, Callable&& f){
    assert(this->is_setup_ && "Call setup() method before trying to locate an event");
    assert((sgn(before.t(), after.t()) == this->direction_) && "Invalid direction");
    if (THIS->locate_impl(out, before, after, std::forward<Callable>(f))){
        is_located_ = true;
        return true;
    }else {
        is_located_ = false;
        return false;
    }
}

template<typename Derived, EventPolicy EP, typename T, typename MaskFunc>
template<isStateInterp<T> Callable>
bool EventBase<Derived, EP, T, MaskFunc>::locate_impl(T& /*t*/, State<T> /*before*/, State<T> /*after*/, Callable&& /*f*/) const{
    static_assert(false, "static override");
    return false;
}

template<typename T, isObjFun<T> Target, typename MaskFunc, EventPolicy EP, typename Derived>
template<isStateInterp<T> Callable>
bool PreciseEvent<T, Target, MaskFunc, EP, Derived>::locate_impl(T& t, State<T> before, State<T> after, Callable&& f) const{
    T val1 = this->obj_fun(before.t(), before.vector());
    T val2 = this->obj_fun(after.t(), after.vector());

    int t_dir = this->direction();
    int d = this->sign_change_dir();
    if ( (((d == 0) && (val1*val2 < 0)) || (t_dir*d*val1 < 0 && 0 < t_dir*d*val2)) && (abs<T>(val1) > ftol)){
        T* vec = this->worker.data();

        auto obj_fun_scalar = [&](T t_dummy) NDSPAN_LAMBDA_INLINE{
            f(vec, t_dummy); // interpolate the state vector at time t_dummy, and pass the value on vec
            return this->obj_fun(t_dummy, vec);
        };

        t = bisect<T, RootPolicy::Right>(obj_fun_scalar, before.t(), after.t(), this->ftol);
        return true;
    }
    return false;
}


template<typename T, typename MaskFunc, EventPolicy EP, typename Derived>
template<isStateInterp<T> Callable>
bool PeriodicEvent<T, MaskFunc, EP, Derived>::locate_impl(T& t, State<T> before, State<T> after, Callable&& /*f*/) const{
    n_aux_ = n_;
    int d = this->direction();

    while (get_t(++n_aux_)*d <= before.t()*d){}

    if (get_t(n_aux_)*d <= after.t()*d){
        t = get_t(n_aux_);
        return true;
    } else {
        return false;
    }
}



template<typename T>
template<isStateInterp<T> Callable>
bool EventCollection<T>::detect_all_between(State<T> before, State<T> after, Callable&& f) {
    if (this->size() == 0){
        return false;
    }

    detections = 0;
    has_masked_state = false;
    located.fill(false);

    // Locate all events and record which ones were found
    for (size_t i=0; i<this->size(); i++){
        if (events[i]->locate(event_times[i], before, after, std::forward<Callable>(f))){
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

        // Interpolate state at mask time
        f(worker.data(), t_mask);

        // Apply the mask transformation
        events[masked_idx]->apply_mask(masked_data.masked_vector.data(), t_mask, worker.data());
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

#endif // ODECRAFT_EVENTS_MEM_IMPL_HPP