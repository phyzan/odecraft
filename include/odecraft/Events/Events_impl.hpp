#ifndef ODECRAFT_EVENTS_IMPL_HPP
#define ODECRAFT_EVENTS_IMPL_HPP

#include "Events_mem_impl.hpp" // IWYU pragma: keep


namespace ode{

// EventBase implementations

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
EventBase<Derived, EP, T, State, MaskFunc>::EventBase(std::string name) requires std::is_same_v<MaskFunc, std::nullptr_t> : name_(std::move(name)) {
    if (name_.empty()){
        throw std::runtime_error("Please provide a non-empty name when instanciating an Event class");
    }
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
EventBase<Derived, EP, T, State, MaskFunc>::EventBase(std::string name, MaskFunc mask, bool delay_mask)
    : name_(std::move(name)),
    mask_(std::move(mask)),
    delay_mask_(delay_mask) {

    static_assert(isRhsFunc<MaskFunc, T, State>, "MaskFunc must be a valid right-hand side function for this constructor");

    if (name_.empty()){
        throw std::runtime_error("Please provide a non-empty name when instanciating an Event class");
    }
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
const std::string& EventBase<Derived, EP, T, State, MaskFunc>::name() const{
    return name_;
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
bool EventBase<Derived, EP, T, State, MaskFunc>::is_masked() const{
    if constexpr (std::is_same_v<MaskFunc, std::nullptr_t>){
        return false;
    } else if constexpr (isNullable<MaskFunc>){
        return mask_ != nullptr;
    } else {
        return true;
    }
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
bool EventBase<Derived, EP, T, State, MaskFunc>::mask_delayed() const{
    return delay_mask_ && this->is_masked();
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
void EventBase<Derived, EP, T, State, MaskFunc>::apply_mask(State& out, const T& t, const State& q) const{

    if constexpr (!std::is_same_v<MaskFunc, std::nullptr_t>){
        if (this->is_masked()){
            mask_(out, t, q);
            return;
        }
    }
    throw std::runtime_error("Event has not been initialized with a mask function");
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
size_t EventBase<Derived, EP, T, State, MaskFunc>::counter() const{
    return counter_;
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
std::unique_ptr<Event<T, State>> EventBase<Derived, EP, T, State, MaskFunc>::clone() const{
    return std::make_unique<Derived>(*THIS);
}


template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
int EventBase<Derived, EP, T, State, MaskFunc>::direction() const{
    return this->direction_;
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
bool EventBase<Derived, EP, T, State, MaskFunc>::is_located() const{
    return is_located_;
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
const T& EventBase<Derived, EP, T, State, MaskFunc>::t_start() const{
    return start_;
}


template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
void EventBase<Derived, EP, T, State, MaskFunc>::setup(T t_start, const State& state, int direction){
    // checks that it has not been already setup
    // no other modifiers can be called if setup has not been called yet
    assert(!this->is_setup_ && "Setup takes place only once");

    if (abs(direction) != 1){
        throw std::domain_error("Event direction must be 1 or -1");
    }
    worker_ = state;
    direction_ = direction;
    is_setup_ = true;
    start_ = t_start;
}


template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
bool EventBase<Derived, EP, T, State, MaskFunc>::locate(T& out,
    const T& t_old,
    const T& t_new,
    const State& q_old,
    const State& q_new,
    const interp_t<T, State>& interp){

    return this->locate_state(out, t_old, t_new, q_old, q_new, interp);
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
bool EventBase<Derived, EP, T, State, MaskFunc>::lock(){
    if (is_located_){
        THIS->register_impl();
        return true;
    }else{
        return false;
    }
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
void EventBase<Derived, EP, T, State, MaskFunc>::reset(int direction){
    THIS->reset_impl(direction);
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
void EventBase<Derived, EP, T, State, MaskFunc>::register_impl(){
    counter_++;
}

template<typename Derived, EventPolicy EP, typename T, typename State, typename MaskFunc>
void EventBase<Derived, EP, T, State, MaskFunc>::reset_impl(int direction){
    counter_ = 0;
    is_located_ = false;
    direction_ = (direction == 0) ? direction_ : direction;
}

// PreciseEvent implementations

template<typename T, typename State, isObjFun<T, State> Target, typename MaskFunc, EventPolicy EP, typename Derived>
PreciseEvent<T, State, Target, MaskFunc, EP, Derived>::PreciseEvent(std::string name, Target objfun, T event_tol, int dir) requires std::is_same_v<MaskFunc, std::nullptr_t> : Base(std::move(name)), target(std::move(objfun)), crossing_dir(dir), ftol(event_tol) {}

template<typename T, typename State, isObjFun<T, State> Target, typename MaskFunc, EventPolicy EP, typename Derived>
PreciseEvent<T, State, Target, MaskFunc, EP, Derived>::PreciseEvent(std::string name, Target objfun, T event_tol, int dir, MaskFunc mask, bool delay_mask) : Base(std::move(name), std::move(mask), delay_mask), target(std::move(objfun)), crossing_dir(dir), ftol(event_tol) {}

template<typename T, typename State, isObjFun<T, State> Target, typename MaskFunc, EventPolicy EP, typename Derived>
T PreciseEvent<T, State, Target, MaskFunc, EP, Derived>::obj_fun(const T& t, const State& q) const{
    return target(t, q);
}

template<typename T, typename State, isObjFun<T, State> Target, typename MaskFunc, EventPolicy EP, typename Derived>
int PreciseEvent<T, State, Target, MaskFunc, EP, Derived>::sign_change_dir() const{
    return crossing_dir;
}


// PeriodicEvent implementations

template<typename T, typename State, typename MaskFunc, EventPolicy EP, typename Derived>
PeriodicEvent<T, State, MaskFunc, EP, Derived>::PeriodicEvent(std::string name, T period) requires std::is_same_v<MaskFunc, std::nullptr_t> : Base(std::move(name)), period_(std::move(period)) {}

template<typename T, typename State, typename MaskFunc, EventPolicy EP, typename Derived>
PeriodicEvent<T, State, MaskFunc, EP, Derived>::PeriodicEvent(std::string name, T period, MaskFunc mask, bool delay_mask) : Base(std::move(name), std::move(mask), delay_mask), period_(std::move(period)) {}

template<typename T, typename State, typename MaskFunc, EventPolicy EP, typename Derived>
const T& PeriodicEvent<T, State, MaskFunc, EP, Derived>::period() const{
    return period_;
}

template<typename T, typename State, typename MaskFunc, EventPolicy EP, typename Derived>
T PeriodicEvent<T, State, MaskFunc, EP, Derived>::get_t(size_t n) const{
    // n*period() first: direction()*n would convert direction() = -1 to size_t and wrap around
    return this->t_start() + n*this->period()*this->direction();
}

template<typename T, typename State, typename MaskFunc, EventPolicy EP, typename Derived>
void PeriodicEvent<T, State, MaskFunc, EP, Derived>::register_impl(){
    Base::register_impl();
    n_ = n_aux_;
}

template<typename T, typename State, typename MaskFunc, EventPolicy EP, typename Derived>
void PeriodicEvent<T, State, MaskFunc, EP, Derived>::reset_impl(int direction){
    Base::reset_impl(direction);
    n_ = n_aux_ = 0;
}



template<typename T, typename State>
EventCollection<T, State>::EventCollection(EventList<T, State> evs) : events(evs.size()), event_times(evs.size()), detection_order(evs.size()), located(evs.size()) {

    if (evs.size() == 0){return;}

    for (size_t i=0; i<evs.size(); i++){
        if (idx_of_name.find(evs[i]->name()) != idx_of_name.end()){
            throw std::runtime_error("Duplicate Event name not allowed: " + evs[i]->name());
        }
        idx_of_name[evs[i]->name()] = static_cast<int>(i);
        this->events[i] = std::move(evs[i]);
    }
}

template<typename T, typename State>
const Event<T, State>& EventCollection<T, State>::event(size_t event_idx) const{
    return *events[event_idx].get_raw_pointer();
}

template<typename T, typename State>
T EventCollection<T, State>::get_time(size_t detection_idx) const{
    assert(detection_idx < detections && "Out of bounds detection_idx requested in get_time");
    return event_times[detection_order[detection_idx]];
}

template<typename T, typename State>
const Event<T, State>* EventCollection<T, State>::get_event(size_t detection_idx) const{
    if (detection_idx >= detections){
        return nullptr;
    }else{
        return events[detection_order[detection_idx]].get_raw_pointer();
    }
}

template<typename T, typename State>
bool EventCollection<T, State>::is_located(size_t event_idx) const{
    return located[event_idx];
}


template<typename T, typename State>
int EventCollection<T, State>::event_idx(const std::string& name) const{
    auto it = idx_of_name.find(name);
    if (it != idx_of_name.end()){
        return int(it->second);
    }
    return -1;
}

template<typename T, typename State>
size_t EventCollection<T, State>::get_event_idx(size_t detection_idx) const{
    assert(detection_idx < detections && "Detection index out of bounds in get_event_idx");
    return detection_order[detection_idx];
}

template<typename T, typename State>
size_t EventCollection<T, State>::size() const{
    return events.size();
}

template<typename T, typename State>
size_t EventCollection<T, State>::detection_size() const{
    return detections;
}

template<typename T, typename State>
void EventCollection<T, State>::setup(T t_start, const State& state, int direction){
    worker_ = state;
    masked_data.masked_vector = state;
    for (size_t i=0; i<this->size(); i++){
        events[i]->setup(t_start, state, direction);
    }
}



template<typename T, typename State>
const MaskedState<T, State>* EventCollection<T, State>::masked_state() const{
    return has_masked_state ? &masked_data : nullptr;
}

template<typename T, typename State>
void EventCollection<T, State>::reset(int direction){

    for (size_t i=0; i<this->size(); i++){
        events[i]->reset(direction);
        located[i] = false;
    }

    detections = 0;
    has_masked_state = false;
}



template<typename T, typename State>
std::vector<EventOptions> EventCollection<T, State>::validate_events(const std::vector<EventOptions>& options) const {
    size_t Nevs = this->size();
    std::vector<EventOptions> res(Nevs);
    bool found;
    for (const auto & option : options) {
        found = false;
        for (size_t j=0; j<Nevs; j++){
            if (this->event(j).name() == option.name){
                found = true;
                break;
            }
        }
        if (!found){
            throw std::logic_error("Event name \""+option.name+"\" is invalid");
        }
    }

    for (size_t i=0; i<Nevs; i++){
        found = false;
        for (const auto& option : options){
            if (option.name == this->event(i).name()){
                found = true;
                res[i] = option;
                res[i].max_events = ndspan::max(option.max_events, -1);
                break;
            }
        }
        if (!found){
            res[i] = {this->event(i).name()};
        }
    }
    return res;
}


} // namespace ode

#endif // ODECRAFT_EVENTS_IMPL_HPP