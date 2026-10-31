#pragma once

#include "RichBase.hpp"
#include <string_view>

namespace ode{

// PUBLIC ACCESSORS

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
const EventCollection<T, typename RichSolver<Derived, T, SD, JD, OdeType, SP>::State>& RichSolver<Derived, T, SD, JD, OdeType, SP>::event_col() const{
    return evt_col;
}

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
bool RichSolver<Derived, T, SD, JD, OdeType, SP>::at_event(int event_idx) const{
    if (event_idx == -1){
        return is_at_event;
    } else if (EventState<T, State> es = this->current_event()){
        return es.idx == size_t(event_idx);
    } else {
        return false;
    }
}

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
int RichSolver<Derived, T, SD, JD, OdeType, SP>::event_idx(const std::string& name) const{
    return evt_col.event_idx(name);
}

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
std::vector<size_t> RichSolver<Derived, T, SD, JD, OdeType, SP>::toEventIdx(const std::vector<std::string>& event_names) const{
    std::vector<size_t> event_idx(event_names.size());
    for (size_t i = 0; i < event_names.size(); ++i){
        int idx = this->event_idx(event_names[i]);
        if (idx == -1){
            throw std::out_of_range("Invalid event name: " + event_names[i]);
        }
        event_idx[i] = size_t(idx);
    }
    return event_idx;
}

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
void RichSolver<Derived, T, SD, JD, OdeType, SP>::show_state(std::ostream& out, int prec) const{
    Base::show_state(out, prec);
    std::string_view event_name = this->current_event().event ? this->current_event().event->name().c_str() : "";
    out << "\tEvent     : " << event_name << "\n";
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
RichSolver<Derived, T, SD, JD, OdeType, SP>::RichSolver(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step, T max_step, T stepsize, int dir, EventList<T, State> evs) : Base(std::move(ode), t0, std::move(q0), std::move(rtol), std::move(atol), std::move(min_step), std::move(max_step), std::move(stepsize), dir), evt_col(std::move(evs)) {
    this->evt_col.setup(t0, this->vector_new(), this->direction());
}

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
bool RichSolver<Derived, T, SD, JD, OdeType, SP>::advance_to_event(const std::vector<size_t>& event_idx){
    for (size_t idx : event_idx){
        if (idx >= evt_col.size()){
            throw std::out_of_range("Invalid event index passed to advance_to_event: " + std::to_string(idx));
        }
    }
    if (evt_col.size() == 0){
        return false;
    }
    do{
        if (!this->advance()){
            return false;
        } else if (EventState<T, State> es = this->current_event()){
            if (event_idx.empty() || std::find(event_idx.begin(), event_idx.end(), es.idx) != event_idx.end()){
                return true;
            }
        }
    }while (true);
    return true;
}


template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
bool RichSolver<Derived, T, SD, JD, OdeType, SP>::advance_to_event(const T& tmax, const std::vector<size_t>& event_idx){
    for (size_t idx : event_idx){
        if (idx >= evt_col.size()){
            throw std::out_of_range("Invalid event index passed to advance_to_event: " + std::to_string(idx));
        }
    }
    if (evt_col.size() == 0){
        return false;
    }
    bool success = false;
    Base::advance_until(
        tmax,
        [&](const T& /*t*/, const State& /*state*/, const T* /*extra*/) NDSPAN_LAMBDA_INLINE {
            if (EventState<T, State> es = this->current_event()){
                if (event_idx.empty() || std::find(event_idx.begin(), event_idx.end(), es.idx) != event_idx.end()){
                    success = true;
                    return false; // stop advancing
                }
            }
            return true; // continue advancing
        }
    );
    return success;
}

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
bool RichSolver<Derived, T, SD, JD, OdeType, SP>::advance_to_event(const std::vector<std::string>& event_names){
    std::vector<size_t> event_idx = this->toEventIdx(event_names);
    return this->advance_to_event(event_idx);
}

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
bool RichSolver<Derived, T, SD, JD, OdeType, SP>::advance_to_event(const T& tmax, const std::vector<std::string>& event_names){
    std::vector<size_t> event_idx = this->toEventIdx(event_names);
    return this->advance_to_event(tmax, event_idx);
}


template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
void RichSolver<Derived, T, SD, JD, OdeType, SP>::Reset(){
    Base::Reset();
    evt_col.reset(this->direction());
    current_idx = 0;
    detection_idx = -1;
    is_at_event = false;
    is_at_canon_event = false;
}

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
void RichSolver<Derived, T, SD, JD, OdeType, SP>::fill_current_vector_impl(State& out) const{
    if (is_at_canon_event && !evt_col.event(current_idx).mask_delayed()){
        // Immediate mask, ReAdjust still pending: the masked state is the current one, but
        // it has not been written into the new state yet. A delayed mask deliberately falls
        // through, since it shows the unmasked state until it has been processed.
        const MaskedState<T, State>* ms = evt_col.masked_state();
        assert(ms != nullptr && "Solver is at a canon event but has no masked state");
        copy_state<SD>(out, ms->masked_vector);
    } else {
        Base::fill_current_vector_impl(out);
    }
}

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
void RichSolver<Derived, T, SD, JD, OdeType, SP>::ReAdjust(const State& new_state){
    Base::ReAdjust(new_state);
    is_at_canon_event = false;
    is_at_event = false;
    is_event_waiting = false;
}


template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
bool RichSolver<Derived, T, SD, JD, OdeType, SP>::RequestTimeFloor(T& out) {
    // no need to call Base::RequestTimeFloor, the Base class does not request it.
    detection_idx = -1; // reset detection index at the start of a new detection round
    // do not also set is_at_event to false, as the Adv_Impl might fail and the step should remain in the same state.
    if ((is_event_waiting = evt_col.detect_all_between(
        /*
        Passing a lambda that calls this->local_interp_impl() at every evaluation,
        bacause it is cheaper than passing this->local_interp_impl() itself.
        Constructing it once can require RHS evaluations depending on the derived class,
        and if no events are detected there is no reason for that, while constructing at every
        evaluation might cost a lot only in the first call.
        */
        this->t_old(),
        this->t_new(),
        this->vector_old(),
        this->vector_new(),
        [this](State& state_out, const T& t){
            return this->local_interp_impl()(state_out, t);
        }
    ))){
        // is_event_waiting has been set to true, preparing the push_event_queue for the first event
        if (Base::RequestTimeFloor(out)){
            out = this->nearest_time(out, evt_col.get_time(0));
        } else {
            out = evt_col.get_time(0);
        }
        return true;
    } else {
        return false;
    }
}

template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
bool RichSolver<Derived, T, SD, JD, OdeType, SP>::push_event_queue(){
    if (is_event_waiting && evt_col.get_time(size_t(detection_idx+1)) == this->t()){
        current_idx = evt_col.get_event_idx(size_t(++detection_idx));
        // determine if this one is a canon event. The ReAdjust that applies the mask is
        // deferred to the next Adv_Impl for delayed and immediate masks alike: it collapses
        // the old and new states, and until it runs this step stays interpolable, which is
        // what lets dense output cover the part of it already traversed. An immediate mask
        // still *reports* the masked state right away -- fill_current_vector serves it from
        // evt_col's MaskedState, which holds it outside the solver's own buffers.
        if (const MaskedState<T, State>* ms = evt_col.masked_state()){
            is_at_canon_event = static_cast<bool>(ms->idx == current_idx);
        }
        // determine if there is another event after this one
        is_event_waiting = size_t(detection_idx) < evt_col.detection_size() - 1;
        is_at_event = true;
        return true;
    } else {
        return false;
    }
}

// PRIVATE METHODS


template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
bool RichSolver<Derived, T, SD, JD, OdeType, SP>::at_canon_event() const{
    return is_at_canon_event;
}


// ============================================================================


template<typename Derived, typename T, typename SD, typename JD, typename OdeType, SolverPolicy SP>
template<typename... Args>
bool RichSolver<Derived, T, SD, JD, OdeType, SP>::Adv_Impl(Args&&... args){
    
    if (evt_col.size() == 0){
        return Base::Adv_Impl(std::forward<Args>(args)...);
    } else if (this->at_canon_event()) {
        const MaskedState<T, State>* ms = evt_col.masked_state();
        assert(ms != nullptr && "Solver is at a canon event but has no masked state");
        // Both mask kinds land here now; see push_event_queue for why it is deferred.
        THIS->ReAdjust(ms->masked_vector);
    }
    
    if (this->is_at_new_state()){
        if (!Base::Adv_Impl(std::forward<Args>(args)...)){
            // new event detection pass was triggered in this command
            return false;
        } else if (!this->push_event_queue()){
            // is_event_waiting has been set to false in the previous Adv_Impl call, no need to set it again here
            is_at_event = false;
            is_at_canon_event = false;
        }
        return true;
    } else if (is_event_waiting){
        if (Base::Adv_Impl(evt_col.get_time(size_t(detection_idx+1)), std::forward<Args>(args)...)){
            if (!this->push_event_queue()){
                is_at_event = false;
                is_at_canon_event = false;
            }
            return true;
        } else {
            return false;
        }
    } else {
        is_at_event = false;
        is_at_canon_event = false;
        return Base::Adv_Impl(std::forward<Args>(args)...);
    }
}

} // namespace ode
