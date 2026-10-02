#ifndef ODECRAFT_RICH_BASE_IMPL_HPP
#define ODECRAFT_RICH_BASE_IMPL_HPP

#include "RichBase.hpp"

namespace ode{

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const EventCollection<T>& RichSolver<Derived, T, N, SP, OdeType>::event_col() const{
    return evt_col;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::at_event(int event_idx) const{
    if (event_idx == -1){
        return is_at_event;
    } else if (EventState<T> es = this->current_event()){
        return es.idx == size_t(event_idx);
    } else {
        return false;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
int RichSolver<Derived, T, N, SP, OdeType>::event_idx(const std::string& name) const{
    return evt_col.event_idx(name);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
std::vector<size_t> RichSolver<Derived, T, N, SP, OdeType>::toEventIdx(const std::vector<std::string>& event_names) const{
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

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void RichSolver<Derived, T, N, SP, OdeType>::show_state(int prec) const{
    decltype(auto) vector = this->scratch_vector();
    this->fill_current_vector(vector.data());
    SolverRichState<T, N>(vector.data(), this->t(), this->stepsize(), this->nsys(), this->step_count(), this->message(), this->current_event().event ? this->current_event().event->name() : "").show(prec);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
RichSolver<Derived, T, N, SP, OdeType>::RichSolver(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, EventList<T> evs) : Base(ode, t0, q0, rtol, atol, min_step, max_step, stepsize, dir), evt_col(std::move(evs)) {
    this->evt_col.setup(t0, this->nsys(), this->direction());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::advance_to_event(const std::vector<size_t>& event_idx){
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
        } else if (EventState<T> es = this->current_event()){
            if (event_idx.empty() || std::find(event_idx.begin(), event_idx.end(), es.idx) != event_idx.end()){
                return true;
            }
        }
    }while (true);
    return true;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::advance_to_event(const T& tmax, const std::vector<size_t>& event_idx){
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
        [&](const T& /*t*/, const T* /*state*/, const T* /*extra*/) NDSPAN_LAMBDA_INLINE {
            if (EventState<T> es = this->current_event()){
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

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::advance_to_event(const std::vector<std::string>& event_names){
    std::vector<size_t> event_idx = this->toEventIdx(event_names);
    return this->advance_to_event(event_idx);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::advance_to_event(const T& tmax, const std::vector<std::string>& event_names){
    std::vector<size_t> event_idx = this->toEventIdx(event_names);
    return this->advance_to_event(tmax, event_idx);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void RichSolver<Derived, T, N, SP, OdeType>::Reset(){
    Base::Reset();
    evt_col.reset(this->direction());
    current_idx = 0;
    detection_idx = -1;
    is_at_event = false;
    is_at_canon_event = false;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void RichSolver<Derived, T, N, SP, OdeType>::fill_current_vector_impl(T* out) const{
    if (is_at_canon_event && !evt_col.event(current_idx).mask_delayed()){
        // Immediate mask, ReAdjust still pending: the masked state is the current one, but
        // it has not been written into new_state_ yet. A delayed mask deliberately falls
        // through, since it shows the unmasked state until it has been processed.
        const MaskedState<T>* ms = evt_col.masked_state();
        assert(ms != nullptr && "Solver is at a canon event but has no masked state");
        std::copy(ms->masked_vector.data(), ms->masked_vector.data() + this->nsys(), out);
    } else {
        Base::fill_current_vector_impl(out);
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void RichSolver<Derived, T, N, SP, OdeType>::ReAdjust(const T* new_vector){
    Base::ReAdjust(new_vector);
    is_at_canon_event = false;
    is_at_event = false;
    is_event_waiting = false;
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::RequestTimeFloor(T& out) {
    // no need to call Base::RequestTimeFloor, the Base class does not request it.
    detection_idx = -1; // reset detection index at the start of a new detection round
    // do not also set is_at_event to false, as the Adv_Impl might fail and the step should remain in the same state.
    if ((is_event_waiting = evt_col.detect_all_between(
        this->old_state(),
        this->new_state(),
        [this](T* q_out, const T& t){
            this->interp(q_out, t);
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

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::push_event_queue(){
    if (is_event_waiting && evt_col.get_time(size_t(detection_idx+1)) == this->t()){
        current_idx = evt_col.get_event_idx(size_t(++detection_idx));
        // determine if this one is a canon event. The ReAdjust that applies the mask is
        // deferred to the next Adv_Impl for delayed and immediate masks alike: it collapses
        // old_state_/new_state_, and until it runs this step stays interpolable, which is
        // what lets dense output cover the part of it already traversed. An immediate mask
        // still *reports* the masked state right away -- fill_current_vector serves it from
        // evt_col's MaskedState, which holds it outside the solver's own buffers.
        if (const MaskedState<T>* ms = evt_col.masked_state()){
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


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool RichSolver<Derived, T, N, SP, OdeType>::at_canon_event() const{
    return is_at_canon_event;
}


// ============================================================================


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename... Args>
bool RichSolver<Derived, T, N, SP, OdeType>::Adv_Impl(Args&&... args){
    
    if (evt_col.size() == 0){
        return Base::Adv_Impl(std::forward<Args>(args)...);
    } else if (this->at_canon_event()) {
        const MaskedState<T>* ms = evt_col.masked_state();
        assert(ms != nullptr && "Solver is at a canon event but has no masked state");
        // Both mask kinds land here now; see push_event_queue for why it is deferred.
        ODECRAFT_CALL_DERIVED(ReAdjust, ms->masked_vector.data());
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


#endif // ODECRAFT_RICH_BASE_IMPL_HPP