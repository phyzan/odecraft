#pragma once

#include "odecraft/Core/SolverFactory.hpp"
#include "OdeResult.hpp"
#include <sstream>

namespace ode{

//===========================================================================================
//                                      OrbitData<T>
//===========================================================================================

template<typename T, typename SD>
size_t OrbitData<T, SD>::size() const {
    return t.size();
}

template<typename T, typename SD>
void OrbitData<T, SD>::clear_all() {
    t = std::vector<T>();
    q = std::vector<State>();
}

template<typename T, typename SD>
void OrbitData<T, SD>::add_point(T time, State state) {
    t.push_back(std::move(time));
    try {
        q.push_back(std::move(state));
    } catch (...) {
        t.pop_back();  // rollback
        throw;
    }
}


//===========================================================================================
//                                      EventData<T>
//===========================================================================================

template<typename T, typename SD>
EventData<T, SD>::EventData(const EventData& other, const std::vector<size_t>& start_indices) : event_names_(other.event_names_){
    assert(start_indices.size() == other.size() && "Start indices size must match the number of events");
    for (size_t i = 0; i < other.size(); ++i) {
        size_t start_idx = start_indices[i];
        assert(start_idx <= other.data(i).t.size() && "Start index must be within the bounds of the provided data");
        OrbitData<T, SD> data;
        const OrbitData<T, SD>& other_data = other.data(i);
        data.t = std::vector<T>(other_data.t.begin() + start_idx, other_data.t.end());
        data.q = std::vector<State>(other_data.q.begin() + start_idx, other_data.q.end());
        event_data_.push_back(std::move(data));
    }
}

template<typename T, typename SD>
size_t EventData<T, SD>::size() const {
    return event_data_.size();
}


template<typename T, typename SD>
const OrbitData<T, SD>& EventData<T, SD>::data(size_t i) const {
    assert(i < event_data_.size() && "Event index out of range");
    return event_data_[i];
}

template<typename T, typename SD>
const OrbitData<T, SD>& EventData<T, SD>::data(const std::string& name) const {
    auto it = std::ranges::find(event_names_, name);
    if (it == event_names_.end()){
        throw std::runtime_error("Event name not found: " + name);
    }
    size_t index = std::distance(event_names_.begin(), it);
    return this->data(index);
}

template<typename T, typename SD>
const std::string& EventData<T, SD>::name(size_t i) const {
    assert(i < event_names_.size() && "Event index out of range");
    return event_names_[i];
}

template<typename T, typename SD>
OrbitData<T, SD>& EventData<T, SD>::data(size_t i) {
    assert(i < event_data_.size() && "Event index out of range");
    return event_data_[i];
}

template<typename T, typename SD>
void EventData<T, SD>::allocate_event(std::string name) {
    if (name.empty()){
        throw std::runtime_error("Event name cannot be empty");
    }
    if (std::ranges::find(event_names_, name) != event_names_.end()){
        throw std::runtime_error("Event name already exists");
    }
    event_data_.emplace_back(OrbitData<T, SD>{.t = {}, .q = {}});

    try {
        event_names_.push_back(std::move(name));
    } catch (...) {
        event_data_.pop_back();  // rollback
        throw;
    }
}

template<typename T, typename SD>
void EventData<T, SD>::add_event(size_t event_idx, T t, State q) {
    if (event_idx >= event_names_.size()){
        throw std::runtime_error("Event index out of range");
    }
    OrbitData<T, SD>& data = event_data_[event_idx];
    // TODO Make a more general "try_append" otherwise will need to do `try` and `catch` everywhere
    data.t.push_back(std::move(t));
    data.q.push_back(std::move(q));
}

template<typename T, typename SD>
void EventData<T, SD>::add_event(const std::string& name, T t, State q) {
    auto it = std::ranges::find(event_names_, name);
    if (it == event_names_.end()){
        throw std::runtime_error("Event name not found: " + name);
    }
    size_t index = std::distance(event_names_.begin(), it);
    this->add_event(index, std::move(t), std::move(q));
}

template<typename T, typename SD>
void EventData<T, SD>::clear_points() {
    for (auto& data : event_data_){
        data = OrbitData<T, SD>{.t = {}, .q = {}};
    }
}


//===========================================================================================
//                                      OdeResult<T, SD>
//===========================================================================================

template<typename T, typename SD>
OdeResult<T, SD>::OdeResult(const OrbitData<T, SD>& orbit_data, EventData<T, SD> event_data, size_t orb_idx_start, StepperStatus status, double runtime, std::string message) : event_data_(std::move(event_data)), message_(std::move(message)), status_(status), runtime_(runtime) {
    assert(orb_idx_start <= orbit_data.t.size() && "Start index must be within the bounds of the provided data");
    // Copy only the relevant portion of the data based on the provided start indices
    orbit_data_.t = std::vector<T>(orbit_data.t.begin() + orb_idx_start, orbit_data.t.end());
    orbit_data_.q = std::vector<State>(orbit_data.q.begin() + orb_idx_start, orbit_data.q.end());
}

template<typename T, typename SD>
std::unique_ptr<OdeResult<T, SD>> OdeResult<T, SD>::clone() const {
    return std::make_unique<OdeResult<T, SD>>(*this);
}

template<typename T, typename SD>
const std::vector<T>& OdeResult<T, SD>::t() const {
    return this->orbit_data_.t;
}

template<typename T, typename SD>
const std::vector<state_t<T, SD>>& OdeResult<T, SD>::q() const {
    return orbit_data_.q;
}


template<typename T, typename SD>
const EventData<T, SD>& OdeResult<T, SD>::event_data() const{
    return event_data_;
}

template<typename T, typename SD>
bool OdeResult<T, SD>::success() const {
    return status_ == StepperStatus::NewStep || status_ == StepperStatus::BetweenSteps;
}

template<typename T, typename SD>
StepperStatus OdeResult<T, SD>::status() const {
    return status_;
}

template<typename T, typename SD>
double OdeResult<T, SD>::runtime() const {
    return runtime_;
}

template<typename T, typename SD>
const std::string& OdeResult<T, SD>::message() const {
    return message_;
}

template<typename T, typename SD>
void OdeResult<T, SD>::examine() const {
    std::cout << std::endl << "\n=============== OdeResult ===============\n"
              << "\tPoints           : " << orbit_data_.t.size() << "\n"
              << "\tSuccess          : " << (success() ? "true" : "false") << "\n"
              << "\tRuntime          : " << runtime_ << "\n"
              << "\tTermination cause: " << message_ << "\n"
              << event_log();
}

template<typename T, typename SD>
std::string OdeResult<T, SD>::event_log() const {
    std::ostringstream oss;
    oss << "\tEvents:\n\t----------\n";
    for (size_t i=0; i<event_data_.size(); i++){
        const OrbitData<T, SD>& data = event_data_.data(i);
        const std::string& name = event_data_.name(i);
        oss << "\t    " << name << " : " << data.t.size() << "\n";
    }
    oss << "\n\t----------\n";
    return oss.str();
}


// ==================================================================================
// ================================ OdeSolution =====================================
// ==================================================================================


template<typename T, typename SD>
OdeSolution<T, SD>::OdeSolution(OrbitData<T, SD> orbit_data, EventData<T, SD> event_data, size_t orb_idx_start, StepperStatus status, double runtime, std::string message, const interp::uni::Interpolator<T, SD>& interpolator) : OdeResult<T, SD>(std::move(orbit_data), std::move(event_data), orb_idx_start, status, runtime, std::move(message)), interpolator_(interpolator.clone()) {}


template<typename T, typename SD>
void OdeSolution<T, SD>::operator()(State& out, const T& t) const{
    if (this->t().size() == 0){
        // Only a solution whose integration never started has no points, and no interpolator.
        throw std::runtime_error("OdeSolution has no dense output: " + this->message());
    }
    return interpolator_->call(out, t);
}

template<typename T, typename SD>
std::unique_ptr<OdeResult<T, SD>> OdeSolution<T, SD>::clone() const{ return std::make_unique<OdeSolution<T, SD>>(*this);}

} // namespace ode
