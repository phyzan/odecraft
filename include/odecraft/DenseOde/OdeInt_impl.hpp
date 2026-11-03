#pragma once

#include <algorithm>
#include <ostream>

#include <odecraft/Core/VirtualBase.hpp>
#include <odecraft/Events/Events.hpp>
#include <odecraft/DenseOde/OdeInt.hpp>
#include "OdeInt_mem_impl.hpp" // IWYU pragma: keep
#include "odecraft/Toolkit/Tools.hpp"



namespace ode{


// EventCounter implementations
template<typename T, typename SD>
EventCounter<T, SD>::EventCounter(std::vector<EventOptions> options) : options_(std::move(options)), counter_(options_.size(), 0), period_counter_(options_.size(), 0) {
    for (const auto & option : options_){
        if (option.period < 1){
            throw std::runtime_error("The period argument in event options must be at least 1.");
        }
    }
}

template<typename T, typename SD>
int EventCounter<T, SD>::operator[](size_t i) const{
    return counter_[i];
}

template<typename T, typename SD>
bool EventCounter<T, SD>::count_it(size_t i){
    if (this->can_fit(i)){
        period_counter_[i]++;
        if (period_counter_[i] == options_[i].period){
            period_counter_[i] = 0;
            counter_[i]++;
            total_++;
            if ((counter_[i] == options_[i].max_events) && options_[i].terminate){
                is_running_ = false;
            }
            return true;
        }
    }
    return false;
}

template<typename T, typename SD>
bool EventCounter<T, SD>::is_running()const{
    return is_running_;
}

template<typename T, typename SD>
bool EventCounter<T, SD>::can_fit(size_t event)const{
    return (counter_[event] != options_[event].max_events) && is_running_;
}

template<typename T, typename SD>
size_t EventCounter<T, SD>::total()const{
    return total_;
}

template<typename T, typename SD>
void ODE<T, SD>::Rhs(State& out, const T& t, const State& q) const{
    solver_->get_rhs(out, t, q);
}

template<typename T, typename SD>
void ODE<T, SD>::Jac(JacMat& out, const T& t, const State& q) const{
    solver_->get_jac(out, t, q);
}

template<typename T, typename SD>
void ODE<T, SD>::Jac(JacMat& out, const T& t, const State& q, const State& dq) const{
    solver_->get_jac(out, t, q, dq);
}

template<typename T, typename SD>
std::unique_ptr<ODE<T, SD>> ODE<T, SD>::clone() const{
    return std::make_unique<ODE<T, SD>>(*this);
}


template<typename T, typename SD>
bool ODE<T, SD>::integrate_until(
    OdeResult<T, SD>* out,
    T t_max,
    const std::vector<EventOptions>& event_options,
    const observer_t<T, State>& observer){

    if (observer == nullptr){
        return this->priv_integrate_until(out, t_max, nullptr, event_options, nullptr, false);
    } else {
        return this->priv_integrate_until(out, t_max, nullptr, event_options, observer, false);
    }
}

template<typename T, typename SD>
bool ODE<T, SD>::integrate_until(
    OdeResult<T, SD>* out,
    T t_max,
    const std::vector<EventOptions>& event_options,
    const observer_t<T, State>& observer,
    const std::vector<T>& t_eval){

    if (observer == nullptr){
        return this->priv_integrate_until(out, t_max, t_eval, event_options, nullptr, false);
    } else {
        return this->priv_integrate_until(out, t_max, t_eval, event_options, observer, false);
    }
}

template<typename T, typename SD>
bool ODE<T, SD>::integrate(
    OdeResult<T, SD>* out,
    T interval,
    const std::vector<EventOptions>& event_options,
    const observer_t<T, State>& observer){

    if (interval < 0){
        throw std::runtime_error("Integration interval must be non-negative");
    }

    T t_max = interval + solver_->get_time()*solver_->get_direction();

    return this->integrate_until(
        out,
        t_max,
        event_options,
        observer
    );
}

template<typename T, typename SD>
bool ODE<T, SD>::integrate(
    OdeResult<T, SD>* out,
    T interval,
    const std::vector<EventOptions>& event_options,
    const observer_t<T, State>& observer,
    const std::vector<T>& t_eval){

    if (interval < 0){
        throw std::runtime_error("Integration interval must be non-negative");
    }

    T t_max = interval + solver_->get_time()*solver_->get_direction();

    return this->integrate_until(
        out,
        t_max,
        event_options,
        observer,
        t_eval
    );
}

template<typename T, typename SD>
bool ODE<T, SD>::rich_integrate_until(
    OdeSolution<T, SD>& out,
    T time,
    const std::vector<EventOptions>& event_options,
    const observer_t<T, State>& observer){

    if (observer == nullptr){
        return this->priv_integrate_until(&out, time, nullptr, event_options, nullptr, true);
    } else {
        return this->priv_integrate_until(&out, time, nullptr, event_options, observer, true);
    }
}

template<typename T, typename SD>
bool ODE<T, SD>::rich_integrate(
    OdeSolution<T, SD>& out,
    T interval,
    const std::vector<EventOptions>& event_options,
    const observer_t<T, State>& observer){

    if (interval < 0){
        throw std::runtime_error("Integration interval must be non-negative");
    }
    T t_max = interval + solver_->get_time()*solver_->get_direction();
    return this->rich_integrate_until(
        out,
        t_max,
        event_options,
        observer
    );
}

template<typename T, typename SD>
bool ODE<T, SD>::diverges() const{
    return solver_->get_status() == StepperStatus::Diverges;
}

template<typename T, typename SD>
bool ODE<T, SD>::is_dead() const{
    return !solver_->get_is_running();
}

template<typename T, typename SD>
size_t ODE<T, SD>::n_points() const{
    return orbit_data_.t.size();
}

template<typename T, typename SD>
const std::vector<T>& ODE<T, SD>::t()const{
    return orbit_data_.t;
}

template<typename T, typename SD>
const std::vector<state_t<T, SD>>& ODE<T, SD>::q()const{
    return orbit_data_.q;
}

template<typename T, typename SD>
const OrbitData<T, SD>& ODE<T, SD>::event_data(const std::string& event) const{
    return event_data_.data(event);
}

template<typename T, typename SD>
double ODE<T, SD>::runtime()const{
    return runtime_;
}

template<typename T, typename SD>
const OdeRichSolver<T, SD>* ODE<T, SD>::solver()const{
    return solver_.get_raw_pointer();
}

template<typename T, typename SD>
void ODE<T, SD>::clear(){
    orbit_data_.clear_all();
    event_data_.clear_points();
    std::ranges::fill(cached_idx_, 0);
    register_state();
}

template<typename T, typename SD>
void ODE<T, SD>::reset(){
    runtime_ = 0;
    solver_->do_reset();
    this->clear();
}

template<typename T, typename SD>
void ODE<T, SD>::register_state(){
    orbit_data_.add_point(solver_->get_time(), solver_->get_new_vector());
    // add_point appended nsys values, so the block starts nsys from the end.
    State& q_last = orbit_data_.q.back();
    solver_->get_current_vector(q_last);
}

template<typename T, typename SD>
void ODE<T, SD>::register_event(size_t i){
    event_data_.add_event(i, solver_->get_time(), solver_->get_new_vector());
    OrbitData<T, SD>& od = event_data_.data(i);
    State& q_last = od.q.back();
    solver_->get_current_vector(q_last);
}


template<typename T, typename SD>
template<typename ArrayType, typename Callable>
bool ODE<T, SD>::priv_integrate_until(OdeResult<T, SD>* out, const T& t_max, ArrayType&& t_store, const std::vector<EventOptions>& event_options, Callable&& observer, bool interpolate){

    static_assert(isObserver<Callable, T, State> || std::is_same_v<std::decay_t<Callable>, std::nullptr_t>, "Callable must be an observer or nullptr");
    static_assert(isArray<ArrayType, T> || std::is_same_v<std::decay_t<ArrayType>, std::nullptr_t>, "ArrayLike must be an array or nullptr");

    /*
    
    **For private use** because its interface might be confusing
    So simple overloaded versions are provided

    ArrayType being an empty array and being nullptr are *completely* different.
    If t_store is an array, only its elements are stored
    So if it is empty, no steps are stored
    However if nullptr, **ALL** steps are stored
    */

    auto emit = [&](OdeResult<T, SD>&& res, BoxedInterp<T, SD>&& dense){
        if (interpolate){
            auto* rich_res = dynamic_cast<OdeSolution<T, SD>*>(out);
            assert(rich_res && "Output must be of type OdeSolution when interpolation is enabled");
            *rich_res = OdeSolution<T, SD>(std::move(res), std::move(dense));
        } else {
            *out = std::move(res);
        }
    };
    
    if (! solver_->get_is_running()){
        if (out){
            emit(OdeResult<T, SD>{
                {},
                {},
                0,
                solver_->get_status(),
                0,
                solver_->get_message(),
            }, BoxedInterp<T, SD>());
        }
        return false;
    } else if (t_max*solver_->get_direction() < solver_->get_time()*solver_->get_direction()){
        if (out){
            emit(OdeResult<T, SD>({}, {}, 0, StepperStatus::BadICS, 0, "Cannot integrate in opposite direction"), BoxedInterp<T, SD>());
        }
        return false; //cannot integrate in opposite direction
    }

    constexpr bool store_explicit_steps = !std::is_same_v<std::decay_t<ArrayType>, std::nullptr_t>;
    const T t0 = solver_->get_time();
    const bool      first_eval_t0 = [&](){
        if constexpr (store_explicit_steps){
            return (t_store.size() > 0 && t_store[0] == t0);
        } else {
            return false;
        }
    }();
    const bool      include_first = (!store_explicit_steps || first_eval_t0);
    const size_t    t_start_idx = orbit_data_.t.size() - include_first;
    const char*     terminate_message = nullptr;
    bool success;
    BoxedInterp<T, SD> interpolator;
    double duration = timeit_ms([&](){
        if constexpr (store_explicit_steps){
            assert(interpolate == false && "Explicit step storage is enabled only for non-interpolating integration");
        }
        // ------------------------------ IMPLEMENTATION --------------------------------------
        
        std::span<const T>    t_eval = [&](){
            if constexpr (store_explicit_steps){
                return std::span<const T>{t_store.data() + first_eval_t0, t_store.size() - first_eval_t0};
            } else {
                return std::span<const T>{nullptr, size_t{0}};
            }
        }();
        
        for (size_t i=0; i < event_data_.size(); i++){
            cached_idx_[i] = event_data_.data(i).size();
        }

        //check that all names in max_events are valid
        EventCounter<T, SD> event_counter = this->solver()->get_event_col().validate_events(event_options);

        auto event_state_valid = [&]()NDSPAN_LAMBDA_INLINE{
            bool res = false;
            if (const EventState<T, State> es = solver_->get_current_event()){
                if (event_counter.count_it(es.idx)){
                    res = true;
                    register_event(es.idx);
                }
            }
            return res;
        };

        // Since we pass an array of t_eval in the solver later, if step_ptr is not null,
        // it is guaranteed to point to an element in t_eval.
        auto main_observer = [&](const T& t, const State& q, const T* step_ptr) NDSPAN_LAMBDA_INLINE -> bool {
            const bool at_valid_event = solver_->get_at_event() && event_state_valid();

            if constexpr (!store_explicit_steps) {
                // step_ptr is true only at the last step
                register_state();
            } else if (step_ptr) {
                register_state();
            }

            if (at_valid_event && !event_counter.is_running()) {
                terminate_message = "Max events reached";
                return false;
            }

            if constexpr (isObserver<Callable, T, State>){
                return observer(t, q, step_ptr);
            } else {
                return true;
            }
        };


        if constexpr (store_explicit_steps){
            success = solver_->do_advance_until(t_max, main_observer, t_eval);
        } else if (interpolate){
            interpolator = solver_->do_interpolate_until(t_max, main_observer);
            success = (t0 != t_max) && (solver_->get_time() == t_max);
        } else {
            success = solver_->do_advance_until(t_max, main_observer);
        }

        if (success) {
            terminate_message = "t-goal";
        } else if (!terminate_message){
            terminate_message = solver_->get_message();
        }
    });
    
    runtime_ += duration;
    if (out){
        EventData<T, SD>    event_res(this->event_data_, cached_idx_);
        OdeResult<T, SD> res{
            orbit_data_,
            event_res,
            t_start_idx,
            solver_->get_status(),
            duration,
            terminate_message,
        };
        emit(std::move(res), std::move(interpolator));
    } else {
        assert (interpolate == false && "Output must be provided to return interpolation result");
    }
    return success;
}

template<typename T, typename SD>
template<hasRhsFunc<T, state_t<T, SD>> OdeType>
void ODE<T, SD>::init(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step, T max_step, T stepsize, int dir, EventList<T, State> events, Stepper method){
    solver_ = make_solver<UtilPolicy::RichVirtual>(method, std::move(ode), std::move(t0), std::move(q0), std::move(rtol), std::move(atol), std::move(min_step), std::move(max_step), std::move(stepsize), dir, std::move(events));
    const EventCollection<T, State>& event_coll = this->solver_->get_event_col();
    cached_idx_.resize(event_coll.size(), 0);
    register_state();
    for (size_t i=0; i<event_coll.size(); i++){
        event_data_.allocate_event(event_coll.event(i).name());
    }
}

} // namespace ode
