#pragma once

#include "odecraft/Core/SolverFactory.hpp"
#include <odecraft/Toolkit/Tools.hpp>
#include <odecraft/Interpolation/Univariate/StateInterp.hpp>

namespace ode{


//===========================================================================================
//                                      OdeResult<T, SD>
//===========================================================================================

//ODERESULT STRUCT TO ENCAPSULATE THE RESULT OF AN ODE INTEGRATION

template<typename T, typename SD>
struct OrbitData{
    using State = state_t<T, SD>;
    std::vector<T> t;
    std::vector<State> q;

    size_t size() const;
    void clear_all();
    void add_point(T time, State state);
};


template<typename T, typename SD>
class EventData{

public:
    using State = state_t<T, SD>;

    EventData() = default;

    EventData(const EventData& other, const std::vector<size_t>& start_indices);

    DEFAULT_RULE_OF_FOUR(EventData)

    size_t size() const;

    const OrbitData<T, SD>& data(size_t i) const;

    const OrbitData<T, SD>& data(const std::string& name) const;

    const std::string& name(size_t i) const;

    OrbitData<T, SD>& data(size_t i);

    void allocate_event(std::string name);

    void add_event(size_t event_idx, T t, State q);

    void add_event(const std::string& name, T t, State q);

    void clear_points();

private:

    std::vector<OrbitData<T, SD>> event_data_;
    std::vector<std::string> event_names_;
};


template<typename T, typename SD>
class OdeResult{

public:
    using State = state_t<T, SD>;

    OdeResult(const OrbitData<T, SD>& orbit_data, EventData<T, SD> event_data, size_t orb_idx_start, StepperStatus status, double runtime, std::string message);

    OdeResult() = default;

    DEFAULT_RULE_OF_FOUR(OdeResult)

    virtual ~OdeResult() = default;

    const std::vector<T>& t() const;

    const std::vector<State>& q() const;

    const EventData<T, SD>& event_data() const;

    bool success() const;

    StepperStatus status() const;

    double runtime() const;

    const std::string& message() const;

    void examine() const;

    std::string event_log() const;

    virtual std::unique_ptr<OdeResult<T, SD>> clone() const;

private:

    OrbitData<T, SD> orbit_data_;
    EventData<T, SD> event_data_;
    std::string message_ = "No integration performed";
    StepperStatus status_ = StepperStatus::BadICS;
    double runtime_ = 0;

};


// ============================================================================================
//                                      OdeSolution<T, SD>
// ============================================================================================


template<typename T, typename SD>
class OdeSolution : public OdeResult<T, SD>{

public:
    using State = state_t<T, SD>;
    OdeSolution() = default;

    OdeSolution(OrbitData<T, SD> orbit_data, EventData<T, SD> event_data, size_t orb_idx_start, StepperStatus status, double runtime, std::string message, const interp::uni::Interpolator<T, SD>& interpolator);

    template<typename Interp>
    OdeSolution(OdeResult<T, SD>&& other, Interp&& interpolator) : OdeResult<T, SD>(std::move(other)), interpolator_(std::forward<Interp>(interpolator)) {}

    void operator()(State& out, const T& t) const;

    std::unique_ptr<OdeResult<T, SD>> clone() const override;

private:

    interp::uni::InterpObj<T, SD> interpolator_;

};

} // namespace ode
