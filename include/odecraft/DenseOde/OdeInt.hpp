#pragma once

#include "odecraft/Toolkit/Tools.hpp"
#include <odecraft/OdeResult/OdeResult.hpp>
#include <odecraft/Core/BaseSolver/BaseSolver.hpp>
#include <odecraft/Core/SolverFactory.hpp> // IWYU pragma: keep


namespace ode {

// ============================================================================
// DECLARATIONS
// ============================================================================


template<typename T, typename SD>
class EventCounter{

public:

    EventCounter(std::vector<EventOptions> options);

    DEFAULT_RULE_OF_FOUR(EventCounter)

    int operator[](size_t i) const;

    bool count_it(size_t i);

    bool is_running() const;

    bool can_fit(size_t event) const;

    size_t total() const;

private:

    std::vector<EventOptions> options_;
    std::vector<int> counter_;
    std::vector<int> period_counter_;
    size_t total_ = 0;
    bool is_running_ = true;
};

template<typename T, typename SD>
class ODE{

public:
    using State = state_t<T, SD>;
    using RealState = real_state_t<T, SD>;
    using JacMat = jacmat_t<T, SD>;

    template<hasRhsFunc<T, State> OdeType>
    ODE(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T, State> events={}, Stepper method=Stepper::RK45);

    DEFAULT_RULE_OF_FOUR(ODE)

    virtual ~ODE() = default;

    void                        Rhs(State& out, const T& t, const State& q) const;

    void                        Jac(JacMat& out, const T& t, const State& q) const;

    void                        Jac(JacMat& out, const T& t, const State& q, const State& dq) const;

    virtual std::unique_ptr<ODE<T, SD>>  clone() const;

    bool                        integrate_until(
        OdeResult<T, SD>* out,
        T t_max,
        const std::vector<EventOptions>& event_options={},
        const observer_t<T, State>& observer=nullptr
    );

    bool                        integrate_until(
        OdeResult<T, SD>* out,
        T t_max,
        const std::vector<EventOptions>& event_options,
        const observer_t<T, State>& observer,
        const std::vector<T>& t_eval
    );

    bool                        integrate(
        OdeResult<T, SD>* out,
        T interval,
        const std::vector<EventOptions>& event_options={},
        const observer_t<T, State>& observer=nullptr
    );

    bool                        integrate(
        OdeResult<T, SD>* out,
        T interval,
        const std::vector<EventOptions>& event_options,
        const observer_t<T, State>& observer,
        const std::vector<T>& t_eval
    );

    bool                        rich_integrate_until(
        OdeSolution<T, SD>& out,
        T time,
        const std::vector<EventOptions>& event_options={},
        const observer_t<T, State>& observer = nullptr
    );

    bool                        rich_integrate(
        OdeSolution<T, SD>& out,
        T interval,
        const std::vector<EventOptions>& event_options={},
        const observer_t<T, State>& observer = nullptr
    );

    bool                        diverges() const;

    bool                        is_dead() const;

    size_t                      n_points() const;

    const std::vector<T>&       t() const;

    const std::vector<State>&   q() const;

    const OrbitData<T, SD>&     event_data(const std::string& event) const;

    double                      runtime() const;

    const OdeRichSolver<T, SD>* solver() const;

    virtual void                clear();

    virtual void                reset();

protected:

    ODE() = default;

    pbox::owner<OdeRichSolver<T, SD>> solver_;
    OrbitData<T, SD> orbit_data_;
    EventData<T, SD> event_data_;
    std::vector<size_t> cached_idx_;
    double runtime_ = 0;

    template<hasRhsFunc<T, State> OdeType>
    void    init(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T, State> events={}, Stepper method = Stepper::RK45);

    virtual void    register_state();

    virtual void    register_event(size_t i);

private:

    template<typename ArrayType, typename Callable>
    bool            priv_integrate_until(OdeResult<T, SD>* out, const T& t_max, ArrayType&& t_store, const std::vector<EventOptions>& event_options, Callable&& observer, bool interpolate);

};

} // namespace ode
