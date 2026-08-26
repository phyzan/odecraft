#ifndef ODECRAFT_ODE_INT_HPP
#define ODECRAFT_ODE_INT_HPP

#include <odecraft/OdeResult/OdeResult.hpp>
#include <odecraft/Core/BaseSolver.hpp>
#include <odecraft/Core/SolverFactory.hpp> // IWYU pragma: keep


namespace ode {

// ============================================================================
// DECLARATIONS
// ============================================================================


template<typename T, size_t N>
class EventCounter{

public:

    EventCounter(std::vector<EventOptions> options);

    DEFAULT_RULE_OF_FOUR(EventCounter)

    int operator[](size_t i) const;

    bool count_it(size_t i);

    bool is_running() const;

    bool can_fit(size_t event)const;

    size_t total()const;

private:

    std::vector<EventOptions> _options;
    std::vector<int> _counter;
    std::vector<int> _period_counter;
    size_t _total=0;
    bool _is_running = true;
};

template<typename T, size_t N=0>
class ODE{

public:

    template<hasRhsFunc<T> OdeType>
    ODE(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T> events={}, Integrator method=Integrator::RK45);

    DEFAULT_RULE_OF_FOUR(ODE)

    virtual ~ODE() = default;

    void                        Rhs(T* out, const T& t, const T* q) const;

    void                        Jac(T* out, const T& t, const T* q) const;

    void                        Jac(T* out, const T& t, const T* q, const T* dt) const;

    virtual std::unique_ptr<ODE<T, N>>  clone() const;

    size_t                      nsys() const;

    bool                        integrate_until(OdeResult<T, N>* out, T time, const std::vector<T>& t_eval={}, const std::vector<EventOptions>& event_options={}, int max_progress_reports = 0, observer_t<T> observer = nullptr);

    bool                        rich_integrate_until(OdeSolution<T, N>& out, T time, const std::vector<EventOptions>& event_options={}, int max_progress_reports = 0, observer_t<T> observer = nullptr);

    bool                        diverges() const;

    bool                        is_dead() const;

    size_t                      n_points() const;

    View1D<T>                   t() const;

    const T&                    t(size_t i) const;

    View2D<T, 0, N>             q() const;

    View1D<T, N>                q(size_t i) const;

    const OrbitData<T>&         event_data(const std::string& event) const;

    double                      runtime() const;

    const OdeRichSolver<T, N>*  solver() const;

    virtual void                clear();

    virtual void                reset();

protected:

    ODE(size_t nsys);

    pbox::owner<OdeRichSolver<T, N>> solver_;
    OrbitData<T> orbit_data_;
    EventData<T> event_data_;
    std::vector<size_t> cached_idx_;
    double runtime_ = 0;

    template<hasRhsFunc<T> OdeType>
    void                                        init(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step=0, T max_step=0, T stepsize=0, int dir=1, EventList<T> events={}, Integrator method = Integrator::RK45);

    virtual void                                register_state();

    virtual void                                register_event(size_t i);

private:

    template<typename ArrayType, typename Callable>
    bool                                        priv_integrate_until(OdeResult<T, N>* out, const T& t_max, ArrayType&& t_store, const std::vector<EventOptions>& event_options, Callable&& observer, int max_progress_reports, bool interpolate);

};

} // namespace ode

#endif // ODECRAFT_ODE_INT_HPP
