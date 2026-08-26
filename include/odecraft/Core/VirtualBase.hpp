#ifndef ODECRAFT_VIRTUAL_BASE_HPP
#define ODECRAFT_VIRTUAL_BASE_HPP

#include <functional>
#include <odecraft/Interpolation/Univariate/StateInterp.hpp>
#include <odecraft/Core/Events.hpp>
#include <polybox/polybox.hpp>
#include <odecraft/Core/SolverFactory.hpp>

namespace ode {

using namespace interp::uni;


template<typename T, size_t N>
using BoxedInterp = pbox::Box<Interpolator<T, N>>;

template<typename T, size_t N=0>
class OdeSolver{

public:

    virtual ~OdeSolver() = default;

    // ODE PROPERTIES
    virtual void                get_rhs(T* out, const T& t, const T* q) const = 0;
    virtual void                get_jac(T* out, const T& t, const T* q) const = 0;
    virtual void                get_jac(T* out, const T& t, const T* q, const T* dt) const = 0;

    // ACCESSORS
    virtual const T&            get_time() const = 0;
    virtual const T&            get_new_time() const = 0;
    virtual const T&            get_old_time() const = 0;
    virtual View1D<T, N>        get_vector() const = 0;
    virtual View1D<T, N>        get_new_vector() const = 0;
    virtual View1D<T, N>        get_old_vector() const = 0;
    virtual State<T>            get_ics() const = 0;
    virtual State<T>            get_new_state() const = 0;
    virtual State<T>            get_old_state() const = 0;
    virtual const T&            get_stepsize() const = 0;
    virtual int                 get_direction() const = 0;
    virtual const T&            get_rtol() const = 0;
    virtual const T&            get_atol() const = 0;
    virtual const T&            get_min_step() const = 0;
    virtual const T&            get_max_step() const = 0;
    virtual size_t              get_nsys() const = 0;
    virtual size_t              get_step_count() const = 0;
    virtual bool                get_is_running() const = 0;
    virtual bool                get_is_dead() const = 0;
    virtual bool                get_diverges() const = 0;
    virtual const std::string&  get_status() const = 0;
    virtual bool                get_validate_ics(T t0, const T* q0) const = 0;
    virtual bool                get_has_valid_ics() const = 0;
    virtual Stepper          get_method() const = 0;
    virtual void                get_interp(T* result, const T& t) const = 0;
    virtual size_t              get_rhs_eval_count() const = 0;
    virtual size_t              get_jac_eval_count() const = 0;
    virtual BoxedInterp<T, N>   get_state_interpolator(int bdr1, int bdr2) const = 0;
    virtual T                   get_auto_step(T t, const T* q) const = 0;
    virtual T                   get_auto_step() const = 0;
    virtual void                show_state(int prec=8) const = 0;
    virtual std::unique_ptr<OdeSolver<T, N>> clone() const = 0;

    // MODIFIERS
    virtual bool                do_advance() = 0;
    virtual bool                do_advance_by(T interval) = 0;

    virtual bool                do_advance_until(T time) = 0;
    virtual bool                do_advance_until(T time, observer_t<T> observer) = 0;
    virtual bool                do_advance_until(T time, observer_t<T> observer, View1D<T> checkpoints) = 0;

    virtual BoxedInterp<T, N>   do_interpolate_until(T time) = 0;
    virtual BoxedInterp<T, N>   do_interpolate_until(T time, observer_t<T> observer) = 0;

    virtual void                do_reset() = 0;
    virtual void                do_kill(std::string message = "") = 0;
    virtual bool                do_set_ics(T t0, const T* y0, T stepsize = 0, int direction = 0) = 0;

protected:

    OdeSolver() = default;
    OdeSolver(const OdeSolver&) = default;
    OdeSolver(OdeSolver&&) noexcept = default;
    OdeSolver& operator=(const OdeSolver&) = default;
    OdeSolver& operator=(OdeSolver&&) noexcept = default;

};


template<typename T, size_t N=0>
class OdeRichSolver : public OdeSolver<T, N>{

public:

    // ACCESSORS
    virtual const EventCollection<T>&       get_event_col() const = 0;
    virtual int                             get_event_idx(const std::string& name) const = 0;
    virtual bool                            get_at_event(int event_idx = -1) const = 0;
    virtual EventState<T>                   get_current_event() const = 0;
    virtual bool                            get_at_canon_event() const = 0;
    // MODIFIERS
    virtual bool                            do_advance_to_event(const std::vector<size_t>& event_idx = {}) = 0;
    virtual bool                            do_advance_to_event(const T& tmax, const std::vector<size_t>& event_idx = {}) = 0;
    virtual bool                            do_advance_to_event(const std::vector<std::string>& event_names) = 0;
    virtual bool                            do_advance_to_event(const T& tmax, const std::vector<std::string>& event_names) = 0;

protected:

    OdeRichSolver() = default;

    DEFAULT_RULE_OF_FOUR(OdeRichSolver)
};


} // namespace ode

#endif // ODECRAFT_VIRTUAL_BASE_HPP