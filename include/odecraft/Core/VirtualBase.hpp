#pragma once

#include <odecraft/Interpolation/Univariate/StateInterp.hpp>
#include <odecraft/Events/Events.hpp>
#include <polybox/polybox.hpp>
#include <odecraft/Core/SolverFactory.hpp>
#include <ostream>
#include <span>

namespace ode {

using namespace interp::uni;


template<typename T, typename SD>
class OdeSolver{

public:
    using State = state_t<T, SD>;
    using JacMat = jacmat_t<T, SD>;
    using RealState = SD::template state_type<T>;

    // ODE PROPERTIES
    virtual void                get_rhs(State& out, const T& t, const State& q) const = 0;
    virtual void                get_jac(JacMat& out, const T& t, const State& q) const = 0;
    virtual void                get_jac(JacMat& out, const T& t, const State& q, const State& h) const = 0;

    // ACCESSORS
    virtual const T&            get_time() const = 0;
    virtual const T&            get_new_time() const = 0;
    virtual const T&            get_old_time() const = 0;
    virtual void                get_current_vector(State& out) const = 0;
    virtual const State&        get_new_vector() const = 0;
    virtual const State&        get_old_vector() const = 0;
    virtual const T&            get_ics_time() const = 0;
    virtual const State&        get_ics_vector() const = 0;
    virtual const T&            get_stepsize() const = 0;
    virtual int                 get_direction() const = 0;
    virtual const RealState&    get_rtol() const = 0;
    virtual const RealState&    get_atol() const = 0;
    virtual const T&            get_min_step() const = 0;
    virtual const T&            get_max_step() const = 0;
    virtual size_t              get_step_count() const = 0;
    virtual bool                get_is_running() const = 0;
    virtual StepperStatus       get_status() const = 0;
    virtual std::string_view    get_message() const = 0;
    virtual bool                get_validate_ics(const T& t0, const State& q0) const = 0;
    virtual bool                get_has_valid_ics() const = 0;
    virtual Stepper             get_method() const = 0;
    virtual BoxedInterp<T, SD>  get_state_interpolator(int bdr1, int bdr2) const = 0;
    virtual T                   get_auto_step(const T& t, const State& q) const = 0;
    virtual T                   get_auto_step() const = 0;
    virtual void                show_state(std::ostream& out, int prec=8) const = 0;
    virtual std::unique_ptr<OdeSolver<T, SD>> clone() const = 0;

    // MODIFIERS
    virtual bool                do_advance() = 0;
    virtual bool                do_advance_by(T interval) = 0;
    virtual bool                do_advance_until(T time) = 0;
    virtual bool                do_advance_until(T time, const observer_t<T, State>& observer) = 0;
    virtual bool                do_advance_until(T time, const observer_t<T, State>& observer, std::span<const T> checkpoints) = 0;
    virtual BoxedInterp<T, SD>  do_interpolate_until(T time) = 0;
    virtual BoxedInterp<T, SD>  do_interpolate_until(T time, const observer_t<T, State>& observer) = 0;
    virtual void                do_reset() = 0;
    virtual void                do_kill() = 0;
    virtual bool                do_set_ics(T t0, const State& y0, T stepsize = 0, int direction = 0) = 0;

    virtual ~OdeSolver() = default;

protected:

    OdeSolver() = default;
    OdeSolver(const OdeSolver&) = default;
    OdeSolver(OdeSolver&&) noexcept = default;
    OdeSolver& operator=(const OdeSolver&) = default;
    OdeSolver& operator=(OdeSolver&&) noexcept = default;
};


template<typename T, typename SD>
class OdeRichSolver : public OdeSolver<T, SD>{
    using Base = OdeSolver<T, SD>;
public:
    using State = state_t<T, SD>;
    // ACCESSORS
    virtual const EventCollection<T, State>&    get_event_col() const = 0;
    virtual int                                 get_event_idx(const std::string& name) const = 0;
    virtual bool                                get_at_event(int event_idx = -1) const = 0;
    virtual EventState<T, State>                get_current_event() const = 0;
    virtual bool                                get_at_canon_event() const = 0;
    // MODIFIERS
    virtual bool                            do_advance_to_event(const std::vector<size_t>& event_idx = {}) = 0;
    virtual bool                            do_advance_to_event(const T& tmax, const std::vector<size_t>& event_idx = {}) = 0;
    virtual bool                            do_advance_to_event(const std::vector<std::string>& event_names) = 0;
    virtual bool                            do_advance_to_event(const T& tmax, const std::vector<std::string>& event_names) = 0;

    ~OdeRichSolver() = default;

protected:

    OdeRichSolver() = default;
    OdeRichSolver(const OdeRichSolver& other) = default;
    OdeRichSolver(OdeRichSolver&& other) noexcept = default;
    OdeRichSolver& operator=(const OdeRichSolver& other) = default;
    OdeRichSolver& operator=(OdeRichSolver&& other) noexcept = default;
};


} // namespace ode
