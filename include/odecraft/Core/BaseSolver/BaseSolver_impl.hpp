#pragma once


#include <odecraft/Toolkit/FinDiff.hpp>
#include <odecraft/Toolkit/Toolkit.hpp>
#include <stdexcept>
#include <tuple>
#include <utility>
#include "BaseSolver_mem_impl.hpp"
#include "odecraft/Toolkit/Tools.hpp"
#include "xdiff/dual/dual_decls.hpp"


namespace ode{


// ODE PROPERTIES

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::Rhs(State& out, const T& t, const State& q) const noexcept(!detail::rhs_may_throw){
    return ode_.Rhs(out, t, q);
}


template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::Jac(JacMat& out, const T& t, const State& q) const noexcept(!detail::rhs_may_throw){
    
    if constexpr (JP == JacPolicy::Approx){
        return this->JacApprox(out, t, q);
    } else if constexpr (JP == JacPolicy::Autodiff){
        decltype(auto) out_duals = dual_scratch.get();
        decltype(auto) dual_seeds = seed_scratch.get();

        xdiff::set_seeds<T, SD>(dual_seeds, q);

        ode_.Rhs(out_duals, t, dual_seeds);

        // Row i of the Jacobian is the gradient of f_i
        SD::JD::linalg::for_each_row([](value_t<T, SD>& out_ij, const value_t<T, SD>& dfi_dxj){
            out_ij = dfi_dxj;
        }, [](const dual_t<T, SD, 1>& fi){
            return std::tie(fi.grad());
        }, std::tie(out), out_duals);
    } else if constexpr (JP == JacPolicy::Nullable){
        // The Jacobian was declared but may be null at runtime, in which case we fall
        // back to a numerical approximation (see the note in getJacPolicy).
        if (ode_.Jac != nullptr){
            ode_.Jac(out, t, q);
        } else {
            return this->JacApprox(out, t, q);
        }
    } else {
        ode_.Jac(out, t, q);
    } 
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::Jac(JacMat& out, const T& t, const State& q, const State& dq) const noexcept(!detail::rhs_may_throw){
    this->JacApprox(out, t, q, dq);
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::JacApprox(JacMat& out, const T& t, const State& q) const noexcept(!detail::rhs_may_throw){
    // jac_approx works on 3 state vectors: the shifted point and the two evaluations around it
    decltype(auto) w_x = scratch_0.get();
    decltype(auto) w_y1 = scratch_1.get();
    decltype(auto) w_y2 = scratch_2.get();

    ode::jac_approx_autostep<T, SD>([this](State& out_, const T& t_, const State& q_){
        THIS->Rhs(out_, t_, q_);
    }, out, w_x, w_y1, w_y2, t, q, this->atol());
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::JacApprox(JacMat& out, const T& t, const State& q, const State& dq) const noexcept(!detail::rhs_may_throw){
    decltype(auto) w_x = scratch_0.get();
    decltype(auto) w_y1 = scratch_1.get();
    decltype(auto) w_y2 = scratch_2.get();

    ode::jac_approx_explicit<T, SD>([this](State& out_, const T& t_, const State& q_){
        THIS->Rhs(out_, t_, q_);
    }, out, w_x, w_y1, w_y2, t, q, dq);
}

// PUBLIC ACCESSORS

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const T& BaseSolver<Derived, T, SD, OdeType, SP>::t() const{
    return t_true_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const BaseSolver<Derived, T, SD, OdeType, SP>::State& BaseSolver<Derived, T, SD, OdeType, SP>::vector_new() const{
    return this->q_new_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const BaseSolver<Derived, T, SD, OdeType, SP>::State& BaseSolver<Derived, T, SD, OdeType, SP>::vector_old() const{
    return this->q_old_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const T& BaseSolver<Derived, T, SD, OdeType, SP>::stepsize() const{
    return habs_new_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
int BaseSolver<Derived, T, SD, OdeType, SP>::direction() const{
    return direction_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const BaseSolver<Derived, T, SD, OdeType, SP>::RealState& BaseSolver<Derived, T, SD, OdeType, SP>::rtol() const{
    return rtol_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const BaseSolver<Derived, T, SD, OdeType, SP>::RealState& BaseSolver<Derived, T, SD, OdeType, SP>::atol() const{
    return atol_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const T& BaseSolver<Derived, T, SD, OdeType, SP>::min_step() const{
    return min_step_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const T& BaseSolver<Derived, T, SD, OdeType, SP>::max_step() const{
    return max_step_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::is_running() const{
    switch (this->status_){
        case StepperStatus::NewStep:
        case StepperStatus::BetweenSteps:
            return true;
        default:
            return false;
    }
}


template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
size_t BaseSolver<Derived, T, SD, OdeType, SP>::step_count() const{
    return step_count_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
StepperStatus BaseSolver<Derived, T, SD, OdeType, SP>::status() const{
    return status_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
std::string_view BaseSolver<Derived, T, SD, OdeType, SP>::message() const{
    return status_message(this->status_);
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::show_state(std::ostream& out, int prec) const{
    decltype(auto) worker = scratch_0.get();
    this->fill_current_vector(worker);

    out << std::setprecision(prec) << "\n" << 
    "OdeSolver current state:\n---------------------------\n"
    "\ttime       : " << this->t() << "\n" <<
    "\tq          : ";
    
    SD::console(out, worker) << "\n" <<
    "\tstepsize   : " << this->stepsize() << "\n" <<
    "\tUpdates    : " << this->step_count() << "\n" <<
    "\tStatus     : " << this->message() << std::endl;
}


template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const T& BaseSolver<Derived, T, SD, OdeType, SP>::t_ics() const{
    return t_ics_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const BaseSolver<Derived, T, SD, OdeType, SP>::State& BaseSolver<Derived, T, SD, OdeType, SP>::vector_ics() const{
    return this->q_ics_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
T BaseSolver<Derived, T, SD, OdeType, SP>::auto_step(const T& t, const State& q) const{
    //returns absolute value of emperically determined first step.

    const int dir = this->direction();
    if (dir == 0){
        //needed even if the resulting stepsize will have a positive value.
        throw std::runtime_error("Cannot auto-determine step when a direction of integration has not been specified.");
    }
    T h0, d2, h1;

    decltype(auto) scale = real_scratch.get();
    decltype(auto) y1 = scratch_0.get();
    decltype(auto) f0 = scratch_1.get();
    decltype(auto) f1 = scratch_2.get();

    SD::for_each([](T& s_i, const T& rtol_i, const T& atol_i, const value_t<T, SD>& q_i){
        s_i = atol_i + rtol_i * abs(q_i);
    }, scale, this->rtol(), this->atol(), q);

    THIS->Rhs(f0, t, q);
    T d0 = rms_norm<SD, T>(q, scale);
    T d1 = rms_norm<SD, T>(f0, scale);
    if (d0 * 1e5 < 1 || d1 * 1e5 < 1){
        h0 = T(1)/1e6;
    } else {
        h0 = d0/d1/100;
    }
    SD::for_each([&h0, dir](value_t<T, SD>& y1_i, const value_t<T, SD>& q_i, const value_t<T, SD>& f0_i){
        y1_i = q_i + h0 * dir * f0_i;
    }, y1, q, f0);

    THIS->Rhs(f1, t+h0*dir, y1);

    State& tmp = y1;//y1 can be recycled, its not used anymore below
    SD::for_each([](value_t<T, SD>& tmp_i, const value_t<T, SD>& f0_i, const value_t<T, SD>& f1_i){
        tmp_i = (f1_i - f0_i);
    }, tmp, f0, f1);
    
    d2 = rms_norm<SD, T>(tmp, scale) / h0;
    if (d1 <= 1e-15 && d2 <= 1e-15){
        h1 = ndspan::max<T>(T(1)/1e6, h0/1e3);
    }else{
        h1 = pow(100*ndspan::max<T>(d1, d2), -T(1)/T(ERR_EST_ORDER+1));
    }
    return ndspan::max<T>(ndspan::min_of_pack<T>(T(100*h0), h1, this->max_step()), this->min_step());
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
T BaseSolver<Derived, T, SD, OdeType, SP>::auto_step() const{
    return auto_step(this->t_new(), this->vector_new());
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
std::unique_ptr<typename BaseSolver<Derived, T, SD, OdeType, SP>::CloneType> BaseSolver<Derived, T, SD, OdeType, SP>::clone() const {
    return std::make_unique<Derived>(*THIS);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::advance(){
    if (this->is_running()){
        return THIS->Adv_Impl();
    } else {
        this->warn_dead();
        return false;
    }
}


template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::advance_until(const T& time){
    return this->generic_advance_until(time, nullptr, nullptr);
}


template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
BoxedInterp<T, SD> BaseSolver<Derived, T, SD, OdeType, SP>::interpolate_until(const T& time){
    return this->generic_interpolate_until(time, nullptr);
}


template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::advance_by(T interval){
    if (interval < 0){
        throw std::logic_error("Interval must be non-negative in advance_by. Its sign is determined by the solver's direction of integration.");
    }
    return this->advance_until(this->t() + interval*this->direction());
}


template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::set_ics(T t0, const State& y0, T stepsize, int direction){
    
    if (direction != 1 && direction != -1 && direction != 0){
        throw std::domain_error("Integration direction must be 1, -1, or 0");
    }

    direction = (direction == 0) ? this->direction() : direction; // if 0, keep existing direction;
    if (this->validate_ics(t0, y0)){
        if (stepsize < 0) {
            this->cerr("Cannot set negative stepsize in solver initialization");
            return false;
        } else if (stepsize == 0) {
            this->direction_ = direction;
            stepsize = this->auto_step(t0, y0);
        } else {
            this->direction_ = direction;
        }

        t_ics_ = std::move(t0);
        habs_ics_ = stepsize;
        q_ics_ = y0;
        status_ = StepperStatus::NewStep;
        THIS->Reset();
        return true;
    }else {
#ifndef ODECRAFT_NO_WARN
        this->cerr("Tried to set invalid initial conditions");
#endif
        return false;
    }
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::kill(){
    if (this->is_running()){
        this->status_ = StepperStatus::Killed;
    }
}

//====================== STATIC OVERRIDES =====================================

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
BoxedInterp<T, SD> BaseSolver<Derived, T, SD, OdeType, SP>::state_interpolator(int bdr1, int bdr2) const{
    if (this->t_old() == this->t_new()){
        return pbox::make_box<LocalInterpolator<T, SD>>(this->t(), q_new_);
    } else {
        // detached: the segment outlives the step (and the solver itself)
        auto interp = this->local_interp_impl().detach();
        return pbox::make_box<CustomLocalInterpolator<T, SD, decltype(interp)>>(std::move(interp), t_old_, t_new_, q_old_, q_new_, bdr1, bdr2);
    }
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
StepResult BaseSolver<Derived, T, SD, OdeType, SP>::adapt_impl(T& t_new, T& habs_new, State& y_new, const T& t, const T& habs, const State& q) noexcept(!detail::rhs_may_throw){
    return THIS->adapt_impl(t_new, habs_new, y_new, t, habs, q);
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
auto BaseSolver<Derived, T, SD, OdeType, SP>::local_interp_impl() const{
    return THIS->local_interp_impl();
}

/// @brief Non-owning interpolator for the closed interval [t_old, t_new]. Valid while the solver is not advanced or goes out of scope
template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
auto BaseSolver<Derived, T, SD, OdeType, SP>::light_interp() const {
    return [f = this->local_interp_impl(), this](State& out, const T& t){
        if (t == this->t_old()) {
            copy_state<SD>(out, q_old_);
        } else if (t == this->t_new()){
            copy_state<SD>(out, q_new_);
        } else if (lt(t_old(), t) && lt(t, t_new())){
            return f(out, t);
        } else {
            throw std::domain_error("Out of bounds interpolation requested");
        }
    };
}

/// @brief Interpolator for the closed interval [t_old, t_new], detached from the Solver. Can be expensive to construct
template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
auto BaseSolver<Derived, T, SD, OdeType, SP>::heavy_interp() const {
    return [
        f = this->local_interp_impl().detach(),
        d = this->direction(),
        t1 = this->t_old(),
        q1 = this->vector_old(),
        t2 = this->t_new(),
        q2 = this->vector_new()
    ](State& out, const T& t){
        if (t == t1) {
            copy_state<SD>(out, q1);
        } else if (t == t2){
            copy_state<SD>(out, q2);
        } else {
            if (d == 1 && t1 < t && t < t2){
                return f(out, t);
            } else if (d == -1 && t2 < t && t < t1){
                return f(out, t);
            } else {
                throw std::domain_error("Out of bounds interpolation requested");
            }
        }
    };
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::fill_current_vector_impl(State& out) const{
    if (this->is_at_new_state()){
        copy_state<SD>(out, q_new_);
    } else {
        this->local_interp_impl()(out, this->t());
    }
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::Reset(){
    if (this->has_valid_ics()){
        status_ = StepperStatus::NewStep;
        step_count_ = 0;
        t_old_ = t_new_ = t_true_ = t_ics_;
        q_old_ = q_new_ = q_ics_;
        habs_new_ = habs_ics_;
    }
}

//=============================================================================

// OVERRIDEN IN RICH SOLVER


// HELPER METHODS


template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::set_status(StepperStatus new_status){
    this->status_ = new_status;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const T& BaseSolver<Derived, T, SD, OdeType, SP>::t_new() const{
    return this->t_new_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
const T& BaseSolver<Derived, T, SD, OdeType, SP>::t_old() const{
    return this->t_old_;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::fill_current_vector(State& out) const{
    return THIS->fill_current_vector_impl(out);
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::warn_dead() const{
#ifndef ODECRAFT_NO_WARN
    this->cerr("\nIntegrator has permanently stopped");
#endif
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::ReAdjust(const State& new_state){
    t_new_ = this->t();
    habs_new_ = this->stepsize();
    copy_state<SD>(q_new_, new_state);
    t_old_ = t_new_;
    q_old_ = q_new_;
    status_ = StepperStatus::NewStep;
    // By having the old state set to the new state, we close the interpolation interval
    // as if restarting from new initial conditions
    // This is to ban interpolation temporarily until a new step is achived,
    // because some steppers have cached dense output coefficients
    // that are valid for [old_state, new_state] pair before re-adjusting
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::validate_ics(const T& t0, const State& q0) const {
    return THIS->validate_ics_impl(t0, q0);
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::has_valid_ics() const {
    switch (status_){
        case StepperStatus::BadICS:
        case StepperStatus::Uninitialized:
            return false;
        default:
            return true;
    }
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::validate_ics_impl(const T& t0, const State& q0) const {
    if (!all_are_finite<T, SD>(q0) || !isfinite(t0)){
        return false;
    }
    decltype(auto) worker = this->scratch_0.get();
    
    THIS->Rhs(worker, t0, q0);

    return all_are_finite<T, SD>(worker);
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::is_at_new_state() const{
    return status_ == StepperStatus::NewStep;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::cerr(const std::string& message) const {
    // One write, so that messages from concurrent solvers never interleave. It starts by clearing
    // the console line, in case a progress line (drawn with \r and no newline) is on it.
    std::cerr << ("\033[2K\r" + message + "\n");
}

// PROTECTED CONSTRUCTOR

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
BaseSolver<Derived, T, SD, OdeType, SP>::BaseSolver(OdeType ode, T t0, State q0, RealState rtol, RealState atol, T min_step, T max_step, T stepsize, int direction, EventList<T, State> events) :
    scratch_0(q0), scratch_1(q0), scratch_2(q0), real_scratch(rtol),
    ode_(std::move(ode)),
    t_ics_(t0), habs_ics_(0), t_old_(t0), t_new_(t0), t_true_(t0), habs_new_(0),
    min_step_(min_step), max_step_(max_step == 0 ? inf<T>() : max_step),
    q_ics_(q0), q_old_(q0), q_new_(std::move(q0)),
    rtol_(std::move(rtol)), atol_(std::move(atol)), direction_(direction){
        if (abs(direction) != 1){
            throw std::runtime_error("The integration direction must be 1 or -1");
        } else if (stepsize < 0){
            throw std::runtime_error("The stepsize argument cannot be negative");
        } else if (max_step_ < min_step_){
            throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
        } else if (!this->validate_ics_impl(t0, q_new_)){
            status_ = StepperStatus::BadICS;
        } else if (events.size() > 0){
            throw std::invalid_argument("Cannot instantiate a Solver with events when it is declared with a non-rich SolverPolicy");
        } else {
            if (stepsize == 0){
                habs_ics_ = habs_new_ = this->auto_step(t0, q_new_);
            } else {
                habs_ics_ = habs_new_ = abs(stepsize);
            }
            decltype(auto) dual_worker = dual_scratch.get();
            decltype(auto) seed_worker = seed_scratch.get();
            SD::resize_like(dual_worker, q_new_);
            SD::resize_like(seed_worker, q_new_);
        }
}


// PRIVATE METHODS

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::validate_it(StepResult result, const T& new_time){
    bool success = true;
    switch (result){
        case StepResult::Success:
            break;
        case StepResult::NonFiniteError:
            status_ = StepperStatus::Diverges;
            success = false;
            break;
        case StepResult::TinyStepError:
            status_ = StepperStatus::TinyStep;
            success = false;
            break;
        case StepResult::MinStepError:
            status_ = StepperStatus::MinStep;
            success = false;
            break;
        case StepResult::MaxStepError:
            status_ = StepperStatus::MaxStep;
            success = false;
            break;
        case StepResult::SingularMatrixError:
            status_ = StepperStatus::SingularMatrix;
            success = false;
            break;
    }

    if (success && (new_time == this->t_new())){
        status_ = StepperStatus::TooSmallStep;
        success = false;
    }

    if (!success){
        //close the interpolation interval as most integration algorithms
        //alter their interpolation polynomials when calling adapt_impl,
        //but since the step failed, the current interpolation interval is no longer valid.
        t_old_ = t_new_;
        q_old_ = q_new_;
    }

    return success;
}


template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
void BaseSolver<Derived, T, SD, OdeType, SP>::move_state(const T& time){
    assert( (this->le(time, this->t_new())) && "Out of bounds time requested in move_state");

    if (time != this->t_new()) {
        t_true_ = time;
        status_ = StepperStatus::BetweenSteps;
    } else if (!this->is_at_new_state()){
        // update the true state to the new state, because time is exactly at t_new
        status_ = StepperStatus::NewStep;
        t_true_ = t_new_;
    }
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::lt(const T& t_a, const T& t_b) const {
    if (direction_ == 1){
        return t_a < t_b;
    } else {
        return t_a > t_b;
    }
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::le(const T& t_a, const T& t_b) const {
    if (direction_ == 1){
        return t_a <= t_b;
    } else {
        return t_a >= t_b;
    }
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
StepResult BaseSolver<Derived, T, SD, OdeType, SP>::adapt_it(State& q_out, T& t_out, T& habs_out){
    if constexpr (detail::rhs_may_throw){
        // A callback that throws abandons the step midway: q_old_ may hold part of it (see
        // take_step), and the stepper's caches may too. So the solver ends up as after a failed
        // step: stopped at the last accepted step with the interpolation interval closed, until
        // set_ics() or Reset() rebuilds everything.
        try {
            return this->adapt_impl(
                t_out,
                habs_out,
                q_out,
                t_new(),
                stepsize(),
                q_new_
            );
        } catch (...) {
            t_old_ = t_new_;
            q_old_ = q_new_;
            status_ = StepperStatus::Interrupted;
            throw;
        }
    } else {
        return this->adapt_impl(
            t_out,
            habs_out,
            q_out,
            t_new(),
            stepsize(),
            q_new_
        );
    }
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
bool BaseSolver<Derived, T, SD, OdeType, SP>::take_step(T& t_out, T& habs_out){
    StepResult result = adapt_it(q_old_, t_out, habs_out);
    if (!validate_it(result, t_out)){
        return false;
    }
    std::swap(q_old_, q_new_);
    std::swap(t_old_, t_new_);
    std::swap(t_new_, t_out);
    std::swap(habs_new_, habs_out);
    t_true_ = t_new_;
    step_count_++;
    return true;
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
template<typename... Args>
bool BaseSolver<Derived, T, SD, OdeType, SP>::Adv_Impl(Args&&... args){
    const int d = this->direction();
    if constexpr (sizeof...(Args) > 0){
        T time_floor = nearest_time(args...);
        if (this->is_at_new_state() && (time_floor*d <= t_new()*d)){
            return false;
        } else if (this->is_at_new_state()){
            T scratch_time, scratch_habs;
            if (this->take_step(scratch_time, scratch_habs)){
                if (THIS->RequestTimeFloor(scratch_time)){
                    assert((scratch_time*d > t_old()*d && scratch_time*d <= t_new()*d) && "Invalid floor requested, with additional requests");
                    time_floor = nearest_time(scratch_time, time_floor);
                }
                
                if (lt(time_floor, t_new())){
                    this->move_state(time_floor);
                }
                return true;
            }else{
                return false;
            }
        } else if (lt(time_floor, t_new())){
            this->move_state(time_floor);
            return true;
        } else {
            this->move_state(t_new());
            return true;
        }
    } else if (this->is_at_new_state()){
        T scratch_time, scratch_habs;
        if (this->take_step(scratch_time, scratch_habs)){
            if (THIS->RequestTimeFloor(scratch_time) && lt(scratch_time, t_new())){
                assert((scratch_time*d > t_old()*d && scratch_time*d <= t_new()*d) && "Invalid floor requested without additional requests.");
                this->move_state(scratch_time);
            }
            return true;
        } else {
            return false;
        }
    } else {
        this->move_state(t_new());
        return true;
    }
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
template<typename... U>
T BaseSolver<Derived, T, SD, OdeType, SP>::nearest_time(const U&... t) const{
    static_assert(sizeof...(U) > 0, "BaseSolver::nearest_time requires at least one argument");
    return nearest_time_priv(t...);
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
template<typename A, typename... Rest>
const T& BaseSolver<Derived, T, SD, OdeType, SP>::nearest_time_priv(const A& t_a, const Rest&... t_rest) const{
    if constexpr (sizeof...(Rest) > 0){
        return nearest_time_helper(t_a, t_rest...);
    } else {
        return t_a;
    }
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
template<typename A, typename B, typename... Rest>
const T& BaseSolver<Derived, T, SD, OdeType, SP>::nearest_time_helper(const A& t_a, const B& t_b, const Rest&... t_rest) const{
    if constexpr (sizeof...(Rest) > 0){
        return nearest_time_helper(nearest_of(t_a, t_b), t_rest...);
    }else{
        return nearest_of(t_a, t_b);
    }
}

template<typename Derived, typename T, typename SD, typename OdeType, SolverPolicy SP>
template<typename A, typename B>
const T& BaseSolver<Derived, T, SD, OdeType, SP>::nearest_of(const A& t_a, const B& t_b) const{
    if (this->direction() == 1){
        return (t_a < t_b ? t_a : t_b);
    }else{
        return (t_a > t_b ? t_a : t_b);
    }
}

} // namespace ode
