#ifndef ODECRAFT_BASESOLVER_IMPL_HPP
#define ODECRAFT_BASESOLVER_IMPL_HPP


#include <odecraft/Toolkit/FinDiff.hpp>
#include <odecraft/Toolkit/Tools.hpp>
#include <stdexcept>
#include <utility>
#include "BaseSolver_mem_impl.hpp"
#include "ndspan/core/ndview.hpp"
#include "odecraft/Toolkit/Scratch.hpp"


namespace ode{


// ODE PROPERTIES

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::Rhs(T* out, const T& t, const T* q) const noexcept(!detail::rhs_may_throw){
    return ode_.Rhs(out, t, q);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::rhs(T* out, const T& t, const T* q) const noexcept(!detail::rhs_may_throw){
    this->Rhs(out, t, q);
    this->rhs_eval_count_++;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::Jac(T* out, const T& t, const T* q) const noexcept(!detail::rhs_may_throw){
    
    if constexpr (JP == JacPolicy::Approx){
        return this->Jac(out, t, q, nullptr);
    } else if constexpr (JP == JacPolicy::Autodiff){
        const size_t nsys = this->nsys();
        auto out_duals = dual_scratch_.template request<N>(nsys);
        MutView<T, Layout::F, N, N> jacmat = this->jac_view(out);

        DualType<T, N, 1>::with_default_nvars(nsys, [&](){
            ode_.Rhs(out_duals.data(), t, SeedVec<T, int(N), 1>{q, int(nsys), 1});
            for (size_t i=0; i<nsys; i++){
                for (size_t j=0; j<nsys; j++){
                    jacmat(i, j) = out_duals[i].get_diff_wrt(j);
                }
            }
        });
    } else if constexpr (JP == JacPolicy::Nullable){
        // The Jacobian was declared but may be null at runtime, in which case we fall
        // back to a numerical approximation (see the note in getJacPolicy).
        if (ode_.Jac != nullptr){
            ode_.Jac(out, t, q);
        } else {
            return this->Jac(out, t, q, nullptr);
        }
    } else {
        ode_.Jac(out, t, q);
    }

}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::jac(T* out, const T& t, const T* q) const noexcept(!detail::rhs_may_throw){
    this->Jac(out, t, q);
    this->jac_eval_count_++;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::Jac(T* out, const T& t, const T* q, const T* dt) const noexcept(!detail::rhs_may_throw){
    const size_t n = this->nsys();

    // jac_approx works on 3 state vectors: the shifted point and the two evaluations around it
    auto scratch = scratch_.template request<3*N>(3*n);

    ode::jac_approx<T>([this](T* out_, const T& t_, const T* q_){
        this->Rhs(out_, t_, q_);
    }, out, scratch.data(), t, q, dt, this->atol(), n);
}

// PUBLIC ACCESSORS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::t() const{
    return t_true_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
View1D<T, N> BaseSolver<Derived, T, N, SP, OdeType>::vector_new() const{
    return View1D<T, N>(this->q_new_.data(), this->nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
View1D<T, N> BaseSolver<Derived, T, N, SP, OdeType>::vector_old() const{
    return View1D<T, N>(this->q_old_.data(), this->nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::stepsize() const{
    return habs_new_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
int BaseSolver<Derived, T, N, SP, OdeType>::direction() const{
    return direction_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::rtol() const{
    return rtol_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
 const T& BaseSolver<Derived, T, N, SP, OdeType>::atol() const{
    return atol_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::min_step() const{
    return min_step_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::max_step() const{
    return max_step_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::is_running() const{
    switch (this->status_){
        case StepperStatus::NewStep:
        case StepperStatus::BetweenSteps:
            return true;
        default:
            return false;
    }
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
size_t BaseSolver<Derived, T, N, SP, OdeType>::step_count() const{
    return step_count_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
StepperStatus BaseSolver<Derived, T, N, SP, OdeType>::status() const{
    return status_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const char* BaseSolver<Derived, T, N, SP, OdeType>::message() const{
    return status_message(this->status_);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::show_state(int prec) const{
    auto vector = scratch_.template request<N>(this->nsys());
    this->fill_current_vector(vector.data());
    SolverState<T, N>(vector.data(), this->t(), this->stepsize(), this->nsys(), this->step_count(), this->message()).show(prec);
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::t_ics() const{
    return t_ics_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
View1D<T, N> BaseSolver<Derived, T, N, SP, OdeType>::vector_ics() const{
    return View1D<T, N>{this->q_ics_.data(), this->nsys()};
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::interp(T* out, const T& t) const{
    if (t == this->t_old()) {
        std::copy(
            q_old_.data(),
            q_old_.data() + this->nsys(),
            out
        );
    } else if (t == this->t_new()){
        std::copy(
            q_new_.data(),
            q_new_.data() + this->nsys(),
            out
        );
    } else if (lt(t_old(), t) && lt(t, t_new())){
        return interp_impl(out, t);
    } else {
        throw std::out_of_range("Out of bounds interpolation requested");
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
size_t BaseSolver<Derived, T, N, SP, OdeType>::rhs_eval_count() const{
    return rhs_eval_count_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
size_t BaseSolver<Derived, T, N, SP, OdeType>::jac_eval_count() const{
    return jac_eval_count_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
T BaseSolver<Derived, T, N, SP, OdeType>::auto_step(T t, const T* q) const{
    //returns absolute value of emperically determined first step.

    const int dir = this->direction();
    if (dir == 0){
        //needed even if the resulting stepsize will have a positive value.
        throw std::runtime_error("Cannot auto-determine step when a direction of integration has not been specified.");
    }
    size_t n = this->nsys();
    T h0, d2, h1;

    auto scratch = scratch_.template request<4*N>(4*n);
    T* y1 = scratch.data();

    T* f1 = y1+n;
    T* scale = y1+2*n;
    T* f0 = y1+3*n;
    for (size_t i=0; i<n; i++){
        scale[i] = atol() + abs(q[i])*rtol();
    }
    this->rhs(f0, t, q);
    T d0 = rms_norm(q, scale, n);
    T d1 = rms_norm(f0, scale, n);
    if (d0 * 100000 < 1 || d1 * 100000 < 1){
        h0 = T(1)/1000000;
    }
    else{
        h0 = d0/d1/100;
    }
    for (size_t i=0; i<n; i++){
        y1[i] = q[i]+h0*dir*f0[i];
    }
    this->rhs(f1, t+h0*dir, y1);
    T* tmp = y1; //y1 can be recycled, its not used anymore below
    for (size_t i=0; i<n; i++){
        tmp[i] = f1[i] - f0[i];
    }
    d2 = rms_norm(tmp, scale, n) / h0;
    if (d1 <= 1e-15 && d2 <= 1e-15){
        h1 = ndspan::max<T>(T(1)/1000000, h0/1000);
    }else{
        h1 = pow(100*ndspan::max<T>(d1, d2), -T(1)/T(ERR_EST_ORDER+1));
    }
    return ndspan::max<T>(ndspan::min_of_pack<T>(T(100*h0), h1, this->max_step()), this->min_step());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
T BaseSolver<Derived, T, N, SP, OdeType>::auto_step() const{
    auto vector = scratch_.template request<N>(this->nsys());
    this->fill_current_vector(vector.data());
    return auto_step(this->t(), vector.data());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
std::unique_ptr<typename BaseSolver<Derived, T, N, SP, OdeType>::CloneType> BaseSolver<Derived, T, N, SP, OdeType>::clone() const {
    return std::make_unique<Derived>(*THIS);
}

// PUBLIC MODIFIERS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::advance(){
    if (this->is_running()){
        return Accessor::call_Adv_Impl(*THIS);
    } else {
        this->warn_dead();
        return false;
    }
}



template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::advance_until(const T& time){
    return this->generic_advance_until(time, nullptr, nullptr);
}



template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
BoxedInterp<T, N> BaseSolver<Derived, T, N, SP, OdeType>::interpolate_until(const T& time){
    return this->generic_interpolate_until(time, nullptr);
}




template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::advance_by(T interval){
    if (interval < 0){
        throw std::logic_error("Interval must be non-negative in advance_by. Its sign is determined by the solver's direction of integration.");
    }
    return this->advance_until(this->t() + interval*this->direction());
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::set_ics(T t0, const T* y0, T stepsize, int direction){
    
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

        t_ics_ = t0;
        habs_ics_ = stepsize;
        std::copy(y0, y0 + this->nsys(), q_ics_.data());
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

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::kill(){
    if (this->is_running()){
        this->status_ = StepperStatus::Killed;
    }
}

//====================== STATIC OVERRIDES =====================================

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
BoxedInterp<T, N> BaseSolver<Derived, T, N, SP, OdeType>::state_interpolator(int bdr1, int bdr2) const{
    if (this->t_old() == this->t_new()){
        return pbox::make_box<LocalInterpolator<T, N>>(this->t(), q_new_.data(), this->nsys());
    } else {
        auto interp = this->local_interp();
        return pbox::make_box<CustomLocalInterpolator<T, N, decltype(interp)>>(std::move(interp), t_old_, t_new_, q_old_.data(), q_new_.data(), this->nsys(), bdr1, bdr2);
    }

}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
StepResult BaseSolver<Derived, T, N, SP, OdeType>::adapt_impl(T& t_new, T& habs_new, T* q_new, const T& t, const T& habs, const T* q) noexcept(!detail::rhs_may_throw){
    return ODECRAFT_CALL_DERIVED(adapt_impl, t_new, habs_new, q_new, t, habs, q);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::interp_impl(T* result, const T& t) const{
    ODECRAFT_CALL_DERIVED(interp_impl, result, t);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
auto BaseSolver<Derived, T, N, SP, OdeType>::local_interp() const{
    return Accessor::call_local_interp(*THIS);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::fill_current_vector_impl(T* out) const{
    if (this->is_at_new_state()){
        std::copy(q_new_.data(), q_new_.data()+this->nsys(), out);
    } else {
        this->interp_impl(out, this->t());
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::Reset(){
    if (this->has_valid_ics()){
        status_ = StepperStatus::NewStep;
        step_count_ = 0;
        rhs_eval_count_ = 0;
        jac_eval_count_ = 0;
        t_old_ = t_new_ = t_true_ = t_ics_;
        q_old_ = q_new_ = q_ics_;
        habs_new_ = habs_ics_;
    }
}

//=============================================================================

// OVERRIDEN IN RICH SOLVER


// HELPER METHODS


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::set_status(StepperStatus new_status){
    this->status_ = new_status;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::t_new() const{
    return this->t_new_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
const T& BaseSolver<Derived, T, N, SP, OdeType>::t_old() const{
    return this->t_old_;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::fill_current_vector(T* out) const{
    return ODECRAFT_CALL_DERIVED(fill_current_vector_impl, out);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::warn_dead() const{
#ifndef ODECRAFT_NO_WARN
    this->cerr("\nIntegrator has permanently stopped");
#endif
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::ReAdjust(const T* new_vector){
    t_new_ = this->t();
    habs_new_ = this->stepsize();
    std::copy(new_vector, new_vector + this->nsys(), q_new_.data());
    t_old_ = t_new_;
    q_old_ = q_new_;
    status_ = StepperStatus::NewStep;
    // By having the old state set to the new state, we close the interpolation interval
    // as if restarting from new initial conditions
    // This is to ban interpolation temporarily until a new step is achived,
    // because some steppers have cached dense output coefficients
    // that are valid for [old_state, new_state] pair before re-adjusting
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::validate_ics(T t0, const T* q0) const {
    return ODECRAFT_CALL_DERIVED(validate_ics_impl, t0, q0);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::has_valid_ics() const {
    switch (status_){
        case StepperStatus::BadICS:
        case StepperStatus::Uninitialized:
            return false;
        default:
            return true;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::validate_ics_impl(T t0, const T* q0) const {

    if (!all_are_finite(q0, this->nsys()) || !isfinite(t0)){
        return false;
    }

    auto rhs_out = scratch_.template request<N>(this->nsys());
    
    this->Rhs(rhs_out.data(), t0, q0);

    return all_are_finite(rhs_out.data(), this->nsys());
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::is_at_new_state() const{
    return status_ == StepperStatus::NewStep;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::cerr(const std::string& message) const {
    // One write, so that messages from concurrent solvers never interleave. It starts by clearing
    // the console line, in case a progress line (drawn with \r and no newline) is on it.
    std::cerr << ("\033[2K\r" + message + "\n");
}


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
MutView<T, Layout::F, N, N> BaseSolver<Derived, T, N, SP, OdeType>::jac_view(T* j) const{
    /*
    returns a high level view of the jacobian matrix, so that its elements
    can be accessed using matrix(i, j). May be used as follows:
    auto matrix = solver->jac_view(jac_ptr);
    matrix(i, j) = ...
    */
    return MutView<T, Layout::F, N, N>(j, this->nsys(), this->nsys());
}



// PROTECTED CONSTRUCTOR

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
BaseSolver<Derived, T, N, SP, OdeType>::BaseSolver(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, EventList<T> events) :
    scratch_(request_scratch_page<T, 6*N>(6*q0.size())),
    scalar_scratch_(request_scratch_page<T, 2>()),
    dual_scratch_(q0.size()),
    ode_(std::move(ode)),
    t_ics_(t0), habs_ics_(0), q_ics_(q0.size()), t_old_(t0), t_new_(t0), t_true_(t0), habs_new_(0),
    rtol_(rtol), atol_(atol),
    min_step_(min_step),
    max_step_(max_step == 0 ? inf<T>() : max_step),
    q_old_(q0.size()), q_new_(q0.size()),
    nsys_(q0.size()),
    direction_(direction){
        assert(this->nsys() > 0 && "Ode system size is 0");
        if (abs(direction) != 1){
            throw std::runtime_error("The integration direction must be 1 or -1");
        } else if (stepsize < 0){
            throw std::runtime_error("The stepsize argument cannot be negative");
        } else if (max_step_ < min_step_){
            throw std::runtime_error("Maximum allowed stepsize cannot be smaller than minimum allowed stepsize");
        } else if (q0.data() == nullptr){
            status_ = StepperStatus::Uninitialized;
        } else if (!this->validate_ics_impl(t0, q0.data())){
            status_ = StepperStatus::BadICS;
        } else if (events.size() > 0){
            throw std::invalid_argument("Cannot instantiate a Solver with events when it is declared with a non-rich SolverPolicy");
        } else {
            if (stepsize == 0){
                habs_ics_ = habs_new_ = this->auto_step(t0, q0.data());
            } else {
                habs_ics_ = habs_new_ = abs(stepsize);
            }
            std::copy(q0.data(), q0.data() + this->nsys(), q_ics_.data());
            q_old_ = q_new_ = q_ics_;
        }
}


// PRIVATE METHODS

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::validate_it(StepResult result, const T& new_time){
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


template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
void BaseSolver<Derived, T, N, SP, OdeType>::move_state(const T& time){
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

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::lt(const T& t_a, const T& t_b) const {
    if (direction_ == 1){
        return t_a < t_b;
    } else {
        return t_a > t_b;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::le(const T& t_a, const T& t_b) const {
    if (direction_ == 1){
        return t_a <= t_b;
    } else {
        return t_a >= t_b;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
StepResult BaseSolver<Derived, T, N, SP, OdeType>::adapt_it(T* q_out, T& t_out, T& habs_out){
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
                q_new_.data()
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
            q_new_.data()
        );
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
bool BaseSolver<Derived, T, N, SP, OdeType>::take_step(T& t_out, T& habs_out){
    StepResult result;
    if constexpr (detail::scratch_is_static<T, N>){
        // The vectors live inside the solver here, so the step goes to a local array first:
        // writing it into q_old_ in place measured much slower.
        auto q_out = scratch_.template request<N>(this->nsys());
        result = adapt_it(q_out.data(), t_out, habs_out);
        if (!validate_it(result, t_out)){
            return false;
        }
        std::swap(q_old_, q_new_);
        std::copy(q_out.data(), q_out.data() + this->nsys(), q_new_.data()); // cheap copy in this branch
    } else {
        // q_old_ is overwritten by q_new_ whether the step succeeds (the swap below) or fails
        // (validate_it), so the step is written straight into it, and on success the two
        // buffers trade places: no scratch vector, and no copy of nsys elements per step.
        result = adapt_it(q_old_.data(), t_out, habs_out);
        if (!validate_it(result, t_out)){
            return false;
        }
        std::swap(q_old_, q_new_);
    }
    std::swap(t_old_, t_new_);
    std::swap(t_new_, t_out);
    std::swap(habs_new_, habs_out);
    t_true_ = t_new_;
    step_count_++;
    return true;
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename... Args>
bool BaseSolver<Derived, T, N, SP, OdeType>::Adv_Impl(Args&&... args){
    const int d = this->direction();
    if constexpr (sizeof...(Args) > 0){
        T time_floor = nearest_time(args...);
        if (this->is_at_new_state() && (time_floor*d <= t_new()*d)){
            return false;
        } else if (this->is_at_new_state()){
            auto scalar_scratch = scalar_scratch_.template request<2>(2);
            T& scratch_time = scalar_scratch[0];
            T& scratch_habs = scalar_scratch[1];
            if (this->take_step(scratch_time, scratch_habs)){
                if (ODECRAFT_CALL_DERIVED(RequestTimeFloor, scratch_time)){
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
        auto scalar_scratch = scalar_scratch_.template request<2>(2);
        T& scratch_time = scalar_scratch[0];
        T& scratch_habs = scalar_scratch[1];
        if (this->take_step(scratch_time, scratch_habs)){
            if (ODECRAFT_CALL_DERIVED(RequestTimeFloor, scratch_time) && lt(scratch_time, t_new())){
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

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename... U>
T BaseSolver<Derived, T, N, SP, OdeType>::nearest_time(const U&... t) const{
    static_assert(sizeof...(U) > 0, "BaseSolver::nearest_time requires at least one argument");
    return nearest_time_priv(t...);
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename A, typename... Rest>
const T& BaseSolver<Derived, T, N, SP, OdeType>::nearest_time_priv(const A& t_a, const Rest&... t_rest) const{
    if constexpr (sizeof...(Rest) > 0){
        return nearest_time_helper(t_a, t_rest...);
    } else {
        return t_a;
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename A, typename B, typename... Rest>
const T& BaseSolver<Derived, T, N, SP, OdeType>::nearest_time_helper(const A& t_a, const B& t_b, const Rest&... t_rest) const{
    if constexpr (sizeof...(Rest) > 0){
        return nearest_time_helper(nearest_of(t_a, t_b), t_rest...);
    }else{
        return nearest_of(t_a, t_b);
    }
}

template<typename Derived, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType>
template<typename A, typename B>
const T& BaseSolver<Derived, T, N, SP, OdeType>::nearest_of(const A& t_a, const B& t_b) const{
    if (this->direction() == 1){
        return (t_a < t_b ? t_a : t_b);
    }else{
        return (t_a > t_b ? t_a : t_b);
    }
}

} // namespace ode

#endif // ODECRAFT_BASESOLVER_IMPL_HPP