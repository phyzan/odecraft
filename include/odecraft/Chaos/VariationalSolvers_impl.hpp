#ifndef ODECRAFT_VARIATIONAL_SOLVERS_IMPL_HPP
#define ODECRAFT_VARIATIONAL_SOLVERS_IMPL_HPP

#include <odecraft/Chaos/VariationalSolvers.hpp>
#include <odecraft/Tools.hpp>

namespace ode::chaos{

template<typename T, size_t N, hasRhsFunc<T> OdeType>
VariationalOdeSys<T, N, OdeType>::VariationalOdeSys(OdeType ode, size_t ode_nsys) : ode_(std::move(ode)), diff_worker_(2*ode_nsys), jac_worker_(2*ode_nsys), jm_(ode_nsys, ode_nsys), nsys_(ode_nsys) {
    if constexpr (N > 0){
        assert(N==ode_nsys && "Incorrect number of equations in VariationalOdeSys");
    }
}
template<typename T, size_t N, hasRhsFunc<T> OdeType>
void VariationalOdeSys<T, N, OdeType>::Rhs(T* out, const T& t, const T* q) const{

    const size_t n = this->nsys_main();
    const T* delta_q = q + n;

    if constexpr (JP == JacPolicy::Autodiff){
        DualType::with_default_nvars(n, 
            [&](){
                DualType* rhs = diff_worker_.data();
                DualType* y = diff_worker_.data() + n;
                for (size_t i=0; i<n; i++){
                    y[i] = DualType(q[i], {.axis=int(i)});
                }
                ode_.Rhs(rhs, t, y);
                std::fill(out+n, out+2*n, 0);
                for (size_t j=0; j<n; j++){
                    out[j] = rhs[j].value();
                    for (size_t i=0; i<n; i++){
                        out[i+n] += rhs[i].get_diff_wrt(j) * delta_q[j];
                    }
                }
            }
        );
    } else {
        ode_.Rhs(out, t, q); //fills the first half (nsys) entries
        // fills jm with the jacobian of the original system at (t, q)
        // this should not call Base::Jac(out, t, q, dt), meaning
        // the approximate overload, since we have demanded that the base solver has an exact jacobian for the original system
        ode_.Jac(jm_.data(), t, q);
        for (size_t i=0; i<n; i++){
            out[i+n] = 0;
            for (size_t j=0; j<n; j++){
                out[i+n] += jm_(i, j) * q[n+j];
            }
        }
    }
}

// Only provided if it does not require finite differences, otherwise the base solver will automatically use the appropriate overload to compute the jacobian of the full system.
template<typename T, size_t N, hasRhsFunc<T> OdeType>
void VariationalOdeSys<T, N, OdeType>::Jac(T* out, const T& t, const T* q) const requires (JP == JacPolicy::Autodiff) {

    const size_t n = this->nsys_main();

    VarDualType::with_default_nvars(n,
        [&](){
            VarDualType* rhs = jac_worker_.data();
            VarDualType* y = jac_worker_.data() + n;

            for (size_t i=0; i<n; i++){
                y[i] = VarDualType(q[i], {.axis=int(i)});
            }

            ode_.Rhs(rhs, t, y);

            ndspan::MutView<T, ndspan::Layout::F> m(out, 2*n, 2*n);
            for (size_t i=0; i<n; i++){
                for (size_t j=0; j<n; j++){
                    m(i, j) = m(i+n, j+n) = rhs[i].get_diff_wrt(j);
                    m(i, j+n) = 0;
                    T sum = 0;
                    for (size_t k=0; k<n; k++){
                        sum += rhs[i].get_diff_wrt(k, j) * q[n+k];
                    }
                    m(i+n, j) = sum;
                }
            }
        }
    );
}

template<typename T, size_t N, hasRhsFunc<T> OdeType>
const OdeType& VariationalOdeSys<T, N, OdeType>::ode() const{
    return ode_;
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
template<typename... Args>
VariationalSolver<S, T, N, SP, OdeType, Derived>::VariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, Args&&... extra) : Base(VariationalOdeSys<T, N, OdeType>(ode, q0.size()), t0,
    !q0.data() || !delta_q0.data() ?
    View1D<T, 2*N>{nullptr, 2*q0.size()} :
    View1D<T, 2*N>{
        join_arrays(q0, delta_q0).data(),
        2*q0.size()
    }, rtol, atol, min_step, max_step, stepsize, dir, std::forward<Args>(extra)...), worker(4*q0.size()), tmp_state_(2*q0.size()), period_(period), t_next_(t0+period*dir), t_last_(t0) {

    if (period <= 0){
        throw std::runtime_error("The renormalization period must be positive");
    }

    if constexpr (is_rich<SP>){
        //make sure there are no masked events, as they would interfere with the renormalization times.
        for (size_t i=0; i<this->event_col().size(); i++){
            if (this->event_col().event(i).is_masked()){
                throw std::runtime_error("VariationalSolver does not support masked events, as they would interfere with the renormalization times.");
            }
        }
    }
    ndspan::copy_array(tmp_state_.data(), this->ics().vector(), 2*q0.size());

}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::elapsed_time() const{
    return this->t() - this->ics_ptr()[0];
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::stretching_number() const{
    const size_t nsys = this->nsys()/2;
    return log(norm(this->true_state_ptr()+2+nsys, nsys));
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::kick() const{
    return stretching_number()/(this->t() - t_last_);
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::period() const{
    return period_;
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::log_ksi() const{
    return logksi_ + this->stretching_number();
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
T VariationalSolver<S, T, N, SP, OdeType, Derived>::lyapunov_exponent() const{
    return np == 0 ? T{0} : T(log_ksi()/elapsed_time());
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<S, T, N, SP, OdeType, Derived>::Reset(){
    Base::Reset();
    ndspan::copy_array(tmp_state_.data(), this->ics().vector(), this->nsys());
    t_last_ = this->ics_ptr()[0];
    t_next_ = t_last_ + period_*this->direction();
    np = 0;
    flagged = false;
    logksi_ =  0;
    logksi_last_ = 0;
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<S, T, N, SP, OdeType, Derived>::RhsMain(T* out, const T& t, const T* q) const{
    this->ode().ode().Rhs(out, t, q); //fills the first half (nsys) entries
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<S, T, N, SP, OdeType, Derived>::JacMain(T* out, const T& t, const T* q) const{
    if constexpr (hasJacFunc<OdeType, T>){
        this->ode().ode().Jac(out, t, q);
        return;
    } else {
        jac_approx<T>([this](T* out_, const T& t_, const T* q_){
            this->RhsMain(out_, t_, q_);
        }, out, worker.data(), t, q, nullptr, this->atol(), this->nsys()/2);
    }
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void VariationalSolver<S, T, N, SP, OdeType, Derived>::ReAdjust(const T* /*new_vector*/){
    assert(false && "ReAdjust is not supported in VariationalSolver because it would interfere with the renormalization process. If you need to re-adjust the state at intermediate times, consider using a different solver or implementing a custom solution.");
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
template<typename... Args>
bool VariationalSolver<S, T, N, SP, OdeType, Derived>::Adv_Impl(Args&&... args) {
    if (flagged){
        Base::ReAdjust(tmp_state_.data());
        flagged = false;
    }

    const int d = this->direction();
    const bool success = Base::Adv_Impl(t_next_, std::forward<Args>(args)...);
    if (success && (this->t() == t_next_)){
        const size_t nsys = this->nsys()/2;
        t_last_ = t_next_;
        t_next_ = this->ics_ptr()[0] + (++np + 1UL)*period_*d;
        ndspan::copy_array(tmp_state_.data(), THIS->true_state_ptr()+2, 2*nsys);
        logksi_last_ = logksi_;
        logksi_ += log(norm(tmp_state_.data()+nsys, nsys));
        detail::normalized(tmp_state_.data(), tmp_state_.data(), nsys);
        flagged = true;
        return true;
    } else if (success){
        return true;
    } else {
        return false;
    }
}

template<Stepper S, typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Array1D<T, 2*N> VariationalSolver<S, T, N, SP, OdeType, Derived>::join_arrays(View1D<T, N> q0, View1D<T, N> delta_q0){
    assert(q0.size() == delta_q0.size() && "q0 and delta_q0 must have the same size");
    Array1D<T, 2*N> tmp(2*q0.size());
    ndspan::copy_array(tmp.data(), q0.data(), q0.size());
    ndspan::copy_array(tmp.data()+q0.size(), delta_q0.data(), delta_q0.size());
    detail::normalized(tmp.data(), tmp.data(), q0.size());
    return tmp;
}


namespace detail{

template<typename T>
void normalized(T* out, const T* src, size_t nsys){
    T N = norm(src+nsys, nsys);
    for (size_t i=0; i<nsys; i++){
        out[i] = src[i];
        out[i+nsys] /= N;
    }
}

} // namespace ode::detail


template<typename T, size_t N>
template<hasRhsFunc<T> OdeType>
VariationalODE<T, N>::VariationalODE(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int dir, EventList<T> events, Stepper method) : Base(2*q0.size()){
    assert(q0.size() == delta_q0.size() && "q0 and delta_q0 must have the same size in VariationalODE");
    // Must create solver BEFORE register_state(), since it accesses solver_
    this->solver_ = make_variational_solver<UtilPolicy::RichVirtual>(method, ode, t0, q0, delta_q0, period, rtol, atol, min_step, max_step, stepsize, dir, std::move(events));

    const EventCollection<T>& event_coll = this->solver()->get_event_col();

    this->cached_idx_.resize(event_coll.size(), 0);
    Base::register_state();
    for (size_t i=0; i<event_coll.size(); i++){
        this->event_data_.allocate_event(event_coll.event(i).name());
    }
}

template<typename T, size_t N>
std::unique_ptr<ODE<T, N>> VariationalODE<T, N>::clone() const{
    return std::make_unique<VariationalODE<T, N>>(*this);
}

template<typename T, size_t N>
const std::vector<T>& VariationalODE<T, N>::renorm_times() const{
    return renorm_times_;
}

template<typename T, size_t N>
const std::vector<T>& VariationalODE<T, N>::lyap_values() const{
    return lyap_values_;
}

template<typename T, size_t N>
const std::vector<T>& VariationalODE<T, N>::kick_values() const{
    return kick_values_;
}

template<typename T, size_t N>
void VariationalODE<T, N>::clear(){
    Base::clear();
    renorm_times_ = std::vector<T>{};
    lyap_values_ = std::vector<T>{};
    kick_values_ = std::vector<T>{};
}

template<typename T, size_t N>
void VariationalODE<T, N>::reset(){
    Base::reset();
    renorm_times_ = std::vector<T>{};
    lyap_values_ = std::vector<T>{};
    kick_values_ = std::vector<T>{};
}

template<typename T, size_t N>
const ChaoticSolver<T, N, UtilPolicy::RichVirtual>* VariationalODE<T, N>::solver() const {
    return static_cast<const ChaoticSolver<T, N, UtilPolicy::RichVirtual>*>(Base::solver());
}

template<typename T, size_t N>
void VariationalODE<T, N>::register_state(){
    Base::register_state();
    renorm_times_.push_back(this->solver()->get_time());
    lyap_values_.push_back(this->solver()->get_lyapunov_exponent());
    kick_values_.push_back(this->solver()->get_kick());
}


template<UtilPolicy UP, typename T, size_t N, hasRhsFunc<T> OdeType, typename... Args>
pbox::Box<ChaoticSolver<T, 2*N, UP>> make_variational_solver(Stepper method, OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, Args&&... args){

    constexpr SolverPolicy SP = UP == UtilPolicy::Virtual ? SolverPolicy::Virtual : SolverPolicy::RichVirtual;

    return choose_integrator_case<pbox::Box<ChaoticSolver<T, 2*N, UP>>>(method,
        [&]<Stepper S>(){
            using Solver = typename ::ode::detail::SolverTypeGetter<S, T, N, SP, OdeType>::type;
            return pbox::make_box<VariationalSolver<
                S,
                typename Solver::value_type,
                Solver::NSYS,
                SP,
                OdeType>>(std::move(ode), t0, q0, delta_q0, period, std::forward<Args>(args)...);
        }
    );
}


template<SolverPolicy SP, Stepper S, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (!is_rich<SP>)
auto getVariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int direction){
    return VariationalSolver<S, T, N, SP, OdeType, void>(std::move(ode), t0, q0, delta_q0, period, rtol, atol, min_step, max_step, stepsize, direction);
}


template<SolverPolicy SP, Stepper S, typename T, size_t N, hasRhsFunc<T> OdeType>
requires (is_rich<SP>)
auto getVariationalSolver(OdeType ode, T t0, View1D<T, N> q0, View1D<T, N> delta_q0, T period, T rtol, T atol, T min_step, T max_step, T stepsize, int direction, EventList<T> events){
    return VariationalSolver<S, T, N, SP, OdeType, void>(std::move(ode), t0, q0, delta_q0, period, rtol, atol, min_step, max_step, stepsize, direction, std::move(events));
}

} // namespace ode

#endif // ODECRAFT_VARIATIONAL_SOLVERS_IMPL_HPP
