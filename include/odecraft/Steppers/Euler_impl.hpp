#ifndef ODECRAFT_EULER_IMPL_HPP
#define ODECRAFT_EULER_IMPL_HPP

#include <odecraft/Steppers/Euler.hpp>

namespace ode{


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Euler<T, N, SP, OdeType, Derived>::Euler(OdeType ode, T t0, View1D<T, N> q0, T stepsize, int dir, EventList<T> events) : Base(ode, t0, q0, 0, 0, 0, 0, stepsize, dir, std::move(events)) {}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Euler<T, N, SP, OdeType, Derived>::Euler(OdeType ode, T t0, View1D<T, N> q0, T /*rtol*/, T /*atol*/, T /*min_step*/, T /*max_step*/, T stepsize, int dir, EventList<T> events) : Euler(ode, t0, q0, stepsize, dir, std::move(events)) {}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Stepper Euler<T, N, SP, OdeType, Derived>::method() const{
    return Stepper::Euler;
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void Euler<T, N, SP, OdeType, Derived>::interp_impl(T* result, const T& t) const{
    return lin_interp(result, t, this->t_old(), this->t_new(), this->vector_old().data(), this->vector_new().data(), this->nsys());
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
StepResult Euler<T, N, SP, OdeType, Derived>::adapt_impl(T& t_new, T& habs_new, T* q_new, const T& t, const T& habs, const T* q){
    const T h = habs * this->direction();
    t_new = t + h;
    habs_new = habs;

    this->rhs(q_new, t, q);
    for (size_t i=0; i<this->nsys(); i++){
        // y2 = y1 + f*dt, f = dy/dt
        q_new[i] = q[i] + q_new[i]*h;
    }
    if (!all_are_finite(q_new, this->nsys())){
        return StepResult::NonFiniteError;
    }else{
        return StepResult::Success;
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
auto Euler<T, N, SP, OdeType, Derived>::local_interp() const{
    return
        [t1=this->t_old(),
            t2=this->t_new(),
            y1=Array1D<T, N>(this->vector_old().data(), this->nsys()),
            y2=Array1D<T, N>(this->vector_new().data(), this->nsys()),
            n=this->nsys()]
            (T* out, const T& t){
                lin_interp(out, t, t1, t2, y1.data(), y2.data(), n);
            };
}


} // namespace ode

#endif // ODECRAFT_EULER_IMPL_HPP