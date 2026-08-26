#ifndef ODECRAFT_RUNGEKUTTA_IMPL_HPP
#define ODECRAFT_RUNGEKUTTA_IMPL_HPP

#include <odecraft/Steppers/RungeKutta.hpp>

namespace ode{

#ifdef ODECRAFT_RK4_DENSE
namespace detail{

template<typename T, size_t N>
class MutArray{
public:
    MutArray(size_t n) : vec{n} {}

    T* data() const{
        return vec.data();
    }

private:
    mutable Vector<T, N> vec;
};

} // namespace ode::detail
#endif


template<typename T, typename RhsType>
void rk4_step(RhsType&& rhs, T* y_new, const T& t, const T& h, const T* y, T* k, size_t n, T* worker){
    // rhs(out, t, y);

    // ======= Perform RK4 core algorithm =======
    T* k1 = k;
    T* k2 = k + n;
    T* k3 = k + 2*n;
    T* k4 = k + 3*n;

    rhs(k1, t, y);
    for (size_t i=0; i<n; i++){
        worker[i] = y[i] + h * k1[i] / 2;
    }
    rhs(k2, t + h/2, worker);
    for (size_t i=0; i<n; i++){
        worker[i] = y[i] + h * k2[i] / 2;
    }
    rhs(k3, t + h/2, worker);
    for (size_t i=0; i<n; i++){
        worker[i] = y[i] + h * k3[i];
    }
    rhs(k4, t + h, worker);
    for (size_t i=0; i<n; i++){
        y_new[i] = y[i] + h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6;
    }
}

template<typename T>
void rk4_interp(T* out, const T& t, const T& t1, const T& t2, const T* y1, const T* y2, const T* y1dot, const T* y2dot, size_t n){
    T h = t2 - t1;
    T theta = (t - t1) / h;

    T x2 = theta * theta;
    T x3 = x2 * theta;
    T h00 = 1 - 3*x2 + 2*x3;
    T h10 = theta - 2*x2 + x3;
    T h01 = 3*x2 - 2*x3;
    T h11 = -x2 + x3;

    for (size_t i=0; i<n; i++){
        out[i] = h00 * y1[i] + h * h10 * y1dot[i] + h01 * y2[i] + h * h11 * y2dot[i];
    }
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
template<typename... Type>
RK4<T, N, SP, OdeType, Derived>::RK4(OdeType ode, T t0, View1D<T, N> q0, T rtol, T atol, T /*min_step*/, T /*max_step*/, T stepsize, int dir, Type&&... extras) : Base(ode, t0, q0, rtol, atol, 0, 0, stepsize, dir, std::forward<Type>(extras)...),
#ifdef ODECRAFT_RK4_DENSE
K(9, q0.size())
#else
K(5, q0.size())
#endif
{
    // min_step and max_step are not used in RK4
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
Stepper RK4<T, N, SP, OdeType, Derived>::method() const{
    return Stepper::RK4;
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
auto RK4<T, N, SP, OdeType, Derived>::local_interp() const{
#ifdef ODECRAFT_RK4_DENSE
    /*
    TODO : This is expensive. If we could guarantee that the RK4 object does not go out of scope,
    we can just reference the solver without copying this->K and this->ode()
    in the lambda below
    */
    return
        [K = detail::MutArray<T, 5*N>{
            5*this->nsys()},
        nsys = this->nsys(),
        t_old = this->t_old(),
        old = Array1D<T, N>(this->old_state_ptr()+2, this->nsys()),
        ode=this->ode()]
        (T* out, const T& t) {
            rk4_step(
                [&ode](T* out_, const T& t_, const T* y_){
                    ode.Rhs(out_, t_, y_);
                },
                out, t_old, t - t_old, old.data(), K.data(), nsys, K.data()+4*nsys
            );
        };
#else
    set_interp_data();
    const T* d = this->interp_new_state_ptr();
    size_t nsys = this->nsys();
    return [nsys, t1=this->t_old(), t2 = d[0], y1=Array1D<T, N>(this->old_state_ptr()+2, nsys), y2=Array1D<T, N>(d+2, nsys), y1dot=Array1D<T, N>(K.data(), nsys), y2dot=Array1D<T, N>(K.data()+nsys, nsys)](T* out, const T& t){
        rk4_interp(out, t, t1, t2, y1.data(), y2.data(), y1dot.data(), y2dot.data(), nsys);
    };
#endif
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
StepResult RK4<T, N, SP, OdeType, Derived>::adapt_impl(T* res, const T* state){
    // standard Runge-Kutta-4 with fixed step size

    const T& t = state[0];

    const T& habs = state[1];
    const T& h = habs * this->direction();
    const T* y = state + 2;
    size_t n = this->nsys();

    res[0] = t + h; // t_new
    res[1] = habs;
    T* y_new = res + 2;

    auto rhs_caller = [this, t, y](T* out, const T& t_dummy, const T* y_dummy) NDSPAN_LAMBDA_INLINE{
        this->rhs(out, t_dummy, y_dummy);
    };

    rk4_step(rhs_caller, y_new, t, h, y, K.data(), n, K.data() + 4*n);
    if (!all_are_finite(y_new, n)){
        return StepResult::NonFiniteError;
    }else{
        return StepResult::Success;
    }

}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK4<T, N, SP, OdeType, Derived>::interp_impl(T* out, const T& t) const{
    size_t nsys = this->nsys();
#ifdef ODECRAFT_RK4_DENSE    
    rk4_step([this](T* out_, const T& t_, const T* y_) NDSPAN_LAMBDA_INLINE{
        this->rhs(out_, t_, y_);
    }, out, this->t_old(), t - this->t_old(), this->old_state_ptr()+2, K.data()+5*nsys, nsys, K.data() + 4*nsys);
#else
    set_interp_data();
    const T* d = this->interp_new_state_ptr();
    rk4_interp(out, t, this->t_old(), d[0], this->old_state_ptr()+2, d+2, K.data(), K.data()+nsys, nsys);
#endif
}


template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK4<T, N, SP, OdeType, Derived>::Reset(){
    Base::Reset();
    K.fill(0);
    interp_data_set = false;

}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK4<T, N, SP, OdeType, Derived>::ReAdjust(const T* new_vector){
    Base::ReAdjust(new_vector);
    set_interp_data();
}

template<typename T, size_t N, SolverPolicy SP, hasRhsFunc<T> OdeType, typename Derived>
void RK4<T, N, SP, OdeType, Derived>::set_interp_data() const{
    if (!interp_data_set){
        const T* d = this->interp_new_state_ptr();
        this->rhs(K.data()+this->nsys(), d[0], d+2);
        interp_data_set = true;
    }
}

} // namespace ode

#endif // ODECRAFT_RUNGEKUTTA_IMPL_HPP