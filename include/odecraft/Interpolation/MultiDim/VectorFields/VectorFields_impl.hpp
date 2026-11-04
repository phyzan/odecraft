#pragma once


#include "VectorFields.hpp"

namespace ode::interp {

template<typename Derived, int NDIM, bool AS_VIRTUAL>
bool VectorField<Derived, NDIM, AS_VIRTUAL>::interp(double* out, const double* coords) const{
    return THIS->interp(out, coords);
}

template<typename Derived, int NDIM, bool AS_VIRTUAL>
int VectorField<Derived, NDIM, AS_VIRTUAL>::ndim() const {
    return THIS->ndim();
}

template<typename Derived, int NDIM, bool AS_VIRTUAL>
bool VectorField<Derived, NDIM, AS_VIRTUAL>::contains(const double* coords) const{
    return THIS->contains(coords);
}

template<typename Derived, int NDIM, bool AS_VIRTUAL>
void VectorField<Derived, NDIM, AS_VIRTUAL>::OdeFuncNorm(double* out, double /*t*/, const double* q) const{
    size_t nd = this->ndim();
    if (!this->interp(out, q)){
        std::fill(out, out + nd, 0);
        return;
    }
    double norm = 0;
    for (size_t i = 0; i < nd; i++) {
        norm += out[i] * out[i];
    }
    norm = sqrt(norm);
    for (size_t i = 0; i < nd; i++) {
        out[i] /= norm;
    }
}


template<typename Derived, int NDIM, bool AS_VIRTUAL>
void VectorField<Derived, NDIM, AS_VIRTUAL>::OdeFunc(double* out, double /*t*/, const double* q) const{
    size_t nd = this->ndim();
    if (!this->interp(out, q)){
        std::fill(out, out + nd, 0);
    }
}


template<typename Derived, int NDIM, bool AS_VIRTUAL>
pbox::Box<OdeResult<double, FieldSD<NDIM>>> VectorField<Derived, NDIM, AS_VIRTUAL>::streamline(
    const double* x0,
    double length,
    double rtol,
    double atol,
    double min_step,
    double max_step,
    double stepsize,
    int direction,
    Stepper method,
    bool normalized,
    const std::vector<double>& t_eval) const{

    pbox::Box<ODE<double, FieldSD<NDIM>>> ode = this->get_streamline_ode(x0, rtol, atol, min_step, max_step, stepsize, direction, method, normalized);
    pbox::Box<OdeResult<double, FieldSD<NDIM>>> result = pbox::make_box<OdeResult<double, FieldSD<NDIM>>>();
    ode->integrate(
        result.get_raw_pointer(),
        length,
        {},
        nullptr,
        t_eval
    );
    return result;
}

template<typename Derived, int NDIM, bool AS_VIRTUAL>
pbox::Box<OdeResult<double, FieldSD<NDIM>>> VectorField<Derived, NDIM, AS_VIRTUAL>::streamline(
    const double* x0,
    double length,
    double rtol,
    double atol,
    double min_step,
    double max_step,
    double stepsize,
    int direction,
    Stepper method,
    bool normalized) const{

    pbox::Box<ODE<double, FieldSD<NDIM>>> ode = this->get_streamline_ode(x0, rtol, atol, min_step, max_step, stepsize, direction, method, normalized);
    pbox::Box<OdeResult<double, FieldSD<NDIM>>> result = pbox::make_box<OdeResult<double, FieldSD<NDIM>>>();
    ode->integrate(
        result.get_raw_pointer(),
        length
    );
    return result;
}


template<typename Derived, int NDIM, bool AS_VIRTUAL>
pbox::Box<ODE<double, FieldSD<NDIM>>> VectorField<Derived, NDIM, AS_VIRTUAL>::get_streamline_ode(
    const double* x0,
    double rtol,
    double atol,
    double min_step,
    double max_step,
    double stepsize,
    int direction,
    Stepper method,
    bool normalized) const{

    using State = FieldState<NDIM>;
    const size_t nd = this->ndim();
    State q0(x0, nd);
    State rtol_state = detail::uniform_state<NDIM>(nd, rtol);
    State atol_state = detail::uniform_state<NDIM>(nd, atol);

    if (normalized){
        return pbox::make_box<ODE<double, FieldSD<NDIM>>>(
            OdeData{
                .Rhs=[this](State& out, double t, const State& q){
                    THIS->OdeFuncNorm(out.data(), t, q.data());
                },},
                0.0, std::move(q0), std::move(rtol_state), std::move(atol_state), min_step, max_step, stepsize, direction, EventList<double, State>{}, method
            );
    } else {
        return pbox::make_box<ODE<double, FieldSD<NDIM>>>(
            OdeData{
                .Rhs=[this](State& out, double t, const State& q){
                    THIS->OdeFunc(out.data(), t, q.data());
                }},
                0.0, std::move(q0), std::move(rtol_state), std::move(atol_state), min_step, max_step, stepsize, direction, EventList<double, State>{}, method
            );
    }
}

} // namespace ode::interp
