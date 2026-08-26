#ifndef ODECRAFT_VECTOR_FIELDS_IMPL_HPP
#define ODECRAFT_VECTOR_FIELDS_IMPL_HPP


#include <odecraft/Interpolation/VectorFields.hpp>

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
pbox::Box<OdeResult<double, NDIM>> VectorField<Derived, NDIM, AS_VIRTUAL>::streamline(const double* x0, double length, double rtol, double atol, double min_step, double max_step, double stepsize, int direction, Integrator method, bool normalized, const std::vector<double>& t_eval) const{
    pbox::Box<ODE<double, NDIM>> ode = this->get_streamline_ode(x0, rtol, atol, min_step, max_step, stepsize, direction, method, normalized);
    pbox::Box<OdeResult<double, NDIM>> result = pbox::make_box<OdeResult<double, NDIM>>();
    ode->integrate_until(result.get_raw_pointer(), length, t_eval);
    return result;
}

template<typename Derived, int NDIM, bool AS_VIRTUAL>
pbox::Box<OdeResult<double, NDIM>> VectorField<Derived, NDIM, AS_VIRTUAL>::streamline(const double* x0, double length, double rtol, double atol, double min_step, double max_step, double stepsize, int direction, Integrator method, bool normalized) const{
    pbox::Box<ODE<double, NDIM>> ode = this->get_streamline_ode(x0, rtol, atol, min_step, max_step, stepsize, direction, method, normalized);
    pbox::Box<OdeResult<double, NDIM>> result = pbox::make_box<OdeResult<double, NDIM>>();
    ode->integrate_until(result.get_raw_pointer(), length);
    return result;
}


template<typename Derived, int NDIM, bool AS_VIRTUAL>
pbox::Box<ODE<double, NDIM>> VectorField<Derived, NDIM, AS_VIRTUAL>::get_streamline_ode(const double* x0, double rtol, double atol, double min_step, double max_step, double stepsize, int direction, Integrator method, bool normalized) const{
    if (normalized){
        return pbox::make_box<ODE<double, NDIM>>(
            OdeData{
                .Rhs=[this](double* out, double t, const double* q){
                    THIS->OdeFuncNorm(out, t, q); 
                }},
                0.0, View1D<double, NDIM>{x0, this->ndim()}, rtol, atol, min_step, max_step, stepsize, direction, EventList<double>{}, method
            );
    } else {
        return pbox::make_box<ODE<double, NDIM>>(
            OdeData{
                .Rhs=[this](double* out, const double& t, const double* q){
                    THIS->OdeFunc(out, t, q);
                }},
                0.0, View1D<double, NDIM>{x0, this->ndim()}, rtol, atol, min_step, max_step, stepsize, direction, EventList<double>{}, method
            );
    }
}

} // namespace ode::interp

#endif // ODECRAFT_VECTOR_FIELDS_IMPL_HPP