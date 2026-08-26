#ifndef ODECRAFT_VECTOR_FIELDS_HPP
#define ODECRAFT_VECTOR_FIELDS_HPP


#include <odecraft/DenseOde/OdeInt.hpp>

namespace ode::interp {


template<size_t NDIM>
struct VirtualVectorField {

    virtual ~VirtualVectorField() = default;

    virtual int ndim() const = 0;

    virtual bool contains(const double* coords) const = 0;
    
    virtual pbox::Box<OdeResult<double, NDIM>> streamline(const double* x0, double length, double rtol, double atol, double min_step, double max_step, double stepsize, int direction, Integrator method, bool normalized, const std::vector<double>& t_eval) const = 0;

    virtual pbox::Box<OdeResult<double, NDIM>> streamline(const double* x0, double length, double rtol, double atol, double min_step, double max_step, double stepsize, int direction, Integrator method, bool normalized) const = 0;

    virtual pbox::Box<ODE<double, NDIM>> get_streamline_ode(const double* x0, double rtol, double atol, double min_step, double max_step, double stepsize, int direction, Integrator method, bool normalized) const = 0;

}; // class VirtualVectorField

struct EmptyVectorField {};

template<typename Derived, int NDIM, bool AS_VIRTUAL>
class VectorField : public std::conditional_t<AS_VIRTUAL, VirtualVectorField<NDIM>, EmptyVectorField> {


public:

    // ============== Static Overrides ==================
    bool interp(double* out, const double* coords) const;
    int ndim() const;
    bool contains(const double* coords) const;
    // =================================================


    void OdeFuncNorm(double* out, double t, const double* q) const;

    void OdeFunc(double* out, double t, const double* q) const;

    pbox::Box<OdeResult<double, NDIM>> streamline(const double* x0, double length, double rtol, double atol, double min_step, double max_step, double stepsize, int direction, Integrator method, bool normalized, const std::vector<double>& t_eval) const;

    pbox::Box<OdeResult<double, NDIM>> streamline(const double* x0, double length, double rtol, double atol, double min_step, double max_step, double stepsize, int direction, Integrator method, bool normalized) const;

    pbox::Box<ODE<double, NDIM>> get_streamline_ode(const double* x0, double rtol, double atol, double min_step, double max_step, double stepsize, int direction, Integrator method, bool normalized) const;

protected:

    VectorField() = default;

    DEFAULT_RULE_OF_FOUR(VectorField)

}; // class VectorField

} // namespace ode::interp

#endif // ODECRAFT_VECTOR_FIELDS_HPP