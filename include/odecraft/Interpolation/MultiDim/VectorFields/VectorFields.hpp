#pragma once


#include <odecraft/DenseOde/OdeInt.hpp>

namespace ode::interp {

// The state of a streamline: one axis, of size NDIM, or a runtime size if NDIM = 0
template<size_t NDIM>
using FieldSD = ::ode::contracts::common::SD<NDIM>;

template<size_t NDIM>
using FieldState = state_t<double, FieldSD<NDIM>>;

namespace detail{

// A state of size ndim with every component set to value
template<size_t NDIM>
inline FieldState<NDIM> uniform_state(size_t ndim, double value){
    FieldState<NDIM> out(ndim);
    for (size_t i = 0; i < ndim; i++){
        out[i] = value;
    }
    return out;
}

} // namespace ode::interp::detail


template<size_t NDIM>
struct VirtualVectorField {

    virtual ~VirtualVectorField() = default;

    virtual int ndim() const = 0;

    virtual bool contains(const double* coords) const = 0;
    
    virtual pbox::Box<OdeResult<double, FieldSD<NDIM>>> streamline(
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
        const std::vector<double>& t_eval
    ) const = 0;

    virtual pbox::Box<OdeResult<double, FieldSD<NDIM>>> streamline(
        const double* x0,
        double length,
        double rtol,
        double atol,
        double min_step,
        double max_step,
        double stepsize,
        int direction,
        Stepper method,
        bool normalized
    ) const = 0;

    virtual pbox::Box<ODE<double, FieldSD<NDIM>>> get_streamline_ode(
        const double* x0,
        double rtol,
        double atol,
        double min_step,
        double max_step,
        double stepsize,
        int direction,
        Stepper method,
        bool normalized
    ) const = 0;

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

    pbox::Box<OdeResult<double, FieldSD<NDIM>>> streamline(
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
        const std::vector<double>& t_eval
    ) const;

    pbox::Box<OdeResult<double, FieldSD<NDIM>>> streamline(
        const double* x0,
        double length,
        double rtol,
        double atol,
        double min_step,
        double max_step,
        double stepsize,
        int direction,
        Stepper method,
        bool normalized
    ) const;

    pbox::Box<ODE<double, FieldSD<NDIM>>> get_streamline_ode(
        const double* x0,
        double rtol,
        double atol,
        double min_step,
        double max_step,
        double stepsize,
        int direction,
        Stepper method,
        bool normalized
    ) const;

protected:

    VectorField() = default;

    DEFAULT_RULE_OF_FOUR(VectorField)

}; // class VectorField

} // namespace ode::interp
