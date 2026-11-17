#pragma once

#include <odecraft/Toolkit/Tools.hpp>

namespace ode::interp {

using namespace ndspan;

struct VirtualNdInterpolator {

    virtual ~VirtualNdInterpolator() = default;

    virtual int ndim() const = 0;

    virtual int nvals_per_point() const = 0;

    virtual bool interp(double* out, const double* coords) const = 0;

    virtual const size_t* output_shape() const = 0;

    virtual size_t output_dims() const = 0;

}; // class VirtualNdInterpolator


struct EmptyNdInterpolator {};

/// @brief The base of NdInterpolator: it derives from the interface an interpolator is reached
/// through (VirtualNdInterpolator, or an empty class), and holds the defaults of the methods every
/// interpolator must implement.
template<typename T, bool AS_VIRTUAL>
class InterpolatorHook : public std::conditional_t<AS_VIRTUAL, VirtualNdInterpolator, EmptyNdInterpolator> {

protected:

    auto            interp_impl(T* /*out*/, const T* /*coords*/) const { static_assert(sizeof(T) == 0, "InterpolatorHook::interp_impl not implemented"); }

    auto            ndim_impl() const { static_assert(sizeof(T) == 0, "InterpolatorHook::ndim_impl not implemented"); }

    auto            contains_impl(const T* /*coords*/) const { static_assert(sizeof(T) == 0, "InterpolatorHook::contains_impl not implemented"); }
};


// Derived classes implement interp_impl, ndim_impl and contains_impl. Methods follow the naming
// contract in Core/BaseSolver/BaseSolver.hpp.
template<typename Derived, typename T, int NDIM, bool AS_VIRTUAL = false>
class NdInterpolator : public InterpolatorHook<T, AS_VIRTUAL> {


public:

    static constexpr Allocation ALLOC = (NDIM == 0 ? Allocation::Heap : Allocation::Stack);

    inline int nvals_per_point() const { return int(field_.size() / n_points()); }

    inline int n_points() const { return field_.shape(0); }

    inline const Array<T>& values() const { return field_; }

    inline const size_t* output_shape() const { return field_.shape()+1; }

    inline size_t output_dims() const {return field_.ndim()-1;}

    bool norm_interp(T* out, const T* coords) const;

    bool            interp(T* out, const T* coords) const;

    int             ndim() const;

    bool            contains(const T* coords) const;

protected:

    template<typename ArrayType>
    NdInterpolator(const ArrayType& values, bool coord_axis_first);

private:
    
    Array<T> field_; // shape = (n_points, ...)
};

} // namespace ode::interp
