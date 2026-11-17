#pragma once


#include "Delaunay.hpp"
#include "../NdInterpolator.hpp"
#include <odecraft/Interpolation/MultiDim/VectorFields/VectorFields.hpp>

namespace ode::interp::sci {

template<int NDIM, bool AS_VIRTUAL = false>
class ScatteredNdInterpolator : public NdInterpolator<ScatteredNdInterpolator<NDIM, AS_VIRTUAL>, double, NDIM, AS_VIRTUAL>{

    using Base = NdInterpolator<ScatteredNdInterpolator<NDIM, AS_VIRTUAL>, double, NDIM, AS_VIRTUAL>;
    friend Base;

public:

    // points: shape (n_points, ndim)
    // values: shape (n_points, ...) if coord_axis_first, or (..., n_points) otherwise
    template<typename ValuesContainer>
    ScatteredNdInterpolator(const double* points, const ValuesContainer& values, int ndim, bool coord_axis_first);

    template<typename ValuesContainer>
    ScatteredNdInterpolator(TriPtr<NDIM> tri, const ValuesContainer& values, bool coord_axis_first);

    TriPtr<NDIM> tri() const;

    // shape (n_points, ndim)
    const CArray2D<double, 0, NDIM>& points() const;

protected:

    //========= Static Overrides ==============
    int             ndim_impl() const;
    bool            interp_impl(double* out, const double* coords) const;
    bool            contains_impl(const double* coords) const;
    //=========================================

private:

    TriPtr<NDIM> tri_;

}; // ScatteredNdInterpolator



template<int NDIM, bool AS_VIRTUAL>
class ScatteredVectorField : public ScatteredNdInterpolator<NDIM, AS_VIRTUAL>, public VectorField<ScatteredVectorField<NDIM, AS_VIRTUAL>, NDIM, AS_VIRTUAL>{

    using InterpBase = ScatteredNdInterpolator<NDIM, AS_VIRTUAL>;
    using VFBase = VectorField<ScatteredVectorField<NDIM, AS_VIRTUAL>, NDIM, AS_VIRTUAL>;
    friend VFBase;

public:

    // Both bases have these, so they are redeclared here, as the interpolator's
    bool interp(double* out, const double* coords) const;
    int ndim() const;
    bool contains(const double* coords) const;

    // points: (n_points, ndim)
    // values: (n_points, ndim) if coord_axis_first, else (ndim, npoints)
    template<typename ValuesContainer>
    ScatteredVectorField(const double* points, const ValuesContainer& values, int ndim, bool coord_axis_first);

    template<typename ValuesContainer>
    ScatteredVectorField(const TriPtr<NDIM>& tri, const ValuesContainer& values, bool coord_axis_first);

protected:

    // ======= Static overrides for VectorField: the interpolator's ======
    bool interp_impl(double* out, const double* coords) const;
    int ndim_impl() const;
    bool contains_impl(const double* coords) const;
    // ==================================================================

};

} // namespace ode::interp::sci
