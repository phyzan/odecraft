#ifndef ODECRAFT_REGULAR_GRID_INTERPOLATOR_HPP
#define ODECRAFT_REGULAR_GRID_INTERPOLATOR_HPP

#include <odecraft/Interpolation/Regular/Grids.hpp>
#include <odecraft/Interpolation/NdInterpolator.hpp>
#include <odecraft/Interpolation/VectorFields.hpp>

namespace ode::interp::rgi {


enum class CoordType : uint8_t {
    Cartesian,
    Polar,
    Spherical
};


template<int NDIM, bool AS_VIRTUAL = false>
class RegularGridInterpolator : public NdInterpolator<RegularGridInterpolator<NDIM, AS_VIRTUAL>, double, NDIM, AS_VIRTUAL>{

    /**
    Represents an N-dimensional regular grid interpolator using multilinear interpolation.
    The grid points along each dimension are provided as 1D arrays, and the field values
    are provided as an N-dimensional array.

    Only multilinear interpolation is supported at the moment.
    */

    using Base = NdInterpolator<RegularGridInterpolator<NDIM, AS_VIRTUAL>, double, NDIM, AS_VIRTUAL>;

public:

    // values must be a contiguous array of shape (n_points, nfields) if coord_axis_first, or (nfields, n_points) if not
    // where the n_points part can be reshaped to (nx, ny, ...) as a C-contiguous array
    template<typename ValuesContainer, typename AxisViewContainer>
    RegularGridInterpolator(const ValuesContainer& values, const AxisViewContainer& grid, bool coord_axis_first);

    inline const RegularGrid<double, NDIM>&  grid() const { return grid_; }

    // ============== Static Override ==================
    int             ndim() const;
    bool            interp(double* out, const double* coords) const;
    bool            contains(const double* coords) const;
    // =================================================

private:

    RegularGrid<double, NDIM> grid_;

}; // RegularGridInterpolator


template<int NDIM, bool AS_VIRTUAL = false>
class RegularVectorField : public RegularGridInterpolator<NDIM, AS_VIRTUAL>, public VectorField<RegularVectorField<NDIM, AS_VIRTUAL>, NDIM, AS_VIRTUAL>{

    using InterpBase = RegularGridInterpolator<NDIM, AS_VIRTUAL>;
    using VFBase = VectorField<RegularVectorField<NDIM, AS_VIRTUAL>, NDIM, AS_VIRTUAL>;

public:

    // overriden to support different coordinate systems (Cartesian, Polar, Spherical)
    void OdeFuncNorm(double* out, double t, const double* q) const;

    // grid[i].data(), grid[i].size() : grid points along axis i
    template<typename ValuesContainer, typename AxisViewContainer>
    RegularVectorField(const ValuesContainer& values, const AxisViewContainer& grid, CoordType coord_type, bool coord_axis_first);

    std::vector<Array2D<double, NDIM, 0>>    streamplot_data(double max_length, double ds, size_t density, double rtol, double atol, double min_step, double max_step, double stepsize, Stepper method) const;


    // ============ Explicit overrides for VectorField ==============
    bool    interp(double* out, const double* coords) const;
    int     ndim() const;
    bool    contains(const double* coords) const;
    // ==============================================================


private:

    template<size_t... I>
    std::vector<Array2D<double, NDIM, 0>>    streamplot_data_core(double max_length, double ds, size_t density, double rtol, double atol, double min_step, double max_step, double stepsize, Stepper method, std::index_sequence<I...>) const;

    CoordType coord_type_;

}; // RegularVectorField


namespace detail{

template<typename AxisViewContainer>
inline int get_point_count(const AxisViewContainer& grid){
    int count = 1;
    for (size_t i=0; i<grid.size(); i++){
        count *= grid[i].size();
    }
    return count;
}

} // namespace ode::detail

} // namespace ode::interp::rgi

#endif // ODECRAFT_REGULAR_GRID_INTERPOLATOR_HPP
