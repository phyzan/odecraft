#ifndef ODECRAFT_COMPILED_ND_INTERPOLATORS_HPP
#define ODECRAFT_COMPILED_ND_INTERPOLATORS_HPP

/**
 * @file Compiled/NdInterpolators.hpp
 * @brief Multidimensional field interpolation and vector fields.
 *        Compiled by src/NdInterpolators.cpp, src/RegularVectorField.cpp and
 *        src/ScatteredVectorField.cpp.
 *
 * Links after ODE: a vector field integrates its own streamlines, so it needs the whole
 * stepper stack behind it.
 *
 * These are the one family in the library that is *not* parameterised on the scalar type:
 * the value type is fixed to `double` throughout, and the parameter left over is the
 * dimension. So instead of ODECRAFT_FOR_EACH_SCALAR, everything here is pinned at
 * `NDIM = 0` -- dynamic dimension, decided at construction, exactly as `N = 0` means a
 * dynamic system size everywhere else -- and at `AS_VIRTUAL = true`, so every type below
 * is reachable through VirtualNdInterpolator or VirtualVectorField.
 *
 * @note The constructors are member templates over the container holding the field values.
 *       Only one container is compiled, `field_values_t`, so build one of those and pass it;
 *       see the note on that alias.
 */

#include <odecraft/Compiled/ODE.hpp>
#include <odecraft/Interpolation/NdInterpolator.hpp>
#include <odecraft/Interpolation/VectorFields.hpp>
#include <odecraft/Interpolation/Regular/Grids.hpp>
#include <odecraft/Interpolation/Regular/RegularGridInterpolator.hpp>
#include <odecraft/Interpolation/Scattered/Delaunay.hpp>
#include <odecraft/Interpolation/Scattered/ScatteredNdInterpolator.hpp>


namespace ode::crafted{

/**
 * @brief The values container every compiled constructor below takes.
 *
 * A dynamic-rank read-only view over contiguous doubles. The constructors are templated on
 * this container in namespace `ode`, and only this one instantiation is compiled, so a
 * container of any other type would be instantiated in the calling translation unit
 * instead of linked. Wrap your data in one of these and reshape it to the layout the
 * constructor documents.
 */
using field_values_t = View<double>;

/// @brief The grid container the regular constructors take: the points along each axis.
using grid_axes_t = std::vector<Array1D<double>>;


/// @brief Abstract interpolator over an N-dimensional field. What every interpolator below is.
using ::ode::interp::VirtualNdInterpolator;

/// @brief Which coordinate system a RegularVectorField's components are expressed in.
using ::ode::interp::rgi::CoordType;

/// @brief Abstract vector field: an interpolator that can also integrate its own streamlines.
using VirtualVectorField = ::ode::interp::VirtualVectorField<0>;

/// @brief The axes of a rectilinear grid.
using RegularGrid = ::ode::interp::rgi::RegularGrid<double, 0>;

/// @brief Multilinear interpolation of a field sampled on a rectilinear grid.
using RegularGridInterpolator = ::ode::interp::rgi::RegularGridInterpolator<0, true>;

/// @brief A RegularGridInterpolator whose values are vectors, so it defines a flow.
using RegularVectorField = ::ode::interp::rgi::RegularVectorField<0, true>;

/// @brief Delaunay triangulation of a scattered point cloud.
using DelaunayTri = ::ode::interp::sci::DelaunayTri<0>;

/// @brief Shared handle to a DelaunayTri; interpolators built on the same cloud share one.
using TriPtr = ::ode::interp::sci::TriPtr<0>;

/// @brief Barycentric interpolation of a field sampled at scattered points.
using ScatteredNdInterpolator = ::ode::interp::sci::ScatteredNdInterpolator<0, true>;

/// @brief A ScatteredNdInterpolator whose values are vectors, so it defines a flow.
using ScatteredVectorField = ::ode::interp::sci::ScatteredVectorField<0, true>;

} // namespace ode::crafted


/**
 * @brief Every instantiation the plain interpolators need. Compiled by src/NdInterpolators.cpp.
 *
 * @param SPEC `extern template` (here) or `template` (the src unit).
 *
 * The CRTP bases are spelled out for the same reason as in Compiled/SolverBase.hpp -- a
 * derived class's explicit instantiation does not reach members it inherits, and a consumer
 * that names one of these concretely emits its vtable under LTO regardless. Both concrete
 * interpolators are designed to be derived from, so both bases have to be here.
 *
 * VirtualVectorField is here rather than with the fields below because it is the one piece
 * the two field units share, and a specialization may only be defined in one of them.
 *
 * The constructors are listed separately because an explicit instantiation of a class does
 * not instantiate its member templates, and these constructors are templated on the values
 * container. They are written in deduction form: a constructor has no name to hang a
 * template-id on, so the argument types are spelled out and deduced from.
 */
#define ODECRAFT_ND_INTERPOLATOR_SET(SPEC)                                                          \
    SPEC struct ode::interp::VirtualVectorField<0>;                                                  \
    SPEC class ode::interp::rgi::RegularGrid<double, 0>;                                            \
    SPEC class ode::interp::NdInterpolator<                                                         \
        ode::interp::rgi::RegularGridInterpolator<0, true>, double, 0, true>;                       \
    SPEC class ode::interp::rgi::RegularGridInterpolator<0, true>;                                  \
    SPEC class ode::interp::sci::DelaunayTri<0>;                                                    \
    SPEC class ode::interp::NdInterpolator<                                                         \
        ode::interp::sci::ScatteredNdInterpolator<0, true>, double, 0, true>;                       \
    SPEC class ode::interp::sci::ScatteredNdInterpolator<0, true>;                                  \
    SPEC ode::interp::rgi::RegularGrid<double, 0>::RegularGrid(                                     \
        const ode::crafted::grid_axes_t&);                                                          \
    SPEC ode::interp::rgi::RegularGridInterpolator<0, true>::RegularGridInterpolator(               \
        const ode::crafted::field_values_t&, const ode::crafted::grid_axes_t&, bool);               \
    SPEC ode::interp::sci::ScatteredNdInterpolator<0, true>::ScatteredNdInterpolator(               \
        const double*, const ode::crafted::field_values_t&, int, bool);                             \
    SPEC ode::interp::sci::ScatteredNdInterpolator<0, true>::ScatteredNdInterpolator(               \
        ode::crafted::TriPtr, const ode::crafted::field_values_t&, bool);


/**
 * @brief The regular vector field. Compiled by src/RegularVectorField.cpp.
 *
 * @param SPEC `extern template` (here) or `template` (the src unit).
 *
 * One unit per field kind, for the same reason the variational solvers get one unit per
 * stepper: VectorField::streamline builds an ODE whose right-hand side is a closure over
 * the field, and that system type is compiled nowhere else, so the whole stepper stack is
 * instantiated again behind each field. Splitting lets the two run in parallel instead of
 * piling into one translation unit.
 *
 * The interpolator it derives from is not repeated here -- that comes from the set above.
 */
#define ODECRAFT_ND_REGULAR_FIELD_SET(SPEC)                                                         \
    SPEC class ode::interp::VectorField<                                                            \
        ode::interp::rgi::RegularVectorField<0, true>, 0, true>;                                    \
    SPEC class ode::interp::rgi::RegularVectorField<0, true>;                                       \
    SPEC ode::interp::rgi::RegularVectorField<0, true>::RegularVectorField(                         \
        const ode::crafted::field_values_t&, const ode::crafted::grid_axes_t&,                      \
        ode::interp::rgi::CoordType, bool);


/// @brief The scattered vector field. Compiled by src/ScatteredVectorField.cpp.
/// @param SPEC `extern template` (here) or `template` (the src unit).
/// @see ODECRAFT_ND_REGULAR_FIELD_SET for why this is a unit of its own.
#define ODECRAFT_ND_SCATTERED_FIELD_SET(SPEC)                                                       \
    SPEC class ode::interp::VectorField<                                                            \
        ode::interp::sci::ScatteredVectorField<0, true>, 0, true>;                                  \
    SPEC class ode::interp::sci::ScatteredVectorField<0, true>;                                     \
    SPEC ode::interp::sci::ScatteredVectorField<0, true>::ScatteredVectorField(                     \
        const double*, const ode::crafted::field_values_t&, int, bool);                             \
    SPEC ode::interp::sci::ScatteredVectorField<0, true>::ScatteredVectorField(                     \
        const ode::crafted::TriPtr&, const ode::crafted::field_values_t&, bool);


// At global scope, as in the stepper headers: an explicit instantiation must appear in a
// namespace enclosing its template's, and these span ode::interp, ode::interp::rgi and
// ode::interp::sci, so the macros spell every name out fully.
ODECRAFT_ND_INTERPOLATOR_SET(extern template)
ODECRAFT_ND_REGULAR_FIELD_SET(extern template)
ODECRAFT_ND_SCATTERED_FIELD_SET(extern template)

#endif // ODECRAFT_COMPILED_ND_INTERPOLATORS_HPP
