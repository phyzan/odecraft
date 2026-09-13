#include <odecraft/Compiled/NdInterpolators.hpp>

// The implementations this unit instantiates. _impl.hpp files do not include one
// another, so the translation unit names every layer it needs: the CRTP base the
// interpolators derive from, the grid and triangulation they are built on, then the
// interpolators themselves.
#include <odecraft/Interpolation/NdInterpolator_impl.hpp>
#include <odecraft/Interpolation/Regular/Grids_impl.hpp>
#include <odecraft/Interpolation/Regular/RegularGridInterpolator_impl.hpp>
#include <odecraft/Interpolation/Scattered/Delaunay_impl.hpp>
#include <odecraft/Interpolation/Scattered/ScatteredNdInterpolator_impl.hpp>

ODECRAFT_ND_INTERPOLATOR_SET(template)
