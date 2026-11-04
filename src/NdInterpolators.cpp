#include <odecraft/Interpolation/MultiDim/CraftedNdInterp.hpp>

// The implementations this unit instantiates. _impl.hpp files do not include one
// another, so the translation unit names every layer it needs: the solver vocabulary of a
// streamline, the CRTP base the interpolators derive from, the grid and triangulation they
// are built on, then the interpolators themselves.
#include <odecraft/Interpolation/Univariate/StateInterp_impl.hpp>
#include <odecraft/Events/Events_impl.hpp>
#include <odecraft/OdeResult/OdeResult_impl.hpp>
#include <odecraft/DenseOde/OdeInt_impl.hpp>
#include <odecraft/Interpolation/MultiDim/NdInterpolator_impl.hpp>
#include <odecraft/Interpolation/MultiDim/Regular/Grids_impl.hpp>
#include <odecraft/Interpolation/MultiDim/Regular/RegularGridInterpolator_impl.hpp>
#include <odecraft/Interpolation/MultiDim/Scattered/Delaunay_impl.hpp>
#include <odecraft/Interpolation/MultiDim/Scattered/ScatteredNdInterpolator_impl.hpp>

ODECRAFT_ND_STREAMLINE_SET(template)
ODECRAFT_ND_INTERPOLATOR_SET(template)
