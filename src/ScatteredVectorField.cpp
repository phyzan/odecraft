#include <odecraft/Compiled/NdInterpolators.hpp>

// As in src/NdInterpolators.cpp, every layer is named explicitly. The stepper and driver
// implementations are needed on top of the interpolation ones: VectorField::streamline
// builds an ODE over a closure that calls back into the field, and that system type is
// compiled nowhere else, so the solvers behind it are instantiated here.
#include <odecraft/Interpolation/NdInterpolator_impl.hpp>
#include <odecraft/Interpolation/VectorFields_impl.hpp>
#include <odecraft/Core/BaseSolver/BaseSolver_impl.hpp>
#include <odecraft/Core/RichSolver/RichBase_impl.hpp>
#include <odecraft/Steppers/Steppers_impl.hpp>
#include <odecraft/DenseOde/OdeInt_impl.hpp>
#include <odecraft/Interpolation/Scattered/Delaunay_impl.hpp>
#include <odecraft/Interpolation/Scattered/ScatteredNdInterpolator_impl.hpp>

ODECRAFT_ND_SCATTERED_FIELD_SET(template)
