#include <odecraft/Steppers/DOPRI/DOP853/CraftedDOP853.hpp>

// The implementations this unit instantiates. _impl.hpp files do not include one
// another, so the translation unit names every layer it needs: the CRTP bases the
// stepper derives from, then the stepper itself.
#include <odecraft/Interpolation/Univariate/StateInterp_impl.hpp>
#include <odecraft/Core/BaseSolver/BaseSolver_impl.hpp>
#include <odecraft/Core/RichSolver/RichBase_impl.hpp>
#include <odecraft/Steppers/DOPRI/DOPRI_impl.hpp>
#include <odecraft/Steppers/DOPRI/DOP853/DOP853_impl.hpp>

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_DEFINE_STEPPER, DOP853)
