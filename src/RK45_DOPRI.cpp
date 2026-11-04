#include <odecraft/Steppers/DOPRI/RK45/CraftedRK45.hpp>

// The implementations this unit instantiates. _impl.hpp files do not include one another, so
// the translation unit names every layer it needs: the CRTP bases, the explicit-RK machinery
// RK45 shares with the other Dormand-Prince steppers, then RK45 itself.
#include <odecraft/Interpolation/Univariate/StateInterp_impl.hpp>
#include <odecraft/Core/BaseSolver/BaseSolver_impl.hpp>
#include <odecraft/Core/RichSolver/RichBase_impl.hpp>
#include <odecraft/Steppers/DOPRI/DOPRI_impl.hpp>
#include <odecraft/Steppers/DOPRI/RK45/RK45_DOPRI_impl.hpp>

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_DEFINE_STEPPER, RK45)
