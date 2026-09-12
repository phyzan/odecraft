#include <odecraft/Compiled/RK23_DOPRI.hpp>

// The implementations this unit instantiates. _impl.hpp files do not include one another, so
// the translation unit names every layer it needs: the CRTP bases, the explicit-RK machinery
// RK23 shares with the other Dormand-Prince steppers, then RK23 itself.
#include <odecraft/Core/BaseSolver/BaseSolver_impl.hpp>
#include <odecraft/Core/RichSolver/RichBase_impl.hpp>
#include <odecraft/Steppers/DOPRI_impl.hpp>
#include <odecraft/Steppers/RK23_DOPRI_impl.hpp>

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_DEFINE_STEPPER, RK23)
