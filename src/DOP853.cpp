#include <odecraft/Compiled/DOP853.hpp>

// The implementations this unit instantiates. _impl.hpp files do not include one
// another, so the translation unit names every layer it needs: the CRTP bases the
// stepper derives from, then the stepper itself.
#include <odecraft/Core/BaseSolver/BaseSolver_impl.hpp>
#include <odecraft/Core/RichSolver/RichBase_impl.hpp>
#include <odecraft/Steppers/DOPRI_impl.hpp>
#include <odecraft/Steppers/DOP853_impl.hpp>

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_DEFINE_STEPPER, DOP853)
