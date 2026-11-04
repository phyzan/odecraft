#include <odecraft/Chaos/CraftedChaos.hpp>

// The implementations this unit instantiates. _impl.hpp files do not include one another, so
// the translation unit names every layer: the CRTP bases, the shared explicit-RK machinery,
// the stepper the variational solver is built on, and the variational layer itself. Only this
// one stepper is needed -- the others are extern-declared in Chaos/CraftedChaos.hpp.
#include <odecraft/Interpolation/Univariate/StateInterp_impl.hpp>
#include <odecraft/Core/BaseSolver/BaseSolver_impl.hpp>
#include <odecraft/Core/RichSolver/RichBase_impl.hpp>
#include <odecraft/Steppers/DOPRI/DOPRI_impl.hpp>
#include <odecraft/Steppers/DOPRI/RK45/RK45_DOPRI_impl.hpp>
#include <odecraft/Chaos/VariationalSolvers_impl.hpp>

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_DEFINE_VARIATIONAL, ::ode::Stepper::RK45, RK45)
