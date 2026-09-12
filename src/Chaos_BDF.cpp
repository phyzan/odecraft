#include <odecraft/Compiled/Chaos.hpp>

// The implementations this unit instantiates. _impl.hpp files do not include one
// another, so the translation unit names every layer: the CRTP bases, the stepper the
// variational solver is built on, and the variational layer itself. Only this one
// stepper is needed -- the other five are extern-declared in Compiled/Chaos.hpp.
#include <odecraft/Core/BaseSolver/BaseSolver_impl.hpp>
#include <odecraft/Core/RichSolver/RichBase_impl.hpp>
#include <odecraft/Steppers/BDF_impl.hpp>
#include <odecraft/Chaos/VariationalSolvers_impl.hpp>

// Variational BDF, for every scalar type. One unit per stepper: each one compiles the whole
// BDF algorithm again for the augmented 2N system, so keeping them apart lets that work
// run in parallel instead of serialising into a single translation unit.

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_DEFINE_VARIATIONAL, ::ode::Stepper::BDF, BDF)
