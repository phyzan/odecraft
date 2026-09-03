#ifndef ODECRAFT_COMPILED_STEPPERS_HPP
#define ODECRAFT_COMPILED_STEPPERS_HPP

#include "odecraft/Core/SolverFactory.hpp"
#include <odecraft/Core/VirtualBase.hpp>


namespace ode::crafted{


using ::ode::Stepper;

template<typename T>
using BoxedInterp = ::ode::BoxedInterp<T, 0>;

template<typename T>
using OdeRichSolver = ::ode::OdeRichSolver<T, 0>;



} // namespace ode::crafted

#endif // ODECRAFT_COMPILED_STEPPERS_HPP
