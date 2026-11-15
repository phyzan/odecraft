#pragma once

#include "Euler/Euler.hpp" // IWYU pragma: keep
#include <odecraft/Steppers/DOPRI/RK23/RK23_DOPRI.hpp> // IWYU pragma: keep
#include <odecraft/Steppers/DOPRI/RK45/RK45_DOPRI.hpp> // IWYU pragma: keep
#include <odecraft/Steppers/DOPRI/DOP853/DOP853.hpp> // IWYU pragma: keep
#include "RK4/RK4.hpp" // IWYU pragma: keep
#include "BDF/BDF.hpp" // IWYU pragma: keep


namespace ode{

template<typename... Extras>
using StepperCatalog = Catalog<tags::Euler, tags::RK4, tags::RK23, tags::RK45, tags::DOP853, tags::BDF, Extras...>;

/**
 * @brief The catalog of the built-in steppers, for choosing one at runtime.
 *
 * Steppers::id<tags::RK45> is the runtime id of RK45, Steppers::getID("RK45") finds it by name,
 * and Steppers::make / Steppers::from_stepper turn an id back into the stepper. A catalog with
 * steppers of your own is declared the same way, e.g. Catalog<tags::RK45, tags::BDF, tags::MyStepper>.
 */
using Steppers = StepperCatalog<>;

} // namespace ode
