#pragma once

/**
 * @file renamed_stepper.hpp
 * @brief A stepper declared outside the library, the way a user declares one of their own.
 *
 * It is RK45 under a name of its own: what is tested is how a stepper joins the tags, catalogs
 * and factories, not a numerical method.
 */

#include <odecraft/odecraft.hpp>

namespace odetest {

template<typename T, typename SD, typename OdeType, ode::SolverPolicy SP, typename Derived = void>
class RenamedRK45 : public ode::RK45<T, SD, OdeType, SP, ode::GetDerived<RenamedRK45<T, SD, OdeType, SP, Derived>, Derived>> {
    ODECRAFT_DERIVED_STEPPER;
    using Base = ode::RK45<T, SD, OdeType, SP, ode::GetDerived<RenamedRK45<T, SD, OdeType, SP, Derived>, Derived>>;
public:
    using Base::Base;
};

} // namespace odetest

ODECRAFT_DECLARE_STEPPER(RenamedRK45, odetest::RenamedRK45)
