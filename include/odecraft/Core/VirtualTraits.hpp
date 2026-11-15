#pragma once

#include "SolverFactory.hpp" // IWYU pragma: keep

namespace ode{


struct EmptySolver{};


namespace traits{

template<typename T, typename SD, SolverPolicy SP>
struct HelperVirtualSolver{ using type = EmptySolver;};

template<typename T, typename SD>
struct HelperVirtualSolver<T, SD, SolverPolicy::Virtual>{ using type = OdeSolver<T, SD>;};

template<typename T, typename SD>
struct HelperVirtualSolver<T, SD, SolverPolicy::RichVirtual>{ using type = OdeRichSolver<T, SD>;};

template<typename T, typename SD, SolverPolicy SP>
using BaseInterface = HelperVirtualSolver<T, SD, SP>::type;


/// @brief Traits struct mapping a solver type to its virtual base interface.
/// Specialize this before the solver class is instantiated to override the default.
///
/// BaseSolver derives from `type` while its Derived class is still incomplete, so a solver
/// cannot name its interface through a member of its own: this is the hook instead. BaseSolver
/// exposes the result as `virtual_base` (see VariationalSolver, reached through ChaoticSolver).
template<typename Derived, typename T, typename SD, SolverPolicy SP>
struct SolverVirtualTypeTraits {
    using type = BaseInterface<T, SD, SP>;
};


} // namespace ode::traits

} // namespace ode
