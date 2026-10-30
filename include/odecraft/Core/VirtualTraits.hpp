#ifndef ODECRAFT_VIRTUAL_TRAITS_HPP
#define ODECRAFT_VIRTUAL_TRAITS_HPP

#include <odecraft/Core/SolverFactory.hpp> // IWYU pragma: keep

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


template<typename Solver, typename T, typename SD, SolverPolicy SP>
using SolverCloneType = std::conditional_t<SP==SolverPolicy::Virtual || SP==SolverPolicy::RichVirtual, OdeSolver<T, SD>, Solver>;


/// @brief Traits struct mapping a solver type to its virtual base interface.
/// Specialize this before the solver class definition to override the default.
template<typename Derived, typename T, typename SD, SolverPolicy SP>
struct SolverVirtualTypeTraits {
    using type = BaseInterface<T, SD, SP>;
};

} // namespace ode::traits

} // namespace ode

#endif // ODECRAFT_VIRTUAL_TRAITS_HPP