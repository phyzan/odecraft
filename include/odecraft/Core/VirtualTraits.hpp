#ifndef ODECRAFT_VIRTUAL_TRAITS_HPP
#define ODECRAFT_VIRTUAL_TRAITS_HPP

#include <odecraft/Core/SolverFactory.hpp> // IWYU pragma: keep

namespace ode{


struct EmptySolver{};


namespace traits{

template<typename T, size_t N, SolverPolicy SP>
struct HelperVirtualSolver{ using type = EmptySolver;};

template<typename T, size_t N>
struct HelperVirtualSolver<T, N, SolverPolicy::Virtual>{ using type = OdeSolver<T, N>;};

template<typename T, size_t N>
struct HelperVirtualSolver<T, N, SolverPolicy::RichVirtual>{ using type = OdeRichSolver<T, N>;};

template<typename T, size_t N, SolverPolicy SP>
using BaseInterface = typename HelperVirtualSolver<T, N, SP>::type;


template<typename Solver, typename T, size_t N, SolverPolicy SP>
using SolverCloneType = std::conditional_t<SP==SolverPolicy::Virtual || SP==SolverPolicy::RichVirtual, OdeSolver<T, N>, Solver>;


/// @brief Traits struct mapping a solver type to its virtual base interface.
/// Specialize this before the solver class definition to override the default.
template<typename Derived, typename T, size_t N, SolverPolicy SP>
struct SolverVirtualTypeTraits {
    using type = BaseInterface<T, N, SP>;
};

} // namespace ode::traits

} // namespace ode

#endif // ODECRAFT_VIRTUAL_TRAITS_HPP