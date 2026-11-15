#include <odecraft/Chaos/CraftedChaos.hpp>
#include <odecraft/Chaos/VariationalSolvers_impl.hpp>

// The driver and the factory. Every individual variational stepper is instantiated in its own
// Chaos_<stepper>.cpp and reached from here through the extern template declarations in
// Chaos/CraftedChaos.hpp, so this unit stays small.

namespace ode::crafted{

template<typename T>
BoxedChaoticSolver<T> make_variational_solver(
    Steppers::Id method,
    ode_t<T> ode,
    T t0,
    VarState<T> q0,
    T period,
    VarState<T> rtol,
    VarState<T> atol,
    T min_step,
    T max_step,
    T stepsize,
    int dir,
    EventList<T, VarState<T>> events
){
    return Steppers::from_stepper<BoxedChaoticSolver<T>>(method, [&]<typename S>(){
        return ::ode::chaos::make_variational_solver<::ode::UtilPolicy::RichVirtual, SD>(
            S{}, std::move(ode), std::move(t0), std::move(q0), std::move(period), std::move(rtol), std::move(atol), std::move(min_step), std::move(max_step), std::move(stepsize), dir, std::move(events));
    });
}

#define ODECRAFT_INSTANTIATE_CHAOS_FACTORY(T)                           \
    template BoxedChaoticSolver<T> make_variational_solver<T>(          \
        Steppers::Id, ode_t<T>, T, VarState<T>, T, VarState<T>, VarState<T>, T, T, T, int, EventList<T, VarState<T>>);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_CHAOS_FACTORY)

#undef ODECRAFT_INSTANTIATE_CHAOS_FACTORY

} // namespace ode::crafted


namespace ode::chaos{

#define ODECRAFT_INSTANTIATE_VARIATIONAL_ODE(T)                                     \
    template class VariationalODE<T, crafted::SD>;                                  \
    ODECRAFT_FOR_EACH_STEPPER(ODECRAFT_VARIATIONAL_ODE_CTOR, template, T)

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_VARIATIONAL_ODE)

#undef ODECRAFT_INSTANTIATE_VARIATIONAL_ODE

} // namespace ode::chaos
