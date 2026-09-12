#include <odecraft/Compiled/Steppers.hpp>


namespace ode::crafted{

template<typename T>
BoxedRichSolver<T> make_rich_vsolver(
    Stepper method,
    ode_t<T> ode,
    T t0,
    View1D<T> q0,
    T rtol,
    T atol,
    T min_step,
    T max_step,
    T stepsize,
    int dir,
    EventList<T> events
){
    return ::ode::make_rich_vsolver<T, 0>(method, std::move(ode), std::move(t0), q0, std::move(rtol), std::move(atol), std::move(min_step), std::move(max_step), std::move(stepsize), dir, std::move(events));
}


#define ODECRAFT_INSTANTIATE_SOLVER_FACTORY(T)                                  \
    template BoxedRichSolver<T> make_rich_vsolver<T>(                           \
        Stepper, ode_t<T>, T, View1D<T>, T, T, T, T, T, int, EventList<T>);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_SOLVER_FACTORY)

#undef ODECRAFT_INSTANTIATE_SOLVER_FACTORY

} // namespace ode::crafted
