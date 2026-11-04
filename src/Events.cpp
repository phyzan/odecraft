#include <odecraft/Events/CraftedEvents.hpp>
#include <odecraft/Events/Events_impl.hpp>


namespace ode::crafted{

template<typename T>
BoxedEvent<T> make_precise_event(
    std::string name,
    objfun_t<T> obj_fun,
    T event_tol,
    int dir,
    rhs_t<T> mask,
    bool delay_mask
){
    return ::ode::make_precise_event<T, State<T>>(std::move(name), std::move(obj_fun), std::move(event_tol), dir, std::move(mask), delay_mask);
}


template<typename T>
BoxedEvent<T> make_periodic_event(
    std::string name,
    T period,
    rhs_t<T> mask,
    bool delay_mask
){
    return ::ode::make_periodic_event<T, State<T>>(std::move(name), std::move(period), std::move(mask), delay_mask);
}


template<typename T>
BoxedEvent<T, VarState<T>> make_variational_precise_event(
    std::string name,
    objfun_t<T, VarState<T>> obj_fun,
    T event_tol,
    int dir,
    rhs_t<T, VarState<T>> mask,
    bool delay_mask
){
    return ::ode::make_precise_event<T, VarState<T>>(std::move(name), std::move(obj_fun), std::move(event_tol), dir, std::move(mask), delay_mask);
}


template<typename T>
BoxedEvent<T, VarState<T>> make_variational_periodic_event(
    std::string name,
    T period,
    rhs_t<T, VarState<T>> mask,
    bool delay_mask
){
    return ::ode::make_periodic_event<T, VarState<T>>(std::move(name), std::move(period), std::move(mask), delay_mask);
}

} // namespace ode::crafted


namespace ode{

#define ODECRAFT_INSTANTIATE_EVENTS_OF(T, S)                                                \
    template class PreciseEvent<T, S, objfun_t<T, S>, rhs_t<T, S>, EventPolicy::Virtual>;   \
    template class PeriodicEvent<T, S, rhs_t<T, S>, EventPolicy::Virtual>;                  \
    template class EventCollection<T, S>;

#define ODECRAFT_INSTANTIATE_EVENTS(T)                      \
    ODECRAFT_INSTANTIATE_EVENTS_OF(T, crafted::State<T>)    \
    ODECRAFT_INSTANTIATE_EVENTS_OF(T, crafted::VarState<T>)

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_EVENTS)

#undef ODECRAFT_INSTANTIATE_EVENTS
#undef ODECRAFT_INSTANTIATE_EVENTS_OF

} // namespace ode


namespace ode::crafted{

#define ODECRAFT_INSTANTIATE_EVENT_MAKERS(T)                                                            \
    template BoxedEvent<T> make_precise_event<T>(std::string, objfun_t<T>, T, int, rhs_t<T>, bool);     \
    template BoxedEvent<T> make_periodic_event<T>(std::string, T, rhs_t<T>, bool);                      \
    template BoxedEvent<T, VarState<T>> make_variational_precise_event<T>(                              \
        std::string, objfun_t<T, VarState<T>>, T, int, rhs_t<T, VarState<T>>, bool);                    \
    template BoxedEvent<T, VarState<T>> make_variational_periodic_event<T>(                             \
        std::string, T, rhs_t<T, VarState<T>>, bool);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_INSTANTIATE_EVENT_MAKERS)

#undef ODECRAFT_INSTANTIATE_EVENT_MAKERS

} // namespace ode::crafted
