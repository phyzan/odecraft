#ifndef ODECRAFT_COMPILED_EVENTS_HPP
#define ODECRAFT_COMPILED_EVENTS_HPP


#include "odecraft/Events/Events.hpp"
#include <odecraft/Events/Events_mem_impl.hpp>


namespace ode::crafted{

using   ::ode::Event,
        ::ode::BoxedEvent,
        ::ode::EventList,
        ::ode::EventOptions,
        ::ode::EventCollection,
        ::ode::EventState,
        ::ode::MaskedState,
        ::ode::make_event_list;


template<typename T>
BoxedEvent<T> make_precise_event(
    std::string name,
    objfun_t<T> obj_fun,
    T event_tol,
    int dir, 
    rhs_t<T> mask = nullptr,
    bool delay_mask = false
);

template<typename T>
BoxedEvent<T> make_periodic_event(
    std::string name,
    T period,
    rhs_t<T> mask = nullptr,
    bool delay_mask = false
);

} // namespace ode::crafted

#endif // ODECRAFT_COMPILED_EVENTS_HPP