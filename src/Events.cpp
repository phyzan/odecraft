#include "odecraft/Events/Events.hpp"
#include <odecraft/Compiled/Events.hpp>


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
    return ::ode::make_precise_event(std::move(name), std::move(obj_fun), std::move(event_tol), dir, std::move(mask), delay_mask);
}


template<typename T>
BoxedEvent<T> make_periodic_event(
    std::string name,
    T period,
    rhs_t<T> mask,
    bool delay_mask
){
    return ::ode::make_periodic_event(std::move(name), std::move(period), std::move(mask), delay_mask);
}

} // namespace ode::crafted