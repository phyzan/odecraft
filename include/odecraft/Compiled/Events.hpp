#ifndef ODECRAFT_COMPILED_EVENTS_HPP
#define ODECRAFT_COMPILED_EVENTS_HPP

/**
 * @file Compiled/Events.hpp
 * @brief Event detection of the compiled interface. Compiled by src/Events.cpp.
 *
 * Links after Interpolators.
 *
 * Both event kinds are compiled with `MaskFunc = rhs_t<T>` only. That single instantiation
 * covers the unmasked case too: EventBase::is_masked() tests a nullable MaskFunc against
 * nullptr at runtime, so passing a null mask behaves exactly like an unmasked event.
 */

#include <odecraft/Compiled/Interpolators.hpp>
#include <odecraft/Events/Events.hpp>
#include <odecraft/Events/Events_mem_impl.hpp>


namespace ode::crafted{

using ::ode::Event,
      ::ode::BoxedEvent,
      ::ode::EventList,
      ::ode::EventOptions,
      ::ode::EventPolicy,
      ::ode::EventCollection,
      ::ode::EventState,
      ::ode::MaskedState,
      ::ode::make_event_list;

/// @brief Zero-crossing event, as compiled into the library.
template<typename T>
using PreciseEvent = ::ode::PreciseEvent<T, objfun_t<T>, rhs_t<T>, EventPolicy::Virtual>;

/// @brief Fixed-interval event, as compiled into the library.
template<typename T>
using PeriodicEvent = ::ode::PeriodicEvent<T, rhs_t<T>, EventPolicy::Virtual>;


/**
 * @brief Build an event that triggers when @p obj_fun crosses zero.
 * @param name       Unique event name.
 * @param obj_fun    Monitored function; the event fires where it crosses zero.
 * @param event_tol  Bisection tolerance for locating the crossing.
 * @param dir        Crossing direction: +1 increasing, -1 decreasing, 0 either.
 * @param mask       Optional state transformation applied at the trigger. Null means unmasked.
 * @param delay_mask If true, the untransformed state is reported at the event and the mask
 *                   only takes effect afterwards.
 *
 * @note `T` must be given explicitly: `make_precise_event<double>(...)`.
 */
template<typename T>
BoxedEvent<T> make_precise_event(
    std::string name,
    objfun_t<T> obj_fun,
    T event_tol,
    int dir,
    rhs_t<T> mask = nullptr,
    bool delay_mask = false
);

/**
 * @brief Build an event that triggers every @p period time units from the start time.
 * @param name       Unique event name.
 * @param period     Interval between triggers.
 * @param mask       Optional state transformation applied at the trigger. Null means unmasked.
 * @param delay_mask As in make_precise_event.
 */
template<typename T>
BoxedEvent<T> make_periodic_event(
    std::string name,
    T period,
    rhs_t<T> mask = nullptr,
    bool delay_mask = false
);

} // namespace ode::crafted


namespace ode{


#define ODECRAFT_EXTERN_EVENTS(T)                                                       \
    extern template class PreciseEvent<T, objfun_t<T>, rhs_t<T>, EventPolicy::Virtual>; \
    extern template class PeriodicEvent<T, rhs_t<T>, EventPolicy::Virtual>;             \
    extern template class EventCollection<T>;

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_EVENTS)

#undef ODECRAFT_EXTERN_EVENTS

} // namespace ode


namespace ode::crafted{

#define ODECRAFT_EXTERN_EVENT_MAKERS(T)                                                             \
    extern template BoxedEvent<T> make_precise_event<T>(std::string, objfun_t<T>, T, int, rhs_t<T>, bool);  \
    extern template BoxedEvent<T> make_periodic_event<T>(std::string, T, rhs_t<T>, bool);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_EVENT_MAKERS)

#undef ODECRAFT_EXTERN_EVENT_MAKERS

} // namespace ode::crafted

#endif // ODECRAFT_COMPILED_EVENTS_HPP
