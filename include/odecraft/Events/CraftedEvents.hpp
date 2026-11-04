#pragma once

/**
 * @file Events/CraftedEvents.hpp
 * @brief Event detection of the compiled interface. Compiled by src/Events.cpp.
 *
 * Links after Interpolators.
 *
 * Both event kinds are compiled with `MaskFunc = rhs_t<T>` only. That single instantiation
 * covers the unmasked case too: EventBase::is_masked() tests a nullable MaskFunc against
 * nullptr at runtime, so passing a null mask behaves exactly like an unmasked event.
 */

#include <odecraft/Interpolation/Univariate/CraftedStateInterp.hpp>
#include "Events.hpp"
#include "Events_mem_impl.hpp" // IWYU pragma: keep


namespace ode::crafted{

using ::ode::EventOptions,
      ::ode::EventPolicy;

// S is the state the event acts on: State<T>, or VarState<T> around a variational solver.

template<typename T, typename S = State<T>>
using Event = ::ode::Event<T, S>;

template<typename T, typename S = State<T>>
using BoxedEvent = ::ode::BoxedEvent<T, S>;

template<typename T, typename S = State<T>>
using EventList = ::ode::EventList<T, S>;

template<typename T, typename S = State<T>>
using EventCollection = ::ode::EventCollection<T, S>;

template<typename T, typename S = State<T>>
using EventState = ::ode::EventState<T, S>;

template<typename T, typename S = State<T>>
using MaskedState = ::ode::MaskedState<T, S>;

/// @brief Zero-crossing event, as compiled into the library.
template<typename T, typename S = State<T>>
using PreciseEvent = ::ode::PreciseEvent<T, S, objfun_t<T, S>, rhs_t<T, S>, EventPolicy::Virtual>;

/// @brief Fixed-interval event, as compiled into the library.
template<typename T, typename S = State<T>>
using PeriodicEvent = ::ode::PeriodicEvent<T, S, rhs_t<T, S>, EventPolicy::Virtual>;


/// @brief Collect events into an EventList, taking ownership of them.
template<typename T, typename... Events>
inline EventList<T> make_event_list(Events... events){
    return ::ode::make_event_list<T, State<T>>(std::move(events)...);
}

/// @brief As make_event_list, for the events of a variational solver.
template<typename T, typename... Events>
inline EventList<T, VarState<T>> make_variational_event_list(Events... events){
    return ::ode::make_event_list<T, VarState<T>>(std::move(events)...);
}


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

/// @brief As make_precise_event, for a variational solver: the callables act on a VarState<T>.
template<typename T>
BoxedEvent<T, VarState<T>> make_variational_precise_event(
    std::string name,
    objfun_t<T, VarState<T>> obj_fun,
    T event_tol,
    int dir,
    rhs_t<T, VarState<T>> mask = nullptr,
    bool delay_mask = false
);

/// @brief As make_periodic_event, for a variational solver: the mask acts on a VarState<T>.
template<typename T>
BoxedEvent<T, VarState<T>> make_variational_periodic_event(
    std::string name,
    T period,
    rhs_t<T, VarState<T>> mask = nullptr,
    bool delay_mask = false
);

} // namespace ode::crafted


namespace ode{


#define ODECRAFT_EXTERN_EVENTS_OF(T, S)                                                         \
    extern template class PreciseEvent<T, S, objfun_t<T, S>, rhs_t<T, S>, EventPolicy::Virtual>;\
    extern template class PeriodicEvent<T, S, rhs_t<T, S>, EventPolicy::Virtual>;               \
    extern template class EventCollection<T, S>;

#define ODECRAFT_EXTERN_EVENTS(T)                                       \
    ODECRAFT_EXTERN_EVENTS_OF(T, crafted::State<T>)                     \
    ODECRAFT_EXTERN_EVENTS_OF(T, crafted::VarState<T>)

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_EVENTS)

#undef ODECRAFT_EXTERN_EVENTS
#undef ODECRAFT_EXTERN_EVENTS_OF

} // namespace ode


namespace ode::crafted{

#define ODECRAFT_EXTERN_EVENT_MAKERS(T)                                                                     \
    extern template BoxedEvent<T> make_precise_event<T>(std::string, objfun_t<T>, T, int, rhs_t<T>, bool);  \
    extern template BoxedEvent<T> make_periodic_event<T>(std::string, T, rhs_t<T>, bool);                   \
    extern template BoxedEvent<T, VarState<T>> make_variational_precise_event<T>(                           \
        std::string, objfun_t<T, VarState<T>>, T, int, rhs_t<T, VarState<T>>, bool);                        \
    extern template BoxedEvent<T, VarState<T>> make_variational_periodic_event<T>(                          \
        std::string, T, rhs_t<T, VarState<T>>, bool);

ODECRAFT_FOR_EACH_SCALAR(ODECRAFT_EXTERN_EVENT_MAKERS)

#undef ODECRAFT_EXTERN_EVENT_MAKERS

} // namespace ode::crafted
