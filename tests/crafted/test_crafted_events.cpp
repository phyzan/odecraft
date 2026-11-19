// Events of the compiled interface, made by its own makers: zero crossings in either direction,
// periodic ones, masks applied at once or after the event, and the variational kinds.

#include <numbers>
#include <type_traits>

#include "../support/crafted_test.hpp"

namespace odetest {
namespace {

using namespace ode::crafted;

/// @brief A solver of the oscillator with RK45 and events, from t = 0.
template<typename T>
BoxedRichSolver<T> solver_with(EventList<T> events){
    const Settings<T> s = settings<tags::RK45, T>();
    const State<T> q0 = oscillator::at(T(0));
    return make_rich_vsolver<T>(Steppers::id<tags::RK45>, oscillator::system<T>(), T(0), q0, filled_like(q0, s.rtol), filled_like(q0, s.atol),
        T(0), T(0), T(0), 1, std::move(events));
}

/// @brief Every time the solver stops at an event before tmax.
template<typename T>
std::vector<T> event_times(OdeRichSolver<T>& solver, const T& tmax){
    std::vector<T> times;
    while (solver.do_advance_to_event(tmax)){
        times.push_back(solver.get_time());
    }
    return times;
}

/// @brief x = cos(t), on the oscillator.
template<typename T>
objfun_t<T> position(){
    return [](const T& /*t*/, const State<T>& q) -> T { return q[0]; };
}

/// @brief Doubles the state.
template<typename T>
rhs_t<T> doubling(){
    return [](State<T>& out, const T& /*t*/, const State<T>& q){
        out[0] = 2 * q[0];
        out[1] = 2 * q[1];
    };
}


TEMPLATE_TEST_CASE("A precise event fires where its function crosses zero in its direction", "[crafted][events]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    // cos(t) falls through zero at pi/2 and 5 pi/2, and rises through it at 3 pi/2
    constexpr double pi = std::numbers::pi;
    const std::vector<std::pair<int, std::vector<double>>> crossings = {
        {-1, {pi / 2, 5 * pi / 2}},
        {+1, {3 * pi / 2}},
        {0, {pi / 2, 3 * pi / 2, 5 * pi / 2}},
    };
    for (const double tol : {0.0, 1e-6}){
        for (const auto& [dir, expected] : crossings){
            CAPTURE(tol, dir);
            auto solver = solver_with<T>(make_event_list<T>(make_precise_event<T>("x = 0", position<T>(), T(tol), dir)));
            const std::vector<T> times = event_times(*solver, T(9));
            REQUIRE(times.size() == expected.size());
            for (size_t i = 0; i < times.size(); i++){
                CHECK_THAT(abs_error(times[i], T(expected[i])), IsAtMost(settings<tags::RK45, T>().error + tol));
            }
        }
    }
}

TEMPLATE_TEST_CASE("A periodic event fires once every period", "[crafted][events]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    auto solver = solver_with<T>(make_event_list<T>(make_periodic_event<T>("tick", T(0.75))));
    const std::vector<T> times = event_times(*solver, T(4));
    REQUIRE(times.size() == 5);
    for (size_t i = 0; i < times.size(); i++){
        CHECK(times[i] == T(int(i) + 1) * T(0.75));
    }
}

TEMPLATE_TEST_CASE("A mask moves the state at its event at once or right after it", "[crafted][events][mask]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    const double bound = 2 * settings<tags::RK45, T>().error;
    for (const bool delayed : {false, true}){
        CAPTURE(delayed);
        SECTION("on a precise event"){
            // At x = 0, t = pi/2, the state doubles: from then on it is 2 (cos t, -sin t)
            auto solver = solver_with<T>(make_event_list<T>(make_precise_event<T>("x = 0", position<T>(), T(0), -1, doubling<T>(), delayed)));
            REQUIRE(solver->do_advance_to_event());
            const T t = solver->get_time();
            const State<T> at_event = current_vector(*solver);
            CHECK_THAT(abs_error(T(at_event[1]), T(delayed ? -1 : -2)), IsAtMost(bound));
            REQUIRE(solver->do_advance_until(T(3)));
            const State<T> exact = oscillator::at(T(3));
            CHECK_THAT(max_error(current_vector(*solver), State<T>{T(2 * exact[0]), T(2 * exact[1])}), IsAtMost(bound));
            CHECK_THAT(abs_error(t, T(std::numbers::pi / 2)), IsAtMost(bound));
        }
        SECTION("on a periodic event"){
            // At t = 1 and t = 2 the state doubles: at t = 2.5 it is 4 (cos t, -sin t)
            auto solver = solver_with<T>(make_event_list<T>(make_periodic_event<T>("tick", T(1), doubling<T>(), delayed)));
            REQUIRE(solver->do_advance_to_event());
            CHECK(solver->get_time() == T(1));
            const State<T> exact_at_event = oscillator::at(T(1));
            const T scale = delayed ? T(1) : T(2);
            CHECK_THAT(max_error(current_vector(*solver), State<T>{T(scale * exact_at_event[0]), T(scale * exact_at_event[1])}), IsAtMost(bound));
            REQUIRE(solver->do_advance_until(T(2.5)));
            const State<T> exact = oscillator::at(T(2.5));
            CHECK_THAT(max_error(current_vector(*solver), State<T>{T(4 * exact[0]), T(4 * exact[1])}), IsAtMost(2 * bound));
        }
    }
}

TEMPLATE_TEST_CASE("Events are made into the compiled event types and held by name", "[crafted][events]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    BoxedEvent<T> zero = make_precise_event<T>("x = 0", position<T>(), T(0), 0);
    BoxedEvent<T> tick = make_periodic_event<T>("tick", T(0.5));
    CHECK(dynamic_cast<const PreciseEvent<T>*>(zero.get_raw_pointer()) != nullptr);
    CHECK(dynamic_cast<const PeriodicEvent<T>*>(tick.get_raw_pointer()) != nullptr);
    CHECK(zero->name() == "x = 0");

    const EventCollection<T> collection(make_event_list<T>(std::move(zero), std::move(tick)));
    CHECK(collection.size() == 2);
    CHECK(collection.event_idx("tick") == 1);
    CHECK(collection.event_idx("missing") == -1);
    CHECK(collection.event(0).name() == "x = 0");

    auto solver = solver_with<T>(make_event_list<T>(make_precise_event<T>("x = 0", position<T>(), T(0), 0), make_periodic_event<T>("tick", T(0.5))));
    STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(solver->get_event_col())>, EventCollection<T>>);
    STATIC_REQUIRE(std::is_same_v<decltype(solver->get_current_event()), EventState<T>>);
    STATIC_REQUIRE(std::is_same_v<decltype(solver->get_event_col().masked_state()), const MaskedState<T>*>);
    CHECK(solver->get_event_idx("tick") == 1);
    REQUIRE(solver->do_advance_to_event(std::vector<std::string>{"x = 0"}));
    const EventState<T> current = solver->get_current_event();
    CHECK(current.active);
    CHECK(current.idx == 0);
    CHECK(current.event->name() == "x = 0");
}

TEMPLATE_TEST_CASE("Variational events act on the state and the deviation", "[crafted][events][chaos]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    const objfun_t<T, VarState<T>> objective = [](const T& /*t*/, const VarState<T>& q) -> T { return q.main[0]; };
    const rhs_t<T, VarState<T>> mask = [](VarState<T>& out, const T& /*t*/, const VarState<T>& q){ out = q; };
    BoxedEvent<T, VarState<T>> zero = make_variational_precise_event<T>("x = 0", objective, T(0), -1);
    BoxedEvent<T, VarState<T>> tick = make_variational_periodic_event<T>("tick", T(1));
    BoxedEvent<T, VarState<T>> masked = make_variational_periodic_event<T>("kick", T(1), mask, true);
    CHECK(dynamic_cast<const PreciseEvent<T, VarState<T>>*>(zero.get_raw_pointer()) != nullptr);
    CHECK(dynamic_cast<const PeriodicEvent<T, VarState<T>>*>(tick.get_raw_pointer()) != nullptr);
    CHECK(masked->is_masked());

    const EventList<T, VarState<T>> list = make_variational_event_list<T>(std::move(zero), std::move(tick));
    CHECK(list.size() == 2);
}

} // namespace
} // namespace odetest
