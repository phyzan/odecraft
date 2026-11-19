// The ODE driver of the compiled interface: built with a stepper chosen at runtime, integrating
// in either direction, recording events under options, storing chosen times, returning dense
// solutions, and held through an OdeDriver handle that copies it whole.

#include <memory>
#include <numbers>
#include <type_traits>

#include "../support/crafted_test.hpp"

namespace odetest {
namespace {

using namespace ode::crafted;

/// @brief A driver of the oscillator from t = 0 with the settings the tests give its stepper.
template<typename T>
ODE<T> oscillator_driver(const std::string& stepper = "RK45", int dir = 1, EventList<T> events = {}){
    const Steppers::Id id = Steppers::getID(stepper);
    const Settings<T> s = settings_of<T>(id);
    const State<T> q0 = oscillator::at(T(0));
    return ODE<T>(oscillator::system<T>(), T(0), q0, filled_like(q0, s.rtol), filled_like(q0, s.atol), T(0), T(0), s.stepsize, dir, std::move(events), id);
}

/// @brief The largest distance from the points of a result to the oscillator's trajectory.
template<typename T, typename Result>
Magnitude distance_to_trajectory(const Result& result){
    if (result.t().empty() || result.t().size() != result.q().size()){
        return {std::numeric_limits<double>::quiet_NaN()};
    }
    Magnitude err{0};
    for (size_t i = 0; i < result.t().size(); i++){
        err = worst(max_error(result.q()[i], oscillator::at(result.t()[i])), err);
    }
    return err;
}

template<typename T>
EventList<T> zero_and_tick(){
    return make_event_list<T>(
        make_precise_event<T>("zero", [](const T& /*t*/, const State<T>& q) -> T { return q[0]; }, T(0), 0),
        make_periodic_event<T>("tick", T(1)));
}


TEMPLATE_TEST_CASE("The ODE driver integrates with any compiled stepper in either direction", "[crafted][ode-driver]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    STATIC_REQUIRE(std::is_base_of_v<OdeDriver<T>, ODE<T>>);
    for (const std::string& name : stepper_names){
        DYNAMIC_SECTION(name){
            for (int dir : {1, -1}){
                CAPTURE(dir);
                ODE<T> driver = oscillator_driver<T>(name, dir);
                CHECK(driver.solver()->get_method() == name);
                OdeResult<T> result;
                REQUIRE(driver.integrate_until(&result, T(4 * dir)));
                CHECK(result.success());
                CHECK(result.t().back() == T(4 * dir));
                CHECK(driver.n_points() == result.t().size());
                CHECK_THAT(distance_to_trajectory<T>(result), IsAtMost(settings_of<T>(Steppers::getID(name)).error));
            }
        }
    }
}

TEMPLATE_TEST_CASE("The ODE driver takes RK45 unless told otherwise", "[crafted][ode-driver]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    const State<T> q0 = oscillator::at(T(0));
    const State<T> tol = filled_like(q0, T(1e-6));
    const ODE<T> driver(oscillator::system<T>(), T(0), q0, tol, tol);
    CHECK(driver.solver()->get_method() == "RK45");
    CHECK(driver.solver()->get_direction() == 1);
}

TEMPLATE_TEST_CASE("The ODE driver records events and stores chosen times", "[crafted][ode-driver][events]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    const double bound = settings<tags::RK45, T>().error;

    SECTION("events, under their names"){
        ODE<T> driver = oscillator_driver<T>("RK45", 1, zero_and_tick<T>());
        OdeResult<T> result;
        REQUIRE(driver.integrate_until(&result, T(5)));
        const EventData<T>& events = result.event_data();
        CHECK(events.size() == 2);
        const OrbitData<T>& zeros = events.data("zero");
        const OrbitData<T>& ticks = events.data("tick");
        REQUIRE(zeros.size() == 2); // pi/2 and 3 pi/2
        CHECK_THAT(abs_error(zeros.t[0], T(std::numbers::pi / 2)), IsAtMost(bound));
        CHECK_THAT(abs_error(zeros.t[1], T(3 * std::numbers::pi / 2)), IsAtMost(bound));
        REQUIRE(ticks.size() == 5);
        CHECK(ticks.t[4] == T(5));
        CHECK_THAT(max_error(ticks.q[2], oscillator::at(T(3))), IsAtMost(bound));
        CHECK(driver.event_data("tick").size() == 5);
    }
    SECTION("events, under options"){
        ODE<T> driver = oscillator_driver<T>("RK45", 1, zero_and_tick<T>());
        OdeResult<T> result;
        const std::vector<EventOptions> options = {{.name = "tick", .max_events = 2, .terminate = true}};
        CHECK_FALSE(driver.integrate_until(&result, T(10), options));
        CHECK(result.message() == "Max events reached");
        CHECK(result.event_data().data("tick").size() == 2);
        CHECK(driver.solver()->get_time() == T(2));
    }
    SECTION("chosen times only"){
        ODE<T> driver = oscillator_driver<T>();
        OdeResult<T> result;
        const std::vector<T> t_eval = {T(0.5), T(1.25), T(3)};
        REQUIRE(driver.integrate_until(&result, T(4), {}, nullptr, t_eval));
        CHECK(result.t() == t_eval);
        CHECK_THAT(distance_to_trajectory<T>(result), IsAtMost(bound));
    }
    SECTION("what an observer sees, until it says stop"){
        ODE<T> driver = oscillator_driver<T>();
        OdeResult<T> result;
        size_t seen = 0;
        const observer_t<T> observer = [&seen](const T&, const State<T>&, const T*){ return ++seen < 5; };
        CHECK_FALSE(driver.integrate_until(&result, T(4), {}, observer));
        CHECK(seen == 5);
        CHECK(driver.solver()->get_time() < T(4));
    }
}

TEMPLATE_TEST_CASE("The ODE driver returns a solution to evaluate anywhere in its span", "[crafted][ode-driver][dense-output]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    const double bound = settings<tags::RK45, T>().error;
    for (int dir : {1, -1}){
        CAPTURE(dir);
        ODE<T> driver = oscillator_driver<T>("RK45", dir);
        OdeSolution<T> solution;
        REQUIRE(driver.rich_integrate_until(solution, T(6 * dir)));
        State<T> q = oscillator::at(T(0));
        for (const double t : {0.4, 2.2, 5.9}){
            solution(q, T(t * dir));
            CHECK_THAT(max_error(q, oscillator::at(T(t * dir))), IsAtMost(bound));
        }
    }
}

TEMPLATE_TEST_CASE("The ODE driver integrates by an interval in its direction", "[crafted][ode-driver]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    for (int dir : {1, -1}){
        CAPTURE(dir);
        ODE<T> driver = oscillator_driver<T>("RK45", dir);
        OdeResult<T> result;
        REQUIRE(driver.integrate(&result, T(1.5)));
        CHECK(driver.solver()->get_time() == T(1.5 * dir));
        OdeSolution<T> solution;
        REQUIRE(driver.rich_integrate(solution, T(0.5)));
        CHECK(driver.solver()->get_time() == T(2 * dir));
    }
}

TEMPLATE_TEST_CASE("An OdeDriver handle copies a driver whole", "[crafted][ode-driver][clone]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    pbox::owner<OdeDriver<T>> original(std::make_unique<ODE<T>>(oscillator_driver<T>()));
    REQUIRE(original->integrate_until(nullptr, T(1)));

    pbox::owner<OdeDriver<T>> copy = original;
    REQUIRE(copy->integrate_until(nullptr, T(2)));
    CHECK(original->solver()->get_time() == T(1));
    CHECK(copy->solver()->get_time() == T(2));
    CHECK(copy->n_points() > original->n_points());

    SECTION("which starts over or forgets its record"){
        copy->reset();
        CHECK(copy->solver()->get_time() == T(0));
        CHECK(copy->n_points() == 1);
        REQUIRE(copy->integrate_until(nullptr, T(1)));
        copy->clear();
        CHECK(copy->solver()->get_time() == T(1));
        CHECK(copy->n_points() == 1);
    }
}

TEMPLATE_TEST_CASE("An event counter counts each event within its options", "[crafted][ode-driver][events]", ODECRAFT_CRAFTED_SCALARS){
    EventCounter<TestType, SD> counter({{.name = "a", .max_events = 2}, {.name = "b"}});
    CHECK(counter.is_running());
    CHECK(counter.count_it(0));
    CHECK(counter.count_it(1));
    CHECK(counter.count_it(0));
    CHECK(counter[0] == 2);
    CHECK(counter.total() == 3);
    CHECK_FALSE(counter.can_fit(0));
    CHECK(counter.can_fit(1));
}

} // namespace
} // namespace odetest
