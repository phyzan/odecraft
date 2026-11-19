// The ODE driver of DenseOde/OdeInt.hpp: integrating with any stepper, in either direction,
// storing every step or chosen times, recording events under options, returning dense solutions,
// and being cloned, reset and cleared. Also the result types it fills.

#include "../support/odecraft_test.hpp" // IWYU pragma: keep

#include <string>
#include <vector>

namespace odetest {
namespace {

using ode::EventOptions;
using ode::StepperStatus;

template<typename SD, typename T>
using Driver = ode::ODE<T, SD>;

/// @brief A driver of the oscillator from t0 with the stepper's settings.
template<typename Stepper, typename SD, typename T>
Driver<SD, T> oscillator_driver(const T& t0, int dir = 1, ode::EventList<T, state_t<T, SD>> events = {}){
    const Settings<T> s = settings<Stepper, T>();
    const auto q0 = Oscillator::at<SD>(t0);
    return Driver<SD, T>(Stepper{}, Oscillator{}, t0, q0, uniform_like<SD, T>(q0, s.rtol), uniform_like<SD, T>(q0, s.atol),
        T(0), T(0), s.stepsize, dir, std::move(events));
}

/// @brief Every recorded point of a result lies on the oscillator's trajectory.
template<typename SD, typename T>
void check_on_trajectory(const ode::OdeResult<T, SD>& result, double bound){
    REQUIRE(result.t().size() == result.q().size());
    for (size_t i = 0; i < result.t().size(); i++){
        CAPTURE(i);
        CHECK_THAT(max_error<SD>(result.q()[i], Oscillator::at<SD>(result.t()[i])), IsAtMost(bound));
    }
}

template<typename SD, typename T>
auto zero_and_tick(){
    using State = state_t<T, SD>;
    return ode::make_event_list<T, State>(
        ode::make_precise_event<T, State>("zero", [](const T& /*t*/, const State& q) -> T { return q[0]; }, T(0)),
        ode::make_periodic_event<T, State>("tick", T(1)));
}


TEMPLATE_TEST_CASE("The ODE driver integrates with any stepper and records every step", "[ode-driver]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            for_each_type<StateKinds<2>>([]<typename SD>(){
                DYNAMIC_SECTION(kind_name<SD>){
                    for (int dir : {1, -1}){
                        CAPTURE(dir);
                        const T t0 = T(1), t_end = T(1 + 3 * dir);
                        auto driver = oscillator_driver<Stepper, SD>(t0, dir);
                        CHECK(driver.solver()->get_method() == Stepper::name);

                        ode::OdeResult<T, SD> result;
                        REQUIRE(driver.integrate_until(&result, t_end));
                        CHECK(result.success());
                        CHECK(result.message() == "t-goal");
                        CHECK(result.runtime() >= 0);
                        CHECK(result.t().front() == t0);
                        CHECK(result.t().back() == t_end);
                        for (size_t i = 1; i < result.t().size(); i++){
                            CHECK(ode::sgn(result.t()[i - 1], result.t()[i]) == dir);
                        }
                        check_on_trajectory<SD>(result, settings<Stepper, T>().error);

                        CHECK(driver.n_points() == result.t().size());
                        CHECK(driver.t() == result.t());
                        CHECK(driver.runtime() >= 0);
                        CHECK_FALSE(driver.is_dead());
                        CHECK_FALSE(driver.diverges());
                    }
                }
            });
        }
    });
}

TEMPLATE_TEST_CASE("integrate moves by an interval in the driver's direction", "[ode-driver]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    for (int dir : {1, -1}){
        CAPTURE(dir);
        auto driver = oscillator_driver<tags::RK45, SD>(T(2), dir);
        ode::OdeResult<T, SD> result;
        REQUIRE(driver.integrate(&result, T(1.5)));
        CHECK(result.t().back() == T(2 + 1.5 * dir));
        CHECK(driver.solver()->get_time() == T(2 + 1.5 * dir));
        check_on_trajectory<SD>(result, settings<tags::RK45, T>().error);

        ode::OdeSolution<T, SD> solution;
        REQUIRE(driver.rich_integrate(solution, T(0.5)));
        CHECK(solution.t().back() == T(2 + 2 * dir));
        CHECK_THROWS_AS(driver.integrate(&result, T(-1)), std::runtime_error);
        CHECK_THROWS_AS(driver.rich_integrate(solution, T(-1)), std::runtime_error);
    }
}

TEMPLATE_TEST_CASE("Integrations continue where the last one ended", "[ode-driver]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    auto driver = oscillator_driver<tags::RK45, SD>(T(0), 1, zero_and_tick<SD, T>());
    ode::OdeResult<T, SD> first, second;
    REQUIRE(driver.integrate_until(&first, T(3)));
    REQUIRE(driver.integrate_until(&second, T(6)));

    CHECK(first.t().back() == T(3));
    CHECK(second.t().front() == T(3));
    CHECK(second.t().back() == T(6));
    // The driver keeps the whole history, the point at t = 3 once
    CHECK(driver.n_points() == first.t().size() + second.t().size() - 1);
    // Each result has the events of its own span: the zero at pi/2, then the one at 3pi/2
    CHECK(first.event_data().data("zero").size() == 1);
    CHECK(second.event_data().data("zero").size() == 1);
    CHECK(second.event_data().data("tick").t == std::vector<T>{T(4), T(5), T(6)});
    CHECK(driver.event_data("tick").size() == 6);
}

TEMPLATE_TEST_CASE("t_eval stores the solution at the requested times only", "[ode-driver][t-eval]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<StateKinds<2>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            const double bound = settings<tags::RK45, T>().error;
            ode::OdeResult<T, SD> result;

            SECTION("starting from the initial time if it is one of them"){
                auto driver = oscillator_driver<tags::RK45, SD>(T(0));
                const std::vector<T> t_eval = {T(0), T(0.5), T(2.5), T(7)};
                REQUIRE(driver.integrate_until(&result, T(8), {}, nullptr, t_eval));
                CHECK(result.t() == t_eval);
                check_on_trajectory<SD>(result, bound);
            }
            SECTION("and otherwise from the first of them"){
                auto driver = oscillator_driver<tags::RK45, SD>(T(0));
                const std::vector<T> t_eval = {T(1), T(1.5)};
                REQUIRE(driver.integrate(&result, T(2), {}, nullptr, t_eval));
                CHECK(result.t() == t_eval);
                check_on_trajectory<SD>(result, bound);
                CHECK(driver.solver()->get_time() == T(2));
            }
            SECTION("backward"){
                auto driver = oscillator_driver<tags::RK45, SD>(T(0), -1);
                const std::vector<T> t_eval = {T(-0.5), T(-2)};
                REQUIRE(driver.integrate_until(&result, T(-3), {}, nullptr, t_eval));
                CHECK(result.t() == t_eval);
                check_on_trajectory<SD>(result, bound);
            }
        }
    });
}

TEMPLATE_TEST_CASE("The driver records events under their names", "[ode-driver][events]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<StateKinds<2>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            const double bound = settings<tags::RK45, T>().error;
            auto driver = oscillator_driver<tags::RK45, SD>(T(0), 1, zero_and_tick<SD, T>());
            ode::OdeResult<T, SD> result;
            REQUIRE(driver.integrate_until(&result, T(5)));

            const auto& events = result.event_data();
            REQUIRE(events.size() == 2);
            CHECK(events.name(0) == "zero");
            CHECK(events.name(1) == "tick");
            const auto& zeros = events.data("zero");
            REQUIRE(zeros.size() == 2);
            const T pi = T(acos(T(-1)));
            CHECK_THAT(abs_error(zeros.t[0], T(pi/2)), IsAtMost(bound));
            CHECK_THAT(abs_error(zeros.t[1], T(3*pi/2)), IsAtMost(bound));
            CHECK_THAT(abs_error(zeros.q[1][0], T(0)), IsAtMost(100 * epsilon<T>()));
            CHECK(events.data(1).t == std::vector<T>{T(1), T(2), T(3), T(4), T(5)});
            CHECK(driver.event_data("tick").q.size() == 5);
            CHECK_THROWS_AS(events.data("none"), std::runtime_error);
            CHECK_THROWS_AS(driver.event_data("none"), std::runtime_error);
            CHECK_THAT(result.event_log(), Catch::Matchers::ContainsSubstring("zero") && Catch::Matchers::ContainsSubstring("tick"));
        }
    });
}

TEMPLATE_TEST_CASE("The driver records every event of a step and the state each one left", "[ode-driver][events][same-step]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    using State = state_t<T, SD>;
    const double tol = 20 * epsilon<T>();
    for (bool delayed : {false, true}){
        CAPTURE(delayed);
        // Unit steps of uniform motion, x = t: x = 0.2 and 0.4 in the first step, and x = 0.6
        // moving x back by one, after which the solver goes on from x = -0.4
        const auto q0 = make_state<SD, T>({0, 0});
        const auto loose = uniform_like<SD, T>(q0, T(1));
        Driver<SD, T> driver(tags::RK45{}, UniformMotion{}, T(0), q0, loose, loose, T(0), T(1), T(1), 1, ode::make_event_list<T, State>(
            ode::make_precise_event<T, State>("x = 0.2", [](const T&, const State& q) -> T { return q[0] - T(0.2); }, T(0)),
            ode::make_precise_event<T, State>("x = 0.4", [](const T&, const State& q) -> T { return q[0] - T(0.4); }, T(0)),
            ode::make_precise_event<T, State>("back", [](const T&, const State& q) -> T { return q[0] - T(0.6); }, T(0), 0,
                [](State& out, const T&, const State& q){ out = q; out[0] = q[0] - 1; }, delayed)));
        ode::OdeResult<T, SD> result;
        REQUIRE(driver.integrate_until(&result, T(1.5)));

        const auto& events = result.event_data();
        CHECK(events.data("x = 0.2").t.size() == 2);   // at 0.2, and at 1.2 after the move back
        CHECK(events.data("x = 0.4").t.size() == 2);   // at 0.4, and at 1.4
        const auto& back = events.data("back");
        REQUIRE(back.size() == 1);
        CHECK_THAT(abs_error(back.t[0], T(0.6)), IsAtMost(tol));
        // An immediate mask records the moved state, a delayed one the state before it
        CHECK_THAT(abs_error(back.q[0][0], T(delayed ? 0.6 : -0.4)), IsAtMost(tol));
        CHECK_THAT(abs_error(events.data("x = 0.2").t[1], T(T(0.2) + 1)), IsAtMost(tol));
    }
}

TEMPLATE_TEST_CASE("Event options cap thin out or end the recording of an event", "[ode-driver][events][options]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    auto driver = oscillator_driver<tags::RK45, SD>(T(0), 1, zero_and_tick<SD, T>());
    ode::OdeResult<T, SD> result;

    SECTION("max_events records that many and carries on"){
        REQUIRE(driver.integrate_until(&result, T(10), {EventOptions{.name = "tick", .max_events = 3}}));
        CHECK(result.event_data().data("tick").t == std::vector<T>{T(1), T(2), T(3)});
        CHECK(result.t().back() == T(10));
        CHECK(result.event_data().data("zero").size() == 3);
    }
    SECTION("terminate ends the integration at the last one"){
        CHECK_FALSE(driver.integrate_until(&result, T(10), {EventOptions{.name = "tick", .max_events = 3, .terminate = true}}));
        CHECK(result.message() == "Max events reached");
        CHECK(result.t().back() == T(3));
        CHECK(driver.solver()->get_is_running());
        // and the next integration picks up from there
        CHECK(driver.integrate_until(&result, T(4)));
    }
    SECTION("period records every k-th"){
        REQUIRE(driver.integrate_until(&result, T(10), {EventOptions{.name = "tick", .period = 3}}));
        CHECK(result.event_data().data("tick").t == std::vector<T>{T(3), T(6), T(9)});
    }
    SECTION("options of events that do not exist, or that make no sense, are errors"){
        CHECK_THROWS_AS(driver.integrate_until(&result, T(1), {EventOptions{.name = "none"}}), std::logic_error);
        CHECK_THROWS_AS(driver.integrate_until(&result, T(1), {EventOptions{.name = "tick", .period = 0}}), std::runtime_error);
    }
}

TEMPLATE_TEST_CASE("The driver's observer sees every recorded state and can stop it", "[ode-driver][observer]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    using State = state_t<T, SD>;
    auto driver = oscillator_driver<tags::RK45, SD>(T(0));
    ode::OdeResult<T, SD> result;

    size_t calls = 0;
    const ode::observer_t<T, State> stop_after_five = [&calls](const T& t, const State& q, const T*){
        CHECK_THAT(max_error<SD>(q, Oscillator::at<SD>(t)), IsAtMost(settings<tags::RK45, T>().error));
        return ++calls < 5;
    };
    CHECK_FALSE(driver.integrate_until(&result, T(10), {}, stop_after_five));
    CHECK(calls == 5);
    CHECK(result.t().size() == 6);   // the initial state and the five observed
    CHECK(result.t().back() < T(10));
}

TEMPLATE_TEST_CASE("A rich integration returns a solution to evaluate anywhere in its span", "[ode-driver][dense-output]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            using SD = StaticSize<2>;
            const double bound = settings<Stepper, T>().error;
            auto driver = oscillator_driver<Stepper, SD>(T(0), 1, zero_and_tick<SD, T>());
            ode::OdeSolution<T, SD> solution;
            REQUIRE(driver.rich_integrate_until(solution, T(4)));
            CHECK(solution.success());

            auto q = Oscillator::at<SD>(T(0));
            for (double t : {0.0, 0.3, 1.0, 1.5707, 2.25, 3.999, 4.0}){
                CAPTURE(t);
                solution(q, T(t));
                CHECK_THAT(max_error<SD>(q, Oscillator::at<SD>(T(t))), IsAtMost(bound));
            }
            CHECK_THROWS(solution(q, T(4.5)));
            CHECK(solution.event_data().data("zero").size() == 1);   // pi/2

            const std::unique_ptr<ode::OdeResult<T, SD>> copy = solution.clone();
            const auto* as_solution = dynamic_cast<const ode::OdeSolution<T, SD>*>(copy.get());
            REQUIRE(as_solution != nullptr);
            (*as_solution)(q, T(2.25));
            CHECK_THAT(max_error<SD>(q, Oscillator::at<SD>(T(2.25))), IsAtMost(bound));
        }
    });
}

TEMPLATE_TEST_CASE("A driver whose solver died integrates no more", "[ode-driver][failure]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const auto q0 = Oscillator::at<SD>(T(0));
    const auto tol = uniform_like<SD, T>(q0, T(1e-6));
    Driver<SD, T> driver(tags::RK45{}, PoisonedOscillator{.t_fail = 0.5}, T(0), q0, tol, tol);

    ode::OdeResult<T, SD> result;
    CHECK_FALSE(driver.integrate_until(&result, T(2)));
    CHECK(result.status() == StepperStatus::Diverges);
    CHECK(result.message() == "Integrator diverges");
    CHECK_FALSE(result.success());
    CHECK(driver.diverges());
    CHECK(driver.is_dead());
    CHECK(result.t().back() <= T(0.5));

    ode::OdeResult<T, SD> again;
    CHECK_FALSE(driver.integrate_until(&again, T(3)));
    CHECK(again.t().empty());
    CHECK(again.status() == StepperStatus::Diverges);

    ode::OdeSolution<T, SD> solution;
    CHECK_FALSE(driver.rich_integrate_until(solution, T(3)));
    auto q = q0;
    CHECK_THROWS_AS(solution(q, T(0.1)), std::runtime_error);
}

TEMPLATE_TEST_CASE("A driver cannot integrate against its direction", "[ode-driver]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    auto driver = oscillator_driver<tags::RK45, SD>(T(1));
    ode::OdeResult<T, SD> result;
    CHECK_FALSE(driver.integrate_until(&result, T(0)));
    CHECK(result.message() == "Cannot integrate in opposite direction");
    CHECK(driver.solver()->get_time() == T(1));
}

TEMPLATE_TEST_CASE("reset and clear start the record over", "[ode-driver][lifecycle]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    auto driver = oscillator_driver<tags::RK45, SD>(T(0), 1, zero_and_tick<SD, T>());
    REQUIRE(driver.integrate_until(nullptr, T(3)));
    REQUIRE(driver.n_points() > 2);

    SECTION("reset goes back to the initial conditions"){
        driver.reset();
        CHECK(driver.solver()->get_time() == T(0));
        CHECK(driver.n_points() == 1);
        CHECK(driver.t().front() == T(0));
        CHECK(driver.runtime() == 0);
        CHECK(driver.event_data("tick").size() == 0);
        ode::OdeResult<T, SD> result;
        REQUIRE(driver.integrate_until(&result, T(3)));
        CHECK(result.event_data().data("tick").size() == 3);
    }
    SECTION("clear keeps the solver where it is"){
        driver.clear();
        CHECK(driver.solver()->get_time() == T(3));
        CHECK(driver.n_points() == 1);
        CHECK(driver.t().front() == T(3));
        CHECK(driver.event_data("zero").size() == 0);
    }
}

TEMPLATE_TEST_CASE("A cloned driver integrates independently and identically", "[ode-driver][clone]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    auto driver = oscillator_driver<tags::RK45, SD>(T(0), 1, zero_and_tick<SD, T>());
    REQUIRE(driver.integrate_until(nullptr, T(2)));
    std::unique_ptr<Driver<SD, T>> copy = driver.clone();

    REQUIRE(copy->integrate_until(nullptr, T(5)));
    CHECK(driver.solver()->get_time() == T(2));   // the original stayed

    REQUIRE(driver.integrate_until(nullptr, T(5)));
    CHECK(driver.t() == copy->t());
    CHECK(same_state<SD>(driver.q().back(), copy->q().back()));
    CHECK(driver.event_data("zero").t == copy->event_data("zero").t);
}

TEMPLATE_TEST_CASE("The driver evaluates its system", "[ode-driver]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const auto driver = oscillator_driver<tags::RK45, SD>(T(0));
    const auto q = make_state<SD, T>({2, 3});

    auto out = q;
    driver.Rhs(out, T(0), q);
    CHECK(same_state<SD>(out, make_state<SD, T>({3, -2})));

    auto J = make_jacmat<SD, T>(q), exact = J;
    Oscillator::jacobian<SD, T>(exact);
    driver.Jac(J, T(0), q);
    CHECK_THAT(max_matrix_error(J, exact, 2), IsAtMost(10 * epsilon<T>()));
    driver.Jac(J, T(0), q, uniform_like<SD, T>(q, T(1e-3)));
    CHECK_THAT(max_matrix_error(J, exact, 2), IsAtMost(1e3 * epsilon<T>()));
}


TEMPLATE_TEST_CASE("Orbit and event data hold points under names", "[ode-driver][result]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const auto q = make_state<SD, T>({1, 2});

    SECTION("orbit data"){
        ode::OrbitData<T, SD> orbit;
        orbit.add_point(T(0), q);
        orbit.add_point(T(1), q);
        CHECK(orbit.size() == 2);
        CHECK(orbit.t[1] == T(1));
        orbit.clear_all();
        CHECK(orbit.size() == 0);
        CHECK(orbit.q.empty());
    }
    SECTION("event data"){
        ode::EventData<T, SD> data;
        data.allocate_event("a");
        data.allocate_event("b");
        CHECK_THROWS_AS(data.allocate_event("a"), std::runtime_error);
        CHECK_THROWS_AS(data.allocate_event(""), std::runtime_error);

        data.add_event(0, T(1), q);
        data.add_event("b", T(2), q);
        data.add_event("b", T(3), q);
        CHECK_THROWS_AS(data.add_event(2, T(4), q), std::runtime_error);
        CHECK_THROWS_AS(data.add_event("c", T(4), q), std::runtime_error);
        CHECK(data.size() == 2);
        CHECK(data.name(1) == "b");
        CHECK(data.data("b").t == std::vector<T>{T(2), T(3)});
        CHECK(data.data(0).size() == 1);

        // A copy holding only what was added after the given counts
        const ode::EventData<T, SD> later(data, {1, 1});
        CHECK(later.data("a").size() == 0);
        CHECK(later.data("b").t == std::vector<T>{T(3)});

        data.clear_points();
        CHECK(data.data("b").size() == 0);
        CHECK(data.size() == 2);
    }
}

TEMPLATE_TEST_CASE("A result holds the points from where it started", "[ode-driver][result]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    ode::OrbitData<T, SD> orbit;
    for (int i = 0; i < 4; i++){
        orbit.add_point(T(i), make_state<SD, T>({T(i), 0}));
    }
    const ode::OdeResult<T, SD> result(orbit, ode::EventData<T, SD>{}, 2, StepperStatus::NewStep, 1.5, "done");
    CHECK(result.t() == std::vector<T>{T(2), T(3)});
    CHECK(result.q()[0][0] == T(2));
    CHECK(result.success());
    CHECK(result.runtime() == 1.5);
    CHECK(result.message() == "done");

    const ode::OdeResult<T, SD> empty;
    CHECK_FALSE(empty.success());
    CHECK(empty.message() == "No integration performed");

    CaptureCout capture;
    result.examine();
    CHECK_THAT(capture.str(), Catch::Matchers::ContainsSubstring("Points           : 2"));
}

TEMPLATE_TEST_CASE("An event counter counts within its options", "[ode-driver][events][options]", ODECRAFT_TEST_SCALARS){
    using SD = StaticSize<2>;
    ode::EventCounter<TestType, SD> counter({EventOptions{.name = "a", .max_events = 2, .terminate = true},
                                           EventOptions{.name = "b", .period = 2}});
    CHECK(counter.count_it(1) == false);   // the first of every two
    CHECK(counter.count_it(1) == true);
    CHECK(counter[1] == 1);
    CHECK(counter.count_it(0));
    CHECK(counter.is_running());
    CHECK(counter.count_it(0));
    CHECK_FALSE(counter.is_running());      // the second of at most two, which terminates
    CHECK_FALSE(counter.can_fit(0));
    CHECK(counter.total() == 3);
    CHECK_THROWS_AS((ode::EventCounter<TestType, SD>({EventOptions{.name = "c", .period = 0}})), std::runtime_error);
}

} // namespace
} // namespace odetest
