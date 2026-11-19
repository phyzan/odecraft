// Runtime events: zero crossings in every direction and at every tolerance, periodic events,
// several events within one step, masks that change the state where they fire, and the event
// machinery itself, through the solver and on its own.

#include "odecraft_test.hpp"

#include <numbers>
#include <string>
#include <vector>

namespace odetest {
namespace {

using ode::SolverPolicy;

/// @brief An event the solver stopped at: which one, when, and the state it reported there.
template<typename T, typename State>
struct Hit {
    std::string name;
    T t;
    State q;
    size_t step;
};

/// @brief Integrates until t_end, collecting every event on the way.
template<typename Solver, typename T>
auto record_events(Solver& solver, const T& t_end){
    using State = typename Solver::State;
    std::vector<Hit<T, State>> hits;
    REQUIRE(solver.advance_until(t_end, [&](const T& t, const State& q, const T* /*t_ptr*/){
        if (solver.at_event()){
            hits.push_back({solver.current_event().event->name(), t, q, solver.step_count()});
        }
        return true;
    }));
    CHECK(solver.t() == t_end);
    return hits;
}

template<typename T, typename State>
std::vector<std::string> names_of(const std::vector<Hit<T, State>>& hits){
    std::vector<std::string> names;
    for (const auto& hit : hits){
        names.push_back(hit.name);
    }
    return names;
}

/// @brief The zero crossings of x = cos t in (0, 5), in increasing t, with the sign of dx/dt there.
template<typename T>
std::vector<std::pair<T, int>> oscillator_crossings(){
    const T pi = T(std::numbers::pi_v<double>) * 0 + T(acos(T(-1)));
    return {{pi/2, -1}, {3*pi/2, +1}};
}

template<typename SD, typename T>
auto zero_of_x(const T& ftol, int crossing){
    return ode::make_precise_event<T, state_t<T, SD>>("x = 0", [](const T& /*t*/, const state_t<T, SD>& q) -> T { return q[0]; }, ftol, crossing);
}

template<typename T>
std::vector<double> event_tolerances(){
    if constexpr (std::is_same_v<T, float>){
        return {0, 1e-4};
    } else {
        return {0, 1e-9, 1e-4};
    }
}


TEMPLATE_TEST_CASE("Precise events locate every crossing in the requested direction", "[events][runtime]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            for_each_type<StateKinds<2>>([]<typename SD>(){
                DYNAMIC_SECTION(kind_name<SD>){
                    const double bound = settings<Stepper, T>().error;
                    for (int dir : {1, -1}){
                        for (int crossing : {1, -1, 0}){
                            for (double tol : event_tolerances<T>()){
                                CAPTURE(dir, crossing, tol);
                                const T ftol = T(tol);
                                const T t0 = dir == 1 ? T(0) : T(5);
                                const T t_end = dir == 1 ? T(5) : T(0);

                                // The crossings asked for, in the order the integration meets them.
                                // The direction is that of x as t increases, whichever way t runs.
                                std::vector<T> expected;
                                for (const auto& [t, sign] : oscillator_crossings<T>()){
                                    if (crossing == 0 || crossing == sign){
                                        expected.push_back(t);
                                    }
                                }
                                if (dir == -1){
                                    std::reverse(expected.begin(), expected.end());
                                }

                                auto solver = make_solver<Stepper, SD>(Oscillator{}, t0, Oscillator::at<SD>(t0), dir,
                                    ode::make_event_list<T, state_t<T, SD>>(zero_of_x<SD>(ftol, crossing)));
                                const auto hits = record_events(solver, t_end);

                                REQUIRE(hits.size() == expected.size());
                                for (size_t i = 0; i < hits.size(); i++){
                                    CAPTURE(i);
                                    // On time, within the tolerance asked for, and on the trajectory
                                    CHECK_THAT(abs_error(hits[i].t, expected[i]), IsAtMost(bound + tol));
                                    CHECK_THAT(abs_error(hits[i].q[0], T(0)), IsAtMost(tol + 100 * epsilon<T>()));
                                    CHECK_THAT(max_error<SD>(hits[i].q, Oscillator::at<SD>(hits[i].t)), IsAtMost(bound));
                                }
                            }
                        }
                    }
                }
            });
        }
    });
}

TEMPLATE_TEST_CASE("A crossing is reported once the root has been passed", "[events][runtime]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    // The event lies on the far side of the root, in the direction of integration, so x has
    // already changed sign there: an event never stops a solver short of its crossing
    for (int dir : {1, -1}){
        CAPTURE(dir);
        const T t0 = dir == 1 ? T(0) : T(5);
        auto solver = make_solver<tags::RK45, SD>(Oscillator{}, t0, Oscillator::at<SD>(t0), dir,
            ode::make_event_list<T, state_t<T, SD>>(zero_of_x<SD>(T(1e-4), 0)));
        const auto hits = record_events(solver, dir == 1 ? T(5) : T(0));
        REQUIRE(hits.size() == 2);
        for (const auto& hit : hits){
            // Moving forward x falls through pi/2 and rises through 3pi/2; backward, the opposite
            const int before = ode::sgn(Oscillator::at<SD>(T(hit.t - T(0.1 * dir)))[0]);
            CHECK(ode::sgn(hit.q[0]) != before);
        }
    }
}

TEMPLATE_TEST_CASE("Periodic events fire every period from the start time", "[events][runtime][periodic]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            for (int dir : {1, -1}){
                CAPTURE(dir);
                using SD = StaticSize<2>;
                const T t0 = T(0.25);
                const T period = T(0.4);
                // A periodic event fires at most once per step, so the steps are kept shorter than
                // its period: at float tolerances DOP853 would otherwise step over some of the ticks
                auto solver = make_solver<Stepper, SD>(Oscillator{}, t0, Oscillator::at<SD>(t0), dir,
                    ode::make_event_list<T, state_t<T, SD>>(ode::make_periodic_event<T, state_t<T, SD>>("tick", period)), T(period / T(2)));
                const auto hits = record_events(solver, T(t0 + T(3 * dir)));

                REQUIRE(hits.size() == 7);
                for (size_t n = 1; n <= hits.size(); n++){
                    CAPTURE(n);
                    CHECK(hits[n - 1].t == t0 + T(n) * period * T(dir));
                    CHECK_THAT(max_error<SD>(hits[n - 1].q, Oscillator::at<SD>(hits[n - 1].t)), IsAtMost(settings<Stepper, T>().error));
                }
            }
        }
    });
}


/// @brief A solver of uniform motion that takes unit steps, which every stepper integrates exactly.
template<typename Stepper, typename SD, typename T>
auto unit_step_solver(const state_t<T, SD>& q0, int dir, ode::EventList<T, state_t<T, SD>> events){
    const auto loose = uniform_like<SD, T>(q0, T(1));
    return ode::getSolver<SD, SolverPolicy::RichStatic>(Stepper{}, UniformMotion{}, T(0), q0, loose, loose, T(0), T(1), T(1), dir, std::move(events));
}

template<typename SD, typename T>
auto level_event(const std::string& name, int component, double level, int crossing = 0){
    return ode::make_precise_event<T, state_t<T, SD>>(name, [component, level](const T& /*t*/, const state_t<T, SD>& q) -> T {
        return q[component] - T(level);
    }, T(0), crossing);
}

TEMPLATE_TEST_CASE("Several events within one step are reported in time order", "[events][runtime][same-step]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([&]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            const double tol = uniform_motion_tolerance<Stepper, T>();
            for_each_type<StateKinds<2>>([&]<typename SD>(){
                DYNAMIC_SECTION(kind_name<SD>){
                    for (int dir : {1, -1}){
                        CAPTURE(dir);
                        // Unit steps, with x = t and y = -t: x reaches 0.2 and 1.1, y reaches -0.7, and a
                        // tick every 0.9, each times dir: several events per step, at most one of each
                        auto solver = unit_step_solver<Stepper, SD, T>(make_state<SD, T>({0, 0}), dir, ode::make_event_list<T, state_t<T, SD>>(
                            level_event<SD, T>("x", 0, 0.2 * dir),
                            level_event<SD, T>("y", 1, -0.7 * dir),
                            level_event<SD, T>("x again", 0, 1.1 * dir),
                            ode::make_periodic_event<T, state_t<T, SD>>("tick", T(0.9))));
                        const auto hits = record_events(solver, T(2 * dir));

                        const std::vector<std::string> names = {"x", "y", "tick", "x again", "tick"};
                        const std::vector<double> times = {0.2, 0.7, 0.9, 1.1, 1.8};
                        const std::vector<size_t> steps = {1, 1, 1, 2, 2};
                        REQUIRE(names_of(hits) == names);
                        for (size_t i = 0; i < hits.size(); i++){
                            CAPTURE(i);
                            CHECK_THAT(abs_error(hits[i].t, T(times[i] * dir)), IsAtMost(tol));
                            CHECK(hits[i].step == steps[i]);
                        }
                    }
                }
            });
        }
    });
}

TEMPLATE_TEST_CASE("Events at the same time are reported once by the first of them", "[events][runtime][same-step]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            using SD = StaticSize<2>;
            auto solver = unit_step_solver<Stepper, SD, T>(make_state<SD, T>({0, 0}), 1, ode::make_event_list<T, state_t<T, SD>>(
                level_event<SD, T>("first", 0, 0.5),
                level_event<SD, T>("twin", 0, 0.5),
                level_event<SD, T>("later", 0, 0.75)));
            const auto hits = record_events(solver, T(2));
            CHECK(names_of(hits) == std::vector<std::string>{"first", "later"});
        }
    });
}


/// @brief Every time x reaches 0.5*dir, the mask moves it back by one unit: x = t - k*dir in between.
template<typename SD, typename T>
auto teleport(int dir, bool delayed){
    return ode::make_precise_event<T, state_t<T, SD>>("teleport",
        [dir](const T& /*t*/, const state_t<T, SD>& q) -> T { return q[0] - T(0.5 * dir); }, T(0), 0,
        [dir](state_t<T, SD>& out, const T& /*t*/, const state_t<T, SD>& q){
            out = q;
            out[0] = q[0] - T(dir);
        }, delayed);
}

TEMPLATE_TEST_CASE("A masked event changes the state from where it fires", "[events][runtime][mask]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([&]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            const double tol = uniform_motion_tolerance<Stepper, T>();
            for_each_type<StateKinds<2>>([&]<typename SD>(){
                DYNAMIC_SECTION(kind_name<SD>){
                    for (int dir : {1, -1}){
                        for (bool delayed : {false, true}){
                            CAPTURE(dir, delayed);
                            auto solver = unit_step_solver<Stepper, SD, T>(make_state<SD, T>({0, 0}), dir,
                                ode::make_event_list<T, state_t<T, SD>>(teleport<SD, T>(dir, delayed)));

                            REQUIRE(solver.advance_to_event());
                            CHECK_THAT(abs_error(solver.t(), T(0.5 * dir)), IsAtMost(tol));
                            CHECK(solver.at_canon_event());
                            CHECK(solver.current_event().is_masked);
                            // An immediate mask shows the moved state at once, a delayed one the state
                            // before it, until the solver moves on
                            const auto at_event = current_state(solver);
                            CHECK_THAT(abs_error(at_event[0], T((delayed ? 0.5 : -0.5) * dir)), IsAtMost(tol));
                            CHECK_THAT(abs_error(at_event[1], T(-0.5 * dir)), IsAtMost(tol));

                            // Either way the trajectory goes on from the moved state
                            REQUIRE(solver.advance_until(T(1.25 * dir)));
                            CHECK_FALSE(solver.at_canon_event());
                            const auto after = current_state(solver);
                            CHECK_THAT(abs_error(after[0], T(0.25 * dir)), IsAtMost(tol));
                            CHECK_THAT(abs_error(after[1], T(-1.25 * dir)), IsAtMost(tol));
                        }
                    }
                }
            });
        }
    });
}

TEMPLATE_TEST_CASE("A masked event discards the later events of its step and they are found again", "[events][runtime][mask][same-step]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([&]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            const double tol = uniform_motion_tolerance<Stepper, T>();
            using SD = StaticSize<2>;
            for (bool delayed : {false, true}){
                CAPTURE(delayed);
                // x = t, moved back by one at every x = 0.5. Before the move, x would reach 0.75 within
                // the same step, which no longer happens, and 0.25 is passed again after every move.
                auto solver = unit_step_solver<Stepper, SD, T>(make_state<SD, T>({0, 0}), 1, ode::make_event_list<T, state_t<T, SD>>(
                    level_event<SD, T>("x = 0.25", 0, 0.25),
                    teleport<SD, T>(1, delayed),
                    level_event<SD, T>("x = 0.75", 0, 0.75)));
                const auto hits = record_events(solver, T(2.6));

                const std::vector<std::string> names = {"x = 0.25", "teleport", "x = 0.25", "teleport", "x = 0.25", "teleport"};
                const std::vector<double> times = {0.25, 0.5, 1.25, 1.5, 2.25, 2.5};
                REQUIRE(names_of(hits) == names);
                for (size_t i = 0; i < hits.size(); i++){
                    CAPTURE(i);
                    CHECK_THAT(abs_error(hits[i].t, T(times[i])), IsAtMost(tol));
                }
            }
        }
    });
}


TEMPLATE_TEST_CASE("A crossing just past the end of a step is found in the next step", "[events][runtime]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            using SD = StaticSize<2>;
            using State = state_t<T, SD>;
            // Unit steps along x = t: the first step ends at |x| = 1, within the event tolerance of
            // the root at |x| = 1.0001 but short of it. That step must not report it, since the
            // crossing has not happened yet, and the next one must.
            const T ftol = T(1e-3);
            for (int dir : {1, -1}){
                const T root = T(1.0001 * dir);
                for (int sense : {1, -1}){ // the objective rises (+1) or falls (-1) through the root, as t grows
                    for (int crossing : {sense, 0}){
                        CAPTURE(dir, sense, crossing);
                        auto solver = unit_step_solver<Stepper, SD, T>(make_state<SD, T>({0, 0}), dir, ode::make_event_list<T, State>(
                            ode::make_precise_event<T, State>("root", [sense, root](const T& /*t*/, const State& q) -> T {
                                return T(sense * (q[0] - root));
                            }, ftol, crossing)));
                        const auto hits = record_events(solver, T(3 * dir));
                        REQUIRE(hits.size() == 1);
                        CHECK(hits[0].step == 2);
                        CHECK(T(dir * hits[0].t) > T(1));
                        CHECK_THAT(abs_error(hits[0].t, root), IsAtMost(to_double(ftol)));
                    }
                }
            }
        }
    });
}

TEMPLATE_TEST_CASE("An event counts how many times it fired until the solver is reset", "[events][runtime]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            using SD = StaticSize<2>;
            auto solver = unit_step_solver<Stepper, SD, T>(make_state<SD, T>({0, 0}), 1, ode::make_event_list<T, state_t<T, SD>>(
                level_event<SD, T>("x = 0.25", 0, 0.25), teleport<SD, T>(1, false), ode::make_periodic_event<T, state_t<T, SD>>("tick", T(1.2))));
            const auto hits = record_events(solver, T(1.6));
            REQUIRE(names_of(hits) == std::vector<std::string>{"x = 0.25", "teleport", "tick", "x = 0.25", "teleport"});

            const auto& events = solver.event_col();
            CHECK(events.event(0).counter() == 2);
            CHECK(events.event(1).counter() == 2);
            CHECK(events.event(2).counter() == 1);
            solver.Reset();
            CHECK(events.event(0).counter() == 0);
            CHECK(events.event(2).counter() == 0);
        }
    });
}


TEMPLATE_TEST_CASE("advance_to_event stops at the next event of a selection", "[events][runtime][advance]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    // On uniform motion: "a" at t = 0.3, "b" at t = 0.6, "a" again never
    const auto events = [](){
        return ode::make_event_list<T, state_t<T, SD>>(level_event<SD, T>("a", 0, 0.3), level_event<SD, T>("b", 0, 0.6), ode::make_periodic_event<T, state_t<T, SD>>("tick", T(1)));
    };
    auto solver = unit_step_solver<tags::RK45, SD, T>(make_state<SD, T>({0, 0}), 1, events());

    SECTION("any event"){
        REQUIRE(solver.advance_to_event());
        CHECK(solver.at_event(0));
        CHECK_FALSE(solver.at_event(1));
        REQUIRE(solver.advance_to_event());
        CHECK(solver.at_event(1));
    }
    SECTION("by index"){
        REQUIRE(solver.advance_to_event(std::vector<size_t>{2}));
        CHECK(solver.current_event().event->name() == "tick");
        CHECK(solver.t() == T(1));
    }
    SECTION("by name"){
        REQUIRE(solver.advance_to_event(std::vector<std::string>{"b"}));
        CHECK(solver.at_event(1));
    }
    SECTION("up to a time, before which there may be none"){
        CHECK_FALSE(solver.advance_to_event(T(0.2)));
        CHECK(solver.t() == T(0.2));
        CHECK(solver.advance_to_event(T(2), std::vector<std::string>{"b"}));
        CHECK(solver.at_event(1));
        CHECK_FALSE(solver.advance_to_event(T(0.9), std::vector<size_t>{0}));
        CHECK(solver.t() == T(0.9));
    }
    SECTION("an event that does not exist is an error"){
        CHECK_THROWS_AS(solver.advance_to_event(std::vector<size_t>{3}), std::out_of_range);
        CHECK_THROWS_AS(solver.advance_to_event(std::vector<std::string>{"c"}), std::out_of_range);
        CHECK_THROWS_AS(solver.advance_to_event(T(2), std::vector<size_t>{3}), std::out_of_range);
    }
    SECTION("a solver without events has none to advance to"){
        auto no_events = unit_step_solver<tags::RK45, SD, T>(make_state<SD, T>({0, 0}), 1, {});
        CHECK_FALSE(no_events.advance_to_event());
        CHECK_FALSE(no_events.advance_to_event(T(1)));
    }
}

TEMPLATE_TEST_CASE("A solver reports the event it is at", "[events][runtime]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    auto solver = unit_step_solver<tags::RK45, SD, T>(make_state<SD, T>({0, 0}), 1, ode::make_event_list<T, state_t<T, SD>>(
        level_event<SD, T>("a", 0, 0.3), teleport<SD, T>(1, false)));

    CHECK(solver.event_col().size() == 2);
    CHECK(solver.event_idx("teleport") == 1);
    CHECK(solver.event_idx("missing") == -1);
    CHECK(solver.get_event_idx("a") == 0);
    CHECK_FALSE(solver.at_event());
    CHECK_FALSE(solver.current_event());

    REQUIRE(solver.advance_to_event());
    const auto state = solver.current_event();
    REQUIRE(state);
    CHECK(state.active);
    CHECK(state.idx == 0);
    CHECK(state.event->name() == "a");
    CHECK_FALSE(state.is_masked);
    CHECK_FALSE(solver.at_canon_event());
    CHECK(solver.get_at_event(0));
    CHECK(solver.get_current_event().idx == 0);

    REQUIRE(solver.advance_to_event());
    CHECK(solver.current_event().idx == 1);
    CHECK(solver.current_event().is_masked);
    CHECK(solver.get_at_canon_event());
    CHECK(solver.event_col().event(1).is_masked());
    CHECK_FALSE(solver.event_col().event(1).mask_delayed());
    CHECK(solver.event_col().event(1).direction() == 1);

    std::ostringstream text;
    solver.ShowState(text);
    CHECK_THAT(text.str(), Catch::Matchers::ContainsSubstring("teleport"));
}

TEMPLATE_TEST_CASE("Reset and clones carry the events along", "[events][runtime][lifecycle]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            using SD = StaticSize<2>;
            auto solver = unit_step_solver<Stepper, SD, T>(make_state<SD, T>({0, 0}), 1, ode::make_event_list<T, state_t<T, SD>>(
                level_event<SD, T>("x = 0.25", 0, 0.25), teleport<SD, T>(1, false)));
            const auto first_run = names_of(record_events(solver, T(1.6)));
            CHECK(first_run == std::vector<std::string>{"x = 0.25", "teleport", "x = 0.25", "teleport"});

            solver.Reset();
            CHECK_FALSE(solver.at_event());
            CHECK(names_of(record_events(solver, T(1.6))) == first_run);

            solver.Reset();
            REQUIRE(solver.advance_to_event());
            auto copy = solver.clone();
            CHECK(names_of(record_events(*copy, T(1.6))) == names_of(record_events(solver, T(1.6))));
        }
    });
}

TEMPLATE_TEST_CASE("An event list is checked when the solver is built", "[events][runtime]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    using State = state_t<T, SD>;
    const auto q0 = make_state<SD, T>({0, 0});

    CHECK_THROWS_AS((unit_step_solver<tags::RK45, SD, T>(q0, 1, ode::make_event_list<T, State>(
        level_event<SD, T>("same", 0, 0.3), level_event<SD, T>("same", 0, 0.6)))), std::runtime_error);
    CHECK_THROWS_AS((ode::make_periodic_event<T, State>("", T(1))), std::runtime_error);
    CHECK_THROWS_AS((level_event<SD, T>("", 0, 0.3)), std::runtime_error);
}


TEMPLATE_TEST_CASE("Events locate themselves without a solver", "[events][runtime][unit]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    using State = state_t<T, SD>;
    // x = t on [0, 1], through the linear interpolation of its ends
    const State q_old = make_state<SD, T>({0, 0}), q_new = make_state<SD, T>({1, 0});
    const ode::interp_t<T, State> interp = [](State& out, const T& t){ out[0] = t; out[1] = 0; };

    SECTION("a precise event"){
        const auto objective = [](const T& /*t*/, const State& q) -> T { return q[0] - T(0.25); };
        ode::PreciseEvent<T, State, decltype(objective)> event("quarter", objective, T(0), 1);
        CHECK(event.name() == "quarter");
        CHECK(event.sign_change_dir() == 1);
        CHECK(event.obj_fun(T(0), q_new) == T(0.75));
        CHECK_FALSE(event.is_masked());
        CHECK_FALSE(event.mask_delayed());
        State out = q_old;
        CHECK_THROWS_AS(event.apply_mask(out, T(0), q_old), std::runtime_error);

        CHECK_THROWS_AS(event.setup(T(0), q_old, 0), std::domain_error);
        event.setup(T(0), q_old, 1);
        CHECK(event.direction() == 1);
        CHECK(event.t_start() == T(0));

        T t_event = 0;
        REQUIRE(event.locate(t_event, T(0), T(1), q_old, q_new, interp));
        CHECK(event.is_located());
        CHECK_THAT(abs_error(t_event, T(0.25)), IsAtMost(4 * epsilon<T>()));
        CHECK(event.lock());
        CHECK(event.counter() == 1);

        // A decreasing crossing it does not look for
        CHECK_FALSE(event.locate(t_event, T(0), T(1), q_new, q_old, [](State& o, const T& t){ o[0] = 1 - t; o[1] = 0; }));
        CHECK_FALSE(event.lock());

        std::unique_ptr<ode::Event<T, State>> copy = event.clone();
        CHECK(copy->counter() == 1);
        event.reset();
        CHECK(event.counter() == 0);
        CHECK(copy->counter() == 1);
    }
    SECTION("a masked precise event"){
        const auto boxed = ode::make_precise_event<T, State>("flip", [](const T& /*t*/, const State& q) -> T { return q[0]; }, T(0), 0,
            [](State& out, const T& /*t*/, const State& q){ out[0] = -q[0]; out[1] = q[1] + 1; }, true);
        CHECK(boxed->is_masked());
        CHECK(boxed->mask_delayed());
        State out = q_old;
        boxed->apply_mask(out, T(0), q_new);
        CHECK(same_state<SD>(out, make_state<SD, T>({-1, 1})));
    }
    SECTION("a periodic event"){
        ode::PeriodicEvent<T, State> event("tick", T(0.3));
        CHECK(event.period() == T(0.3));
        event.setup(T(0), q_old, 1);
        T t_event = 0;
        REQUIRE(event.locate(t_event, T(0), T(1), q_old, q_new, interp));
        CHECK(t_event == T(0.3));
        REQUIRE(event.lock());
        REQUIRE(event.locate(t_event, T(0.3), T(1), q_old, q_new, interp));
        CHECK(t_event == T(2) * T(0.3));
        CHECK_FALSE(event.locate(t_event, T(0.3), T(0.5), q_old, q_new, interp));
    }
    SECTION("a collection of events, sorted by the time they are found at"){
        ode::EventCollection<T, State> col(ode::make_event_list<T, State>(
            level_event<SD, T>("late", 0, 0.75),
            level_event<SD, T>("early", 0, 0.25),
            ode::make_precise_event<T, State>("masked", [](const T& /*t*/, const State& q) -> T { return q[0] - T(0.5); }, T(0), 0,
                [](State& out, const T& /*t*/, const State& q){ out = q; out[1] = 7; })));
        CHECK(col.size() == 3);
        CHECK(col.event_idx("early") == 1);
        CHECK(col.event_idx("none") == -1);
        col.setup(T(0), q_old, 1);

        REQUIRE(col.detect_all_between(T(0), T(1), q_old, q_new, interp));
        // "late" falls after the masked event, which changes the state from 0.5 on: it is dropped
        REQUIRE(col.detection_size() == 2);
        CHECK(col.get_event(0)->name() == "early");
        CHECK(col.get_event_idx(1) == 2);
        CHECK(col.get_event(2) == nullptr);
        CHECK_THAT(abs_error(col.get_time(0), T(0.25)), IsAtMost(4 * epsilon<T>()));
        CHECK(col.is_located(1));
        CHECK_FALSE(col.is_located(0));

        const auto* masked = col.masked_state();
        REQUIRE(masked != nullptr);
        CHECK(masked->idx == 2);
        CHECK_THAT(abs_error(masked->time, T(0.5)), IsAtMost(4 * epsilon<T>()));
        CHECK(masked->masked_vector[1] == T(7));

        col.reset();
        CHECK(col.detection_size() == 0);
        CHECK(col.masked_state() == nullptr);

        const auto options = col.validate_events({ode::EventOptions{.name = "early", .max_events = 2}});
        REQUIRE(options.size() == 3);
        CHECK(options[1].max_events == 2);
        CHECK(options[0].name == "late");
        CHECK(options[0].max_events == -1);
        CHECK_THROWS_AS(col.validate_events({ode::EventOptions{.name = "none"}}), std::logic_error);
    }
    SECTION("an inactive event state"){
        const ode::EventState<T, State> none{};
        CHECK_FALSE(none);
        CHECK(none.event == nullptr);
    }
}

} // namespace
} // namespace odetest
