// Dense output: interpolate_until covers exactly what it integrated, accurately, through events,
// with the jump a masked event makes, and up to where a failing solver stops.

#include "../support/odecraft_test.hpp" // IWYU pragma: keep

#include <vector>

namespace odetest {
namespace {

using ode::SolverPolicy;
using ode::interp::uni::BoxedInterp;

/// @brief n + 1 times evenly spread over [a, b], ends included.
template<typename T>
std::vector<T> times_across(const T& a, const T& b, size_t n){
    std::vector<T> out;
    for (size_t i = 0; i <= n; i++){
        out.push_back(a + (b - a) * T(i) / T(n));
    }
    return out;
}

template<typename SD, typename T>
state_t<T, SD> evaluate(const BoxedInterp<T, SD>& dense, const T& t, const state_t<T, SD>& like){
    auto out = like;
    dense->call(out, t);
    return out;
}

template<typename Body>
void for_each_stepper_and_kind(Body&& body){
    for_each_type<AllSteppers>([&]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            for_each_type<StateKinds<2>>([&]<typename SD>(){
                DYNAMIC_SECTION(kind_name<SD>){
                    body.template operator()<Stepper, SD>();
                }
            });
        }
    });
}


TEMPLATE_TEST_CASE("interpolate_until gives the dense output of exactly the span it integrated", "[dense-output]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        const double bound = settings<Stepper, T>().error;
        for (int dir : {1, -1}){
            CAPTURE(dir);
            const T t0 = T(0.5), t_end = T(0.5 + 6 * dir);
            const auto q0 = Oscillator::at<SD>(t0);
            auto solver = make_solver<Stepper, SD>(Oscillator{}, t0, q0, dir);
            const BoxedInterp<T, SD> dense = solver.interpolate_until(t_end);

            REQUIRE(solver.t() == t_end);
            CHECK(dense->t_start() == t0);
            CHECK(dense->t_end() == t_end);
            CHECK(dense->dir() == dir);
            // Its ends are the states the solver started and ended at, exactly
            CHECK(same_state<SD>(evaluate(dense, t0, q0), q0));
            CHECK(same_state<SD>(evaluate(dense, t_end, q0), current_state(solver)));
            for (const T& t : times_across(t0, t_end, 97)){
                CAPTURE(to_double(t));
                CHECK_THAT(max_error<SD>(evaluate(dense, t, q0), Oscillator::at<SD>(t)), IsAtMost(bound));
            }
            CHECK_THROWS_AS(evaluate(dense, T(t_end + T(0.1 * dir)), q0), std::runtime_error);
            CHECK_THROWS_AS(evaluate(dense, T(t0 - T(0.1 * dir)), q0), std::runtime_error);
        }
    });
}

TEMPLATE_TEST_CASE("Consecutive interpolate_until calls cover consecutive spans", "[dense-output]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            using SD = StaticSize<2>;
            const auto q0 = Oscillator::at<SD>(T(0));
            auto solver = make_solver<Stepper, SD>(Oscillator{}, T(0), q0);
            const auto first = solver.interpolate_until(T(1));
            const auto second = solver.interpolate_until(T(2.5));

            CHECK((first->t_start() == T(0) && first->t_end() == T(1)));
            CHECK((second->t_start() == T(1) && second->t_end() == T(2.5)));
            CHECK(same_state<SD>(evaluate(first, T(1), q0), evaluate(second, T(1), q0)));
            CHECK_THAT(max_error<SD>(evaluate(second, T(1.7), q0), Oscillator::at<SD>(T(1.7))), IsAtMost(settings<Stepper, T>().error));
        }
    });
}

TEMPLATE_TEST_CASE("interpolate_until reports every step to its observer", "[dense-output][observer]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        const auto q0 = Oscillator::at<SD>(T(0));
        auto solver = make_solver<Stepper, SD>(Oscillator{}, T(0), q0);

        SECTION("with the state at each of them"){
            size_t calls = 0;
            const auto dense = solver.interpolate_until(T(1), [&](const T& t, const auto& q, const T* /*t_ptr*/){
                ++calls;
                CHECK(t == solver.t());
                CHECK(same_state<SD>(q, current_state(solver)));
                return true;
            });
            CHECK(calls == solver.step_count());
            CHECK(dense->t_end() == T(1));
        }
        SECTION("which may stop it, the dense output then ending there"){
            const auto dense = solver.interpolate_until(T(1), [&](const T& /*t*/, const auto& /*q*/, const T* /*t_ptr*/){
                return solver.step_count() < 3;
            });
            CHECK(solver.step_count() == 3);
            CHECK(dense->t_end() == solver.t());
            CHECK(same_state<SD>(evaluate(dense, solver.t(), q0), current_state(solver)));
        }
    });
}

TEMPLATE_TEST_CASE("Dense output runs smoothly through events that change nothing", "[dense-output][events]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            using SD = StaticSize<2>;
            using State = state_t<T, SD>;
            const auto q0 = Oscillator::at<SD>(T(0));
            auto solver = make_solver<Stepper, SD>(Oscillator{}, T(0), q0, 1, ode::make_event_list<T, State>(
                ode::make_precise_event<T, State>("x = 0", [](const T& /*t*/, const State& q) -> T { return q[0]; }, T(0)),
                ode::make_periodic_event<T, State>("tick", T(0.7))));
            const auto dense = solver.interpolate_until(T(5));
            for (const T& t : times_across(T(0), T(5), 113)){
                CHECK_THAT(max_error<SD>(evaluate(dense, t, q0), Oscillator::at<SD>(t)), IsAtMost(settings<Stepper, T>().error));
            }
        }
    });
}


/// @brief Every time x reaches 0.5*dir, the mask moves it back by `shift`*dir.
template<typename SD, typename T>
auto teleport(int dir, bool delayed, double shift){
    return ode::make_precise_event<T, state_t<T, SD>>("teleport",
        [dir](const T& /*t*/, const state_t<T, SD>& q) -> T { return q[0] - T(0.5 * dir); }, T(0), 0,
        [dir, shift](state_t<T, SD>& out, const T& /*t*/, const state_t<T, SD>& q){
            out = q;
            out[0] = q[0] - T(shift * dir);
        }, delayed);
}

/// @brief The dense output of uniform motion in unit steps with a teleporting mask, and the time and
/// state the solver reported at each event on the way.
template<typename Stepper, typename SD, typename T>
auto teleporting_dense_output(int dir, bool delayed, double shift, const T& t_end){
    const auto q0 = make_state<SD, T>({0, 0});
    const auto loose = uniform_like<SD, T>(q0, T(1));
    auto solver = ode::getSolver<SD, SolverPolicy::RichStatic>(Stepper{}, UniformMotion{}, T(0), q0, loose, loose, T(0), T(1), T(1), dir,
        ode::make_event_list<T, state_t<T, SD>>(teleport<SD, T>(dir, delayed, shift)));
    std::vector<std::pair<T, state_t<T, SD>>> events;
    // The state is read from the solver: what the observer is handed is tested on its own
    auto dense = solver.interpolate_until(t_end, [&](const T& t, const auto& /*q*/, const T* /*t_ptr*/){
        if (solver.at_event()){
            events.emplace_back(t, current_state(solver));
        }
        return true;
    });
    return std::pair{std::move(dense), std::move(events)};
}

TEMPLATE_TEST_CASE("Dense output jumps where a masked event moves the state", "[dense-output][events][mask]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([&]<typename Stepper, typename SD>(){
        const double tol = uniform_motion_tolerance<Stepper, T>();
        for (int dir : {1, -1}){
            for (bool delayed : {false, true}){
                CAPTURE(dir, delayed);
                // x = t until it reaches 0.5, which moves it back by 0.75: then x = t - 0.75 until it
                // reaches 0.5 again at t = 1.25, x = t - 1.5 until t = 2, and so on (all times dir).
                // Every stepper and dense output follows uniform motion exactly.
                // Not a structured binding: check_x captures dense, which Clang only allows from
                // version 16, and never with OpenMP on (-fopenmp-simd)
                const auto result = teleporting_dense_output<Stepper, SD>(dir, delayed, 0.75, T(2.6 * dir));
                const auto& dense = result.first;
                const auto& events = result.second;
                REQUIRE(dense->t_end() == T(2.6 * dir));
                REQUIRE(events.size() == 3);

                // x = t - offset at t (times dir), computed in T, as the solvers do
                const auto q0 = make_state<SD, T>({0, 0});
                const auto check_x = [&](double t, double offset){
                    CAPTURE(t);
                    const T at = T(t * dir);
                    const auto q = evaluate(dense, at, q0);
                    CHECK_THAT(abs_error(q[0], T(at - T(offset * dir))), IsAtMost(tol));
                    CHECK_THAT(abs_error(q[1], T(-at)), IsAtMost(tol));   // y never jumps
                };
                const double just = 1e-3;
                check_x(0.25, 0);
                check_x(0.5 - just, 0);
                check_x(0.5 + just, 0.75);
                check_x(1.0, 0.75);
                check_x(1.25 - just, 0.75);
                check_x(1.25 + just, 1.5);
                check_x(1.6, 1.5);
                check_x(2.0 + just, 2.25);
                check_x(2.55, 2.25);

                // At each event itself, the state the solver reported there: the moved one for an
                // immediate mask, the one before the move for a delayed mask
                for (const auto& [t_event, q_event] : events){
                    CAPTURE(to_double(t_event));
                    CHECK_THAT(abs_error(q_event[0], T((delayed ? 0.5 : -0.25) * dir)), IsAtMost(tol));
                    CHECK(same_state<SD>(evaluate(dense, t_event, q0), q_event));
                }
            }
        }
    });
}

TEMPLATE_TEST_CASE("Dense output agrees with a masked event at the very end of a step", "[dense-output][events][mask]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            using SD = StaticSize<2>;
            for (bool delayed : {false, true}){
                CAPTURE(delayed);
                // Moved back by a whole unit, x reaches 0.5 again exactly where each unit step ends
                const auto [dense, events] = teleporting_dense_output<Stepper, SD>(1, delayed, 1.0, T(2.6));
                REQUIRE(events.size() == 3);
                for (const auto& [t_event, q_event] : events){
                    CAPTURE(to_double(t_event));
                    CHECK(same_state<SD>(evaluate(dense, t_event, make_state<SD, T>({0, 0})), q_event));
                }
            }
        }
    });
}

TEMPLATE_TEST_CASE("Dense output ends where a solver that fails stops", "[dense-output][failure]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        const auto q0 = Oscillator::at<SD>(T(0));

        SECTION("midway"){
            auto solver = make_solver<Stepper, SD>(PoisonedOscillator{.t_fail = 0.5}, T(0), q0);
            const auto dense = solver.interpolate_until(T(2));
            REQUIRE_FALSE(solver.is_running());
            CHECK(dense->t_start() == T(0));
            CHECK(dense->t_end() == solver.t());
            for (const T& t : times_across(T(0), solver.t(), 31)){
                CHECK_THAT(max_error<SD>(evaluate(dense, t, q0), Oscillator::at<SD>(t)), IsAtMost(settings<Stepper, T>().error));
            }
            CHECK_THROWS_AS(evaluate(dense, T(solver.t() + T(1e-3)), q0), std::runtime_error);
        }
        SECTION("at its first step, leaving only its initial state"){
            // Fine at t0 = 0, NaN anywhere after it. Euler only evaluates it at the start of a step,
            // so for Euler that is its second step.
            if constexpr (std::is_same_v<Stepper, tags::Euler>){
                return;
            }
            auto solver = make_solver<Stepper, SD>(PoisonedOscillator{.t_fail = 0.0}, T(0), q0);
            const auto dense = solver.interpolate_until(T(2));
            REQUIRE_FALSE(solver.is_running());
            REQUIRE(solver.step_count() == 0);
            CHECK(dense->t_start() == T(0));
            CHECK(dense->t_end() == T(0));
            CHECK(same_state<SD>(evaluate(dense, T(0), q0), q0));
            CHECK_THROWS_AS(evaluate(dense, T(1e-3), q0), std::runtime_error);
        }
    });
}

TEMPLATE_TEST_CASE("Dense output through the virtual interface", "[dense-output][virtual]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    using State = state_t<T, SD>;
    const Settings<T> s = settings<tags::RK45, T>();
    const auto q0 = Oscillator::at<SD>(T(0));
    auto solver = ode::make_rich_vsolver<SD>(tags::RK45{}, Oscillator{}, T(0), q0, uniform_like<SD, T>(q0, s.rtol), uniform_like<SD, T>(q0, s.atol));

    const auto plain = solver->do_interpolate_until(T(1));
    size_t calls = 0;
    const ode::observer_t<T, State> observer = [&calls](const T&, const State&, const T*){ ++calls; return true; };
    const auto observed = solver->do_interpolate_until(T(2), observer);
    const auto unobserved = solver->do_interpolate_until(T(3), nullptr);

    CHECK((plain->t_end() == T(1) && observed->t_start() == T(1) && unobserved->t_end() == T(3)));
    CHECK(calls > 0);
    CHECK_THAT(max_error<SD>(evaluate(observed, T(1.5), q0), Oscillator::at<SD>(T(1.5))), IsAtMost(s.error));
}

} // namespace
} // namespace odetest
