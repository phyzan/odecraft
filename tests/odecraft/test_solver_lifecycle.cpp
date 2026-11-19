// The life of a solver: what it reports once built, how each advancing routine moves it, and how
// it is restarted from new initial conditions, reset and cloned.

#include "../support/odecraft_test.hpp" // IWYU pragma: keep

#include <vector>

namespace odetest {
namespace {

using ode::SolverPolicy;
using ode::StepperStatus;

/// @brief A static-policy solver of the oscillator on its reference trajectory, from t0.
template<typename Stepper, typename SD, typename T>
auto oscillator_solver(const T& t0, int dir = 1){
    return make_solver<Stepper, SD, SolverPolicy::Static>(Oscillator{}, t0, Oscillator::at<SD>(t0), dir);
}

/// @brief Whether the solver's current state is on the reference trajectory.
template<typename Stepper, typename SD, typename Solver>
void check_on_trajectory(const Solver& solver){
    using T = typename Solver::real_type;
    CHECK_THAT(max_error<SD>(current_state(solver), Oscillator::at<SD>(solver.t())), IsAtMost(settings<Stepper, T>().error));
}

/// @brief Runs body.template operator()<Stepper, SD>() for every stepper and kind of state, each in a section.
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


TEMPLATE_TEST_CASE("A new solver reports its initial conditions and settings", "[lifecycle]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        const Settings<T> s = settings<Stepper, T>();
        const T t0 = T(0.5);
        const auto q0 = Oscillator::at<SD>(t0);
        const auto tol = uniform_like<SD, T>(q0, T(1e-6));
        const T min_step = 0;
        const T max_step = T(0.25);
        // Fixed-step steppers ignore the tolerances and step limits, and need a step
        auto solver = ode::getSolver<SD, SolverPolicy::Static>(Stepper{}, Oscillator{}, t0, q0, tol, tol, min_step, max_step, s.stepsize, -1);

        CHECK(solver.is_running());
        CHECK(solver.status() == StepperStatus::NewStep);
        CHECK(solver.message() == "Running");
        CHECK(solver.has_valid_ics());
        CHECK(solver.method() == Stepper::name);
        CHECK(solver.step_count() == 0);
        CHECK(solver.direction() == -1);

        CHECK((solver.t() == t0 && solver.t_new() == t0 && solver.t_old() == t0 && solver.t_ics() == t0));
        CHECK(same_state<SD>(solver.vector_new(), q0));
        CHECK(same_state<SD>(solver.vector_old(), q0));
        CHECK(same_state<SD>(solver.vector_ics(), q0));
        CHECK(same_state<SD>(current_state(solver), q0));
        // The size first: comparing, or computing a step from, tolerances of the wrong size reads past their end
        REQUIRE(size_of<SD>(solver.rtol()) == size_of<SD>(q0));
        REQUIRE(size_of<SD>(solver.atol()) == size_of<SD>(q0));
        CHECK(same_state<SD>(solver.rtol(), tol));
        CHECK(same_state<SD>(solver.atol(), tol));

        if constexpr (is_fixed_step<Stepper>){
            CHECK(solver.stepsize() == s.stepsize);
        } else {
            CHECK(solver.min_step() == min_step);
            CHECK(solver.max_step() == max_step);
            CHECK(solver.stepsize() > 0);
            CHECK(solver.stepsize() <= max_step);
        }
        CHECK(solver.auto_step() > 0);
        CHECK(solver.auto_step(t0, q0) == solver.auto_step());
        CHECK(solver.ValidateICs(t0, q0));
    });
}

TEMPLATE_TEST_CASE("A solver without a maximum step is unbounded", "[lifecycle]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const auto solver = oscillator_solver<tags::RK45, SD>(T(0));
    CHECK(solver.max_step() == ode::inf<T>());
}

TEMPLATE_TEST_CASE("Invalid settings are rejected at construction", "[lifecycle]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        const auto q0 = Oscillator::at<SD>(T(0));
        const auto tol = uniform_like<SD, T>(q0, T(1e-6));
        const auto build = [&](T min_step, T max_step, T stepsize, int dir){
            return ode::getSolver<SD, SolverPolicy::Static>(Stepper{}, Oscillator{}, T(0), q0, tol, tol, min_step, max_step, stepsize, dir);
        };
        CHECK_THROWS_AS(build(T(0), T(0), T(0), 0), std::runtime_error);
        CHECK_THROWS_AS(build(T(0), T(0), T(0), 2), std::runtime_error);
        CHECK_THROWS_AS(build(T(0), T(0), T(-1), 1), std::runtime_error);
        if constexpr (!is_fixed_step<Stepper>){
            CHECK_THROWS_AS(build(T(1), T(0.5), T(0), 1), std::runtime_error);
        }

        // Events need a policy with event support
        auto events = ode::make_event_list<T, state_t<T, SD>>(ode::make_periodic_event<T, state_t<T, SD>>("tick", T(1)));
        CHECK_THROWS_AS((ode::getSolver<SD, SolverPolicy::Static>(Stepper{}, Oscillator{}, T(0), q0, tol, tol, T(0), T(0), T(0), 1, std::move(events))), std::invalid_argument);
    });
}

TEMPLATE_TEST_CASE("Non-finite initial conditions leave the solver unable to run", "[lifecycle]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        SECTION("a NaN in the state"){
            auto q0 = Oscillator::at<SD>(T(0));
            q0[1] = T(std::numeric_limits<double>::quiet_NaN());
            auto solver = make_solver<Stepper, SD, SolverPolicy::Static>(Oscillator{}, T(0), q0);
            CHECK(solver.status() == StepperStatus::BadICS);
            CHECK_FALSE(solver.is_running());
            CHECK_FALSE(solver.has_valid_ics());
            CHECK_FALSE(solver.advance());
            CHECK_FALSE(solver.advance_until(T(1)));
            CHECK(solver.step_count() == 0);
        }
        SECTION("a right-hand side that is NaN there"){
            auto solver = make_solver<Stepper, SD, SolverPolicy::Static>(PoisonedOscillator{.t_fail = -1}, T(0), Oscillator::at<SD>(T(0)));
            CHECK(solver.status() == StepperStatus::BadICS);
            CHECK_FALSE(solver.ValidateICs(T(0), Oscillator::at<SD>(T(0))));
        }
    });
}


TEMPLATE_TEST_CASE("advance takes one step at a time in the solver's direction", "[lifecycle][advance]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        for (int dir : {1, -1}){
            CAPTURE(dir);
            const T t0 = T(0.5);
            auto solver = oscillator_solver<Stepper, SD>(t0, dir);
            T t_previous = t0;
            for (size_t step = 1; step <= 5; step++){
                REQUIRE(solver.advance());
                CHECK(solver.step_count() == step);
                CHECK(solver.t() == solver.t_new());
                CHECK(solver.t_old() == t_previous);
                CHECK(ode::sgn(solver.t_old(), solver.t_new()) == dir);
                check_on_trajectory<Stepper, SD>(solver);
                t_previous = solver.t();
            }
            CHECK(solver.status() == StepperStatus::NewStep);
        }
    });
}

TEMPLATE_TEST_CASE("advance_until lands exactly on its target", "[lifecycle][advance]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        for (int dir : {1, -1}){
            CAPTURE(dir);
            auto solver = oscillator_solver<Stepper, SD>(T(0), dir);
            const T target = T(0.7 * dir);
            REQUIRE(solver.advance_until(target));
            CHECK(solver.t() == target);
            CHECK(solver.is_running());
            check_on_trajectory<Stepper, SD>(solver);
            // A target inside a step leaves the solver between steps, interpolating its state
            if (solver.t() != solver.t_new()){
                CHECK(solver.status() == StepperStatus::BetweenSteps);
                CHECK(ode::sgn(solver.t_old(), target) == dir);
                CHECK(ode::sgn(target, solver.t_new()) == dir);
            }

            CHECK_FALSE(solver.advance_until(target));                 // already there
            CHECK_THROWS_AS(solver.advance_until(T(-dir)), std::runtime_error); // behind it
            REQUIRE(solver.advance_until(T(1.3 * dir)));               // and on it goes
            check_on_trajectory<Stepper, SD>(solver);
        }
    });
}

TEMPLATE_TEST_CASE("An observer sees every step and can stop the integration", "[lifecycle][advance][observer]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        auto solver = oscillator_solver<Stepper, SD>(T(0));
        const T target = T(1);

        SECTION("every step, with the target time on the last call only"){
            size_t calls = 0;
            const T* last_pointer = nullptr;
            const bool reached = solver.advance_until(target, [&](const T& t, const auto& q, const T* t_ptr){
                ++calls;
                CHECK(t == solver.t());
                CHECK(same_state<SD>(q, current_state(solver)));
                if (t != target){
                    CHECK(t_ptr == nullptr);
                }
                last_pointer = t_ptr;
                return true;
            });
            CHECK(reached);
            CHECK(calls == solver.step_count());
            REQUIRE(last_pointer != nullptr);
            CHECK(*last_pointer == target);
        }
        SECTION("returning false stops right there, and the integration can resume"){
            const bool reached = solver.advance_until(target, [](const T& /*t*/, const auto& /*q*/, const T* /*t_ptr*/){
                return false;
            });
            CHECK_FALSE(reached);
            CHECK(solver.step_count() == 1);
            CHECK(solver.t() == solver.t_new());
            CHECK(solver.advance_until(target));
            CHECK(solver.t() == target);
        }
    });
}

TEMPLATE_TEST_CASE("advance_until stops exactly at every checkpoint", "[lifecycle][advance][checkpoints]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        for (int dir : {1, -1}){
            CAPTURE(dir);
            auto solver = oscillator_solver<Stepper, SD>(T(0), dir);
            const std::vector<T> checkpoints = {T(0.3 * dir), T(0.35 * dir), T(1.2 * dir)};
            const T target = T(2 * dir);

            std::vector<T> landed;
            REQUIRE(solver.advance_until(target, [&](const T& t, const auto& q, const T* t_ptr){
                if (t_ptr != nullptr){
                    CHECK(*t_ptr == t);
                    landed.push_back(t);
                    CHECK_THAT(max_error<SD>(q, Oscillator::at<SD>(t)), IsAtMost(settings<Stepper, T>().error));
                }
                return true;
            }, checkpoints));
            CHECK(landed == checkpoints);
            CHECK(solver.t() == target);

            // Checkpoints must lie ahead of the solver and not beyond the target
            CHECK_THROWS_AS(solver.advance_until(T(3 * dir), [](const T&, const auto&, const T*){ return true; },
                                                 std::vector<T>{T(4 * dir)}), std::runtime_error);
            CHECK_THROWS_AS(solver.advance_until(T(3 * dir), [](const T&, const auto&, const T*){ return true; },
                                                 std::vector<T>{T(dir)}), std::runtime_error);
        }
    });
}

TEMPLATE_TEST_CASE("advance_by moves by an interval in the solver's direction", "[lifecycle][advance]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        for (int dir : {1, -1}){
            CAPTURE(dir);
            auto solver = oscillator_solver<Stepper, SD>(T(0.25), dir);
            REQUIRE(solver.advance_by(T(0.5)));
            CHECK(solver.t() == T(0.25) + T(0.5) * dir);
            check_on_trajectory<Stepper, SD>(solver);
            CHECK_THROWS_AS(solver.advance_by(T(-0.5)), std::logic_error);
        }
    });
}


TEMPLATE_TEST_CASE("SetICs restarts the solver from new initial conditions", "[lifecycle][ics]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        // A fixed-step stepper is restarted with its own step, an adaptive one picks its first again
        const T step = settings<Stepper, T>().stepsize;
        auto solver = oscillator_solver<Stepper, SD>(T(0));
        REQUIRE(solver.advance_until(T(1)));
        const T t1 = T(2);
        const auto q1 = Oscillator::at<SD>(t1);

        SECTION("in the same direction"){
            REQUIRE(solver.SetICs(t1, q1, step));
            CHECK(solver.t() == t1);
            CHECK(solver.t_ics() == t1);
            CHECK(same_state<SD>(solver.vector_ics(), q1));
            CHECK(same_state<SD>(current_state(solver), q1));
            CHECK(solver.step_count() == 0);
            CHECK(solver.status() == StepperStatus::NewStep);
            CHECK(solver.direction() == 1);
            CHECK(solver.stepsize() > 0);
            REQUIRE(solver.advance_until(T(3)));
            check_on_trajectory<Stepper, SD>(solver);
        }
        SECTION("backward, with a step of its own"){
            REQUIRE(solver.SetICs(t1, q1, T(1e-3), -1));
            CHECK(solver.direction() == -1);
            CHECK(solver.stepsize() == T(1e-3));
            REQUIRE(solver.advance_until(T(1)));
            check_on_trajectory<Stepper, SD>(solver);
        }
        SECTION("invalid ones are refused, and the solver carries on as it was"){
            auto bad = q1;
            bad[0] = T(std::numeric_limits<double>::quiet_NaN());
            CHECK_FALSE(solver.SetICs(t1, bad));
            CHECK_FALSE(solver.SetICs(t1, q1, T(-1)));
            CHECK_THROWS_AS(solver.SetICs(t1, q1, T(0), 2), std::domain_error);

            CHECK(solver.t() == T(1));
            CHECK(solver.is_running());
            REQUIRE(solver.advance_until(T(1.5)));
            check_on_trajectory<Stepper, SD>(solver);
        }
    });
}

TEMPLATE_TEST_CASE("Reset returns the solver to its initial conditions and reproduces its run", "[lifecycle][reset]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        auto solver = oscillator_solver<Stepper, SD>(T(0));
        REQUIRE(solver.advance_until(T(1.5)));
        const auto first_run = solver.vector_new();
        const size_t first_steps = solver.step_count();

        solver.Reset();
        CHECK(solver.is_running());
        CHECK(solver.t() == T(0));
        CHECK(solver.step_count() == 0);
        CHECK(same_state<SD>(solver.vector_new(), solver.vector_ics()));

        REQUIRE(solver.advance_until(T(1.5)));
        CHECK(solver.step_count() == first_steps);
        CHECK(same_state<SD>(solver.vector_new(), first_run));
    });
}


TEMPLATE_TEST_CASE("A clone continues independently and identically", "[lifecycle][clone]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        auto solver = oscillator_solver<Stepper, SD>(T(0));
        REQUIRE(solver.advance_until(T(0.75)));

        using Solver = decltype(solver);
        STATIC_REQUIRE(std::is_same_v<typename Solver::clone_type, Solver>);
        std::unique_ptr<Solver> copy = solver.clone();
        CHECK(copy->t() == solver.t());
        CHECK(same_state<SD>(current_state(*copy), current_state(solver)));

        REQUIRE(copy->advance_until(T(2)));
        CHECK(solver.t() == T(0.75));   // the original stayed where it was

        REQUIRE(solver.advance_until(T(2)));
        CHECK(same_state<SD>(solver.vector_new(), copy->vector_new()));
        CHECK(solver.step_count() == copy->step_count());
    });
}


TEMPLATE_TEST_CASE("The dense output of the last step", "[lifecycle][interpolation]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        const double bound = settings<Stepper, T>().error;
        auto solver = oscillator_solver<Stepper, SD>(T(0));
        auto out = solver.vector_new();

        SECTION("before the first step it is a single point"){
            auto point = solver.state_interpolator(1, 1);
            CHECK(point->t_start() == point->t_end());
            point->call(out, T(0));
            CHECK(same_state<SD>(out, solver.vector_new()));
        }

        REQUIRE(solver.advance());
        REQUIRE(solver.advance());
        const T t_old = solver.t_old(), t_new = solver.t_new();
        const T t_mid = (t_old + t_new) / 2;

        SECTION("light_interp covers the closed step, and only that"){
            auto f = solver.light_interp();
            f(out, t_old);
            CHECK(same_state<SD>(out, solver.vector_old()));
            f(out, t_new);
            CHECK(same_state<SD>(out, solver.vector_new()));
            f(out, t_mid);
            CHECK_THAT(max_error<SD>(out, Oscillator::at<SD>(t_mid)), IsAtMost(bound));
            CHECK_THROWS_AS(f(out, T(t_new + (t_new - t_old))), std::domain_error);
        }
        SECTION("heavy_interp keeps the step after the solver has moved on"){
            auto f = solver.heavy_interp();
            REQUIRE(solver.advance_until(T(1)));
            f(out, t_mid);
            CHECK_THAT(max_error<SD>(out, Oscillator::at<SD>(t_mid)), IsAtMost(bound));
            f(out, t_new);
            CHECK_THAT(max_error<SD>(out, Oscillator::at<SD>(t_new)), IsAtMost(bound));
            CHECK_THROWS_AS(f(out, T(t_old - (t_new - t_old))), std::domain_error);
        }
        SECTION("state_interpolator owns its data and has the ends it is asked for"){
            auto f = solver.state_interpolator(1, -1);
            REQUIRE(solver.advance_until(T(1)));
            CHECK((f->t_start() == t_old && f->t_end() == t_new));
            CHECK((f->start_bdr() == 1 && f->end_bdr() == -1));
            f->call(out, t_mid);
            CHECK_THAT(max_error<SD>(out, Oscillator::at<SD>(t_mid)), IsAtMost(bound));
            CHECK_THROWS_AS(f->call(out, t_new), std::runtime_error);
        }
    });
}

TEMPLATE_TEST_CASE("A solver evaluates its system and prints its state", "[lifecycle]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    auto solver = oscillator_solver<tags::RK45, SD>(T(0));
    REQUIRE(solver.advance_until(T(0.5)));

    auto out = solver.vector_new();
    solver.rhs(out, T(0), make_state<SD, T>({2, 3}));
    CHECK(same_state<SD>(out, make_state<SD, T>({3, -2})));

    std::ostringstream text;
    solver.ShowState(text, 6);
    CHECK_THAT(text.str(), Catch::Matchers::ContainsSubstring("time") && Catch::Matchers::ContainsSubstring("0.5")
                         && Catch::Matchers::ContainsSubstring("Running"));
    std::ostringstream through_alias;
    solver.show_state(through_alias, 6);
    CHECK(through_alias.str() == text.str());
}

} // namespace
} // namespace odetest
