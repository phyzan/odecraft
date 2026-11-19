// A solver that stops, for whatever reason, keeps its last accepted state and nothing else: its
// old and new states collapse into that one, so that nothing can be interpolated across the
// step that failed, and the state it reports is that last state.

#include "odecraft_test.hpp"

namespace odetest {
namespace {

using ode::SolverPolicy;
using ode::StepperStatus;

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

/// @brief A solver of the oscillator that diverges past t = 0.5 in direction `dir`, run until it does.
template<typename Stepper, typename SD, typename T>
auto diverged_solver(int dir){
    const T t0 = 0;
    auto solver = make_solver<Stepper, SD, SolverPolicy::RichStatic>(PoisonedOscillator{.t_fail = 0.5 * dir, .dir = dir}, t0, Oscillator::at<SD>(t0), dir);
    REQUIRE_FALSE(solver.advance_until(T(2 * dir)));
    return solver;
}

// q' = q^2 from q(0) = 1, in each of two components: q = 1/(1 - t), which blows up at t = 1.
// Not one component: GCC 13 takes the copies of a one-element vector for out of bounds.
struct Blowup {
    static void Rhs(auto& out, const auto& /*t*/, const auto& q){
        out[0] = q[0]*q[0];
        out[1] = q[1]*q[1];
    }
};

/// @brief A solver of q' = q^2, run into its singularity at t = 1.
template<typename Stepper, typename SD, typename T>
auto singular_solver(){
    auto solver = make_solver<Stepper, SD, SolverPolicy::Static>(Blowup{}, T(0), make_state<SD, T>({1, 1}));
    REQUIRE_FALSE(solver.advance_until(T(2)));
    return solver;
}

/// @brief BDF on a stiff decay whose transient needs steps far below the minimum allowed, run until it stops.
template<typename SD, typename T>
auto min_step_solver(){
    const auto q0 = make_state<SD, T>({2});
    const auto tols = make_state<SD, T>({T(1e-8)});
    auto solver = ode::getSolver<SD, SolverPolicy::Static>(tags::BDF{}, StiffDecay{}, T(0), q0, tols, tols, T(0.1), T(0), T(0.2));
    REQUIRE_FALSE(solver.advance_until(T(5)));
    return solver;
}

/// @brief The old and new state of a stopped solver are one and the same, at its current time.
template<typename SD, typename Solver>
void check_collapsed(const Solver& solver){
    CHECK_FALSE(solver.is_running());
    CHECK(solver.t_old() == solver.t_new());
    CHECK(solver.t() == solver.t_new());
    CHECK(same_state<SD>(solver.vector_old(), solver.vector_new()));
}


TEMPLATE_TEST_CASE("A solver that diverges stops on its last good state", "[failure]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        for (int dir : {1, -1}){
            CAPTURE(dir);
            const auto solver = diverged_solver<Stepper, SD, T>(dir);
            CHECK(solver.status() == StepperStatus::Diverges);
            CHECK(solver.message() == "Integrator diverges");
            check_collapsed<SD>(solver);

            // It stopped short of where the right-hand side breaks, and the state it kept is sound.
            // Euler only evaluates it at the start of a step, so its last one may end one step past.
            const double reach = std::is_same_v<Stepper, tags::Euler> ? to_double(solver.stepsize()) : 0.0;
            CHECK(dir * to_double(solver.t()) <= 0.5 + reach);
            CHECK(ode::all_are_finite<T, SD>(solver.vector_new()));
            CHECK_THAT(max_error<SD>(solver.vector_new(), Oscillator::at<SD>(solver.t())), IsAtMost(settings<Stepper, T>().error));
        }
    });
}

TEMPLATE_TEST_CASE("A stopped solver interpolates nowhere but at its last state", "[failure][interpolation]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        const auto solver = diverged_solver<Stepper, SD, T>(1);
        const T t = solver.t();
        const T elsewhere = t - T(1e-3);
        auto out = solver.vector_new();

        SECTION("its step interpolator is a single point"){
            const auto step = solver.state_interpolator(1, 1);
            CHECK(step->t_start() == t);
            CHECK(step->t_end() == t);
            step->call(out, t);
            CHECK(same_state<SD>(out, solver.vector_new()));
            CHECK_THROWS_AS(step->call(out, elsewhere), std::runtime_error);
        }
        SECTION("light_interp and heavy_interp answer at that time only"){
            auto light = solver.light_interp();
            light(out, t);
            CHECK(same_state<SD>(out, solver.vector_new()));
            CHECK_THROWS_AS(light(out, elsewhere), std::domain_error);

            auto heavy = solver.heavy_interp();
            heavy(out, t);
            CHECK(same_state<SD>(out, solver.vector_new()));
            CHECK_THROWS_AS(heavy(out, elsewhere), std::domain_error);
        }
    });
}

TEMPLATE_TEST_CASE("A stopped solver refuses to advance until it is restarted", "[failure]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper_and_kind([]<typename Stepper, typename SD>(){
        auto solver = diverged_solver<Stepper, SD, T>(1);
        const T t = solver.t();
        const size_t steps = solver.step_count();

        CHECK_FALSE(solver.advance());
        CHECK_FALSE(solver.advance_until(T(3)));
        CHECK_FALSE(solver.advance_by(T(1)));
        CHECK(solver.t() == t);
        CHECK(solver.step_count() == steps);

        SECTION("Reset starts it over"){
            solver.Reset();
            CHECK(solver.is_running());
            CHECK(solver.t() == T(0));
            CHECK(solver.advance_until(T(0.25)));
        }
        SECTION("SetICs starts it from new conditions"){
            REQUIRE(solver.SetICs(T(0.1), Oscillator::at<SD>(T(0.1)), settings<Stepper, T>().stepsize));
            CHECK(solver.is_running());
            CHECK(solver.advance_until(T(0.3)));
        }
    });
}


TEMPLATE_TEST_CASE("A solver that runs into a singularity stops with its states collapsed", "[failure]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            for_each_type<StateKinds<2>>([]<typename SD>(){
                DYNAMIC_SECTION(kind_name<SD>){
                    const auto solver = singular_solver<Stepper, SD, T>();
                    CAPTURE(solver.message());
                    CHECK(solver.status() != StepperStatus::NewStep);
                    CHECK(solver.status() != StepperStatus::BetweenSteps);
                    check_collapsed<SD>(solver);
                    // An adaptive stepper sees its steps vanish at the singularity, give or take
                    // its own error, a fixed-step one only diverges past it
                    if constexpr (!is_fixed_step<Stepper>){
                        CHECK_THAT(abs_error(solver.t(), T(1)), IsAtMost(1e-3));
                    }
                }
            });
        }
    });
}

TEMPLATE_TEST_CASE("BDF stops at its minimum step with its states collapsed", "[failure][bdf]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<StateKinds<1>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            const auto solver = min_step_solver<SD, T>();
            CHECK(solver.status() == StepperStatus::MinStep);
            CHECK(solver.message() == "Stepsize reached its minimum");
            check_collapsed<SD>(solver);
        }
    });
}


// FillCurrentState, and with it interpolate_until, read the current state. Kept to test cases of
// their own: a solver that gets this wrong may abort rather than fail a check.

/// @brief Runs check.template operator()<SD>(solver) on a solver stopped in each way a solver stops
/// by itself, each in a section of its own: diverged, run into a singularity, or held above the
/// steps it needs.
template<typename T, typename Check>
void for_each_way_to_stop(Check&& check){
    for_each_type<AllSteppers>([&]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            for_each_type<StateKinds<2>>([&]<typename SD>(){
                DYNAMIC_SECTION("diverged with a " << kind_name<SD>){
                    auto solver = diverged_solver<Stepper, SD, T>(1);
                    check.template operator()<SD>(solver);
                }
            });
            for_each_type<StateKinds<2>>([&]<typename SD>(){
                DYNAMIC_SECTION("at a singularity with a " << kind_name<SD>){
                    auto solver = singular_solver<Stepper, SD, T>();
                    check.template operator()<SD>(solver);
                }
            });
        }
    });
    for_each_type<StateKinds<1>>([&]<typename SD>(){
        DYNAMIC_SECTION("BDF at its minimum step with a " << kind_name<SD>){
            auto solver = min_step_solver<SD, T>();
            check.template operator()<SD>(solver);
        }
    });
}

TEMPLATE_TEST_CASE("The current state of a stopped solver is its last state", "[failure]", ODECRAFT_TEST_SCALARS){
    for_each_way_to_stop<TestType>([]<typename SD>(const auto& solver){
        CAPTURE(solver.message());
        auto out = solver.vector_old();
        solver.FillCurrentState(out);
        CHECK(same_state<SD>(out, solver.vector_new()));
    });
}

TEMPLATE_TEST_CASE("Dense output from a stopped solver is its last state alone", "[failure][interpolation]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_way_to_stop<T>([]<typename SD>(auto& solver){
        CAPTURE(solver.message());
        const T t = solver.t();
        const auto dense = solver.interpolate_until(T(t + T(1)));
        CHECK(dense->t_start() == t);
        CHECK(dense->t_end() == t);
        auto out = solver.vector_new();
        dense->call(out, t);
        CHECK(same_state<SD>(out, solver.vector_new()));
    });
}

} // namespace
} // namespace odetest
