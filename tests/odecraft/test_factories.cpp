// Building solvers: getSolver for every policy, the boxing factories, the stepper catalogs and
// their ids, a stepper declared outside the library, and the virtual interface the boxed
// solvers are used through.

#include "../support/odecraft_test.hpp" // IWYU pragma: keep

#include <span>
#include <string>
#include <vector>

#include "../support/renamed_stepper.hpp"


namespace odetest {
namespace {

using ode::SolverPolicy;
using ode::UtilPolicy;

template<typename T, typename SD>
struct Setup {
    T t0 = 0;
    state_t<T, SD> q0 = Oscillator::at<SD>(T(0));
    real_state_t<T, SD> tol = uniform_like<SD, T>(q0, T(std::is_same_v<T, float> ? 1e-6 : 1e-10));
};


TEMPLATE_TEST_CASE("getSolver builds a stepper for every policy", "[factories]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            const auto check_policy = [&]<SolverPolicy SP, typename VirtualBase, typename CloneType>(){
                auto solver = make_solver<Stepper, SD, SP>(Oscillator{}, T(0), Oscillator::at<SD>(T(0)));
                using Solver = decltype(solver);
                STATIC_REQUIRE(std::is_same_v<Solver, typename Stepper::template type<T, SD, Oscillator, SP>>);
                STATIC_REQUIRE(Solver::Policy == SP);
                STATIC_REQUIRE(std::is_same_v<typename Solver::virtual_base, VirtualBase>);
                STATIC_REQUIRE(std::is_same_v<typename Solver::clone_type, std::conditional_t<std::is_void_v<CloneType>, Solver, CloneType>>);
                REQUIRE(solver.advance_until(T(1)));
                CHECK_THAT(max_error<SD>(current_state(solver), Oscillator::at<SD>(T(1))), IsAtMost(settings<Stepper, T>().error));
            };
            check_policy.template operator()<SolverPolicy::Static, void, void>();
            check_policy.template operator()<SolverPolicy::RichStatic, void, void>();
            check_policy.template operator()<SolverPolicy::Virtual, ode::OdeSolver<T, SD>, ode::OdeSolver<T, SD>>();
            check_policy.template operator()<SolverPolicy::RichVirtual, ode::OdeRichSolver<T, SD>, ode::OdeSolver<T, SD>>();
        }
    });
    STATIC_REQUIRE(ode::is_rich<SolverPolicy::RichStatic>);
    STATIC_REQUIRE_FALSE(ode::is_rich<SolverPolicy::Virtual>);
}

TEMPLATE_TEST_CASE("The boxing factories put a solver behind its interface", "[factories][virtual]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<StateKinds<2>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            const Setup<T, SD> s;
            ode::BoxedSolver<T, SD, UtilPolicy::Virtual> plain = ode::make_vsolver<SD>(tags::DOP853{}, Oscillator{}, s.t0, s.q0, s.tol, s.tol);
            ode::BoxedSolver<T, SD, UtilPolicy::RichVirtual> rich = ode::make_rich_vsolver<SD>(tags::BDF{}, Oscillator{}, s.t0, s.q0, s.tol, s.tol,
                T(0), T(0), T(0), 1, ode::make_event_list<T, state_t<T, SD>>(ode::make_periodic_event<T, state_t<T, SD>>("tick", T(0.5))));
            ode::BoxedSolver<T, SD, UtilPolicy::Virtual> generic = ode::make_solver<UtilPolicy::Virtual, SD>(tags::RK23{}, Oscillator{}, s.t0, s.q0, s.tol, s.tol);

            CHECK(plain->get_method() == "DOP853");
            CHECK(rich->get_method() == "BDF");
            CHECK(generic->get_method() == "RK23");
            CHECK(rich->get_event_col().size() == 1);
            REQUIRE(rich->do_advance_to_event());
            CHECK(rich->get_time() == T(0.5));
            REQUIRE(plain->do_advance_until(T(1)));
            state_t<T, SD> q = s.q0;
            plain->get_current_vector(q);
            CHECK_THAT(max_error<SD>(q, Oscillator::at<SD>(T(1))), IsAtMost(settings<tags::DOP853, T>().error));
        }
    });
}


TEMPLATE_TEST_CASE("A catalog builds the stepper an id names", "[factories][catalog]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const Setup<T, SD> s;
    using Ids = ode::Steppers;
    for (const Ids::Id id : {Ids::id<tags::Euler>, Ids::id<tags::RK4>, Ids::id<tags::RK23>, Ids::id<tags::RK45>, Ids::id<tags::DOP853>, Ids::id<tags::BDF>}){
        const std::string name(Ids::getName(id));
        CAPTURE(name);
        // The fixed-step ones need a step
        const T step = (name == "Euler" || name == "RK4") ? T(1e-3) : T(0);
        ode::BoxedSolver<T, SD, UtilPolicy::RichVirtual> rich = Ids::make<UtilPolicy::RichVirtual, SD>(id, Oscillator{}, s.t0, s.q0, s.tol, s.tol, T(0), T(0), step);
        ode::BoxedSolver<T, SD, UtilPolicy::Virtual> plain = Ids::make<UtilPolicy::Virtual, SD>(id, Oscillator{}, s.t0, s.q0, s.tol, s.tol, T(0), T(0), step);
        CHECK(rich->get_method() == name);
        CHECK(plain->get_method() == name);
        REQUIRE(rich->do_advance_until(T(0.5)));
        CHECK_THAT(max_error<SD>(rich->get_new_vector(), Oscillator::at<SD>(rich->get_new_time())), IsAtMost(1e-2));
    }
}

TEMPLATE_TEST_CASE("A stepper declared outside the library is chosen like the built-in ones", "[factories][catalog]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const Setup<T, SD> s;
    STATIC_REQUIRE(tags::RenamedRK45::name == "RenamedRK45");

    auto solver = ode::getSolver<SD, SolverPolicy::Static>(tags::RenamedRK45{}, Oscillator{}, s.t0, s.q0, s.tol, s.tol);
    CHECK(solver.method() == "RenamedRK45");
    REQUIRE(solver.advance_until(T(1)));
    CHECK_THAT(max_error<SD>(current_state(solver), Oscillator::at<SD>(T(1))), IsAtMost(settings<tags::RK45, T>().error));

    using Mine = ode::Catalog<tags::RK45, tags::RenamedRK45>;
    auto boxed = Mine::make<UtilPolicy::RichVirtual, SD>(Mine::getID("RenamedRK45"), Oscillator{}, s.t0, s.q0, s.tol, s.tol);
    CHECK(boxed->get_method() == "RenamedRK45");
    CHECK(boxed->clone()->get_method() == "RenamedRK45");
    CHECK(ode::make_rich_vsolver<SD>(tags::RenamedRK45{}, Oscillator{}, s.t0, s.q0, s.tol, s.tol)->get_method() == "RenamedRK45");
}


TEMPLATE_TEST_CASE("The virtual interface reaches every method of a solver", "[factories][virtual]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<StateKinds<2>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            using State = state_t<T, SD>;
            const Setup<T, SD> s;
            const auto events = [](){
                return ode::make_event_list<T, State>(ode::make_periodic_event<T, State>("tick", T(0.25)),
                    ode::make_precise_event<T, State>("x = 0", [](const T& /*t*/, const State& q) -> T { return q[0]; }, T(0)));
            };
            // The boxed solver against the same one held by its concrete type, moved in lockstep
            auto concrete = ode::getSolver<SD, SolverPolicy::RichVirtual>(tags::RK45{}, Oscillator{}, s.t0, s.q0, s.tol, s.tol, T(0), T(0), T(0), 1, events());
            ode::BoxedSolver<T, SD, UtilPolicy::RichVirtual> boxed = ode::make_rich_vsolver<SD>(tags::RK45{}, Oscillator{}, s.t0, s.q0, s.tol, s.tol, T(0), T(0), T(0), 1, events());
            ode::OdeRichSolver<T, SD>& solver = *boxed;

            const auto same_as_concrete = [&](){
                CHECK(solver.get_time() == concrete.t());
                CHECK(solver.get_new_time() == concrete.t_new());
                CHECK(solver.get_old_time() == concrete.t_old());
                CHECK(same_state<SD>(solver.get_new_vector(), concrete.vector_new()));
                CHECK(same_state<SD>(solver.get_old_vector(), concrete.vector_old()));
                State q = s.q0;
                solver.get_current_vector(q);
                CHECK(same_state<SD>(q, current_state(concrete)));
                CHECK(solver.get_stepsize() == concrete.stepsize());
                CHECK(solver.get_step_count() == concrete.step_count());
                CHECK(solver.get_status() == concrete.status());
                CHECK(solver.get_at_event() == concrete.at_event());
            };

            SECTION("accessors"){
                CHECK(solver.get_ics_time() == s.t0);
                CHECK(same_state<SD>(solver.get_ics_vector(), s.q0));
                CHECK(solver.get_direction() == 1);
                CHECK(same_state<SD>(solver.get_rtol(), s.tol));
                CHECK(same_state<SD>(solver.get_atol(), s.tol));
                CHECK(solver.get_min_step() == T(0));
                CHECK(solver.get_max_step() == ode::inf<T>());
                CHECK(solver.get_is_running());
                CHECK(solver.get_message() == "Running");
                CHECK(solver.get_validate_ics(s.t0, s.q0));
                CHECK(solver.get_has_valid_ics());
                CHECK(solver.get_method() == "RK45");
                CHECK(solver.get_auto_step() == concrete.auto_step());
                CHECK(solver.get_auto_step(T(1), s.q0) == concrete.auto_step(T(1), s.q0));
                State out = s.q0;
                solver.get_rhs(out, T(0), make_state<SD, T>({2, 3}));
                CHECK(same_state<SD>(out, make_state<SD, T>({3, -2})));
                auto J = make_jacmat<SD, T>(s.q0), J_concrete = J;
                solver.get_jac(J, T(0), s.q0);
                concrete.jac(J_concrete, T(0), s.q0);
                CHECK(max_matrix_error(J, J_concrete, 2).value == 0);
                solver.get_jac(J, T(0), s.q0, uniform_like<SD, T>(s.q0, T(1e-3)));
                concrete.jac(J_concrete, T(0), s.q0, uniform_like<SD, T>(s.q0, T(1e-3)));
                CHECK(max_matrix_error(J, J_concrete, 2).value == 0);
                std::ostringstream text;
                solver.show_state(text, 5);
                CHECK_THAT(text.str(), Catch::Matchers::ContainsSubstring("Running"));
            }
            SECTION("advancing"){
                REQUIRE(solver.do_advance());
                REQUIRE(concrete.advance());
                same_as_concrete();

                REQUIRE(solver.do_advance_by(T(0.1)));
                REQUIRE(concrete.advance_by(T(0.1)));
                same_as_concrete();

                REQUIRE(solver.do_advance_until(T(0.4)));
                REQUIRE(concrete.advance_until(T(0.4)));
                same_as_concrete();

                size_t calls = 0;
                const ode::observer_t<T, State> count = [&calls](const T&, const State&, const T*){ ++calls; return true; };
                REQUIRE(solver.do_advance_until(T(0.6), count));
                REQUIRE(concrete.advance_until(T(0.6)));
                CHECK(calls > 0);
                same_as_concrete();
                REQUIRE(solver.do_advance_until(T(0.7), nullptr));
                REQUIRE(concrete.advance_until(T(0.7)));
                same_as_concrete();

                std::vector<T> landed;
                const std::vector<T> checkpoints = {T(0.8), T(0.9)};
                const ode::observer_t<T, State> record = [&landed](const T& t, const State&, const T* t_ptr){
                    if (t_ptr){
                        landed.push_back(t);
                    }
                    return true;
                };
                REQUIRE(solver.do_advance_until(T(1), record, std::span<const T>(checkpoints)));
                REQUIRE(solver.do_advance_until(T(1.1), nullptr, std::span<const T>(checkpoints.data(), 0)));
                REQUIRE(concrete.advance_until(T(1.1)));
                CHECK(landed == checkpoints);
                same_as_concrete();
            }
            SECTION("events"){
                CHECK(solver.get_event_col().size() == 2);
                CHECK(solver.get_event_idx("x = 0") == 1);
                REQUIRE(solver.do_advance_to_event());
                CHECK(solver.get_at_event(0));
                CHECK_FALSE(solver.get_at_canon_event());
                CHECK(solver.get_current_event().event->name() == "tick");
                REQUIRE(solver.do_advance_to_event(std::vector<size_t>{1}));
                CHECK(solver.get_current_event().idx == 1);
                REQUIRE(solver.do_advance_to_event(std::vector<std::string>{"tick"}));
                CHECK(solver.get_at_event(0));
                REQUIRE(solver.do_advance_to_event(T(10), std::vector<size_t>{0}));
                CHECK_FALSE(solver.do_advance_to_event(T(solver.get_time() + T(0.1)), std::vector<std::string>{"x = 0"}));
            }
            SECTION("restarting and cloning"){
                REQUIRE(solver.do_advance_until(T(0.5)));
                std::unique_ptr<ode::OdeSolver<T, SD>> copy = solver.clone();
                CHECK(copy->get_time() == T(0.5));
                CHECK(copy->get_method() == "RK45");

                solver.do_reset();
                CHECK(solver.get_time() == s.t0);
                CHECK(solver.get_step_count() == 0);
                CHECK(copy->get_time() == T(0.5)); // a solver of its own
                REQUIRE(solver.do_set_ics(T(1), Oscillator::at<SD>(T(1))));
                CHECK(solver.get_time() == T(1));
                const auto step = solver.get_state_interpolator(1, 1);
                CHECK(step->t_start() == step->t_end());
            }
        }
    });
}

} // namespace
} // namespace odetest
