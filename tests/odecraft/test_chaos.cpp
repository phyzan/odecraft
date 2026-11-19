// Variational solvers: the largest Lyapunov exponent of a linear saddle, which is known exactly,
// with every stepper, forward and backward, whichever way the main system is differentiated,
// through every integration routine, the chaotic solver interface and the variational driver.

#include "odecraft_test.hpp"

#include <vector>

namespace odetest {
namespace {

using ode::SolverPolicy;
using ode::UtilPolicy;
using ode::contracts::VarSD;
namespace chaos = ode::chaos;

/// @brief A tolerance of a variational state: the same for the state and for its deviation.
template<typename SD, typename T>
var_state_t<SD, T> var_tolerance(const T& value){
    const auto like = make_state<SD, T>({0, 0});
    return {.main = uniform_like<SD, T>(like, value), .variation = uniform_like<SD, T>(like, value)};
}

/// @brief A variational solver of the saddle from q = (1, 1), deviation d0, renormalized every `period`.
template<typename Stepper, typename SD, SolverPolicy SP = SolverPolicy::Static, typename T, typename OdeType>
auto saddle_solver(OdeType ode, int dir, const state_t<T, SD>& d0, const T& period, ode::EventList<T, var_state_t<SD, T>> events = {}){
    const Settings<T> s = settings<Stepper, T>();
    const auto q0 = var_state_t<SD, T>{.main = make_state<SD, T>({1, 1}), .variation = d0};
    return chaos::getVariationalSolver<SP, SD>(Stepper{}, std::move(ode), T(0), q0, period, var_tolerance<SD>(s.rtol), var_tolerance<SD>(s.atol),
        T(0), T(0), s.stepsize, dir, std::move(events));
}

/// @brief The saddle's driver from q = (1, 1) and deviation (0.6, 0.8), renormalized every 0.5, with an event every unit of time.
template<typename SD, typename T>
chaos::VariationalODE<T, SD> saddle_driver(int dir){
    using VarState = var_state_t<SD, T>;
    const Settings<T> s = settings<tags::RK45, T>();
    return chaos::VariationalODE<T, SD>(tags::RK45{}, Saddle{}, T(0), make_var_state<SD, T>({1, 1}, {0.6, 0.8}), T(0.5),
        var_tolerance<SD>(s.rtol), var_tolerance<SD>(s.atol), T(0), T(0), T(0), dir,
        ode::make_event_list<T, VarState>(ode::make_periodic_event<T, VarState>("tick", T(1))));
}

/// @brief How close a Lyapunov exponent over |t| <= 3 comes to the exact one, with a stepper's settings.
template<typename Stepper, typename T>
double lyapunov_accuracy(){
    // The deviation grows like e^|t|: what remains of it is the stepper's relative error over the span
    return 20 * settings<Stepper, T>().error;
}

/// @brief What a differenced Jacobian adds to that. A difference with steps of sqrt(epsilon) is
/// off by about sqrt(epsilon) relative to the Jacobian, and so is the exponent it drives.
template<typename T>
double difference_accuracy(){
    return 10 * std::sqrt(epsilon<T>());
}

/// @brief The Euclidean norm of a state of SD, computed in T.
template<typename SD, typename T, typename State>
T norm_of(const State& state){
    T sum = 0;
    SD::for_each([&sum](const auto& x){ sum += x * x; }, state);
    return sqrt(sum);
}

/// @brief Integrates for 3 time units in direction `dir`, and compares the exponent with the exact one.
template<typename Stepper, typename SD, typename T, typename OdeType>
void check_lyapunov_exponent(int dir, double bound){
    CAPTURE(dir);
    auto solver = saddle_solver<Stepper, SD>(OdeType{}, dir, make_state<SD, T>({0.6, 0.8}), T(0.5));
    const T span = T(3 * dir);
    REQUIRE(solver.advance_until(span));
    CHECK(solver.elapsed_time() == span);
    CHECK_THAT(abs_error(solver.lyapunov_exponent(), Saddle::lyapunov_exponent(span, T(0.6), T(0.8))), IsAtMost(bound));
}

template<typename Body>
void for_each_stepper(Body&& body){
    for_each_type<AllSteppers>([&]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            body.template operator()<Stepper>();
        }
    });
}


TEMPLATE_TEST_CASE("The Lyapunov exponent of a saddle is exact with autodiff", "[chaos][lyapunov][autodiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_stepper([]<typename Stepper>(){
        const double bound = lyapunov_accuracy<Stepper, T>();
        check_lyapunov_exponent<Stepper, StaticSize<2>, T, Saddle>(+1, bound);
        check_lyapunov_exponent<Stepper, StaticSize<2>, T, Saddle>(-1, bound);
    });
    SECTION("on runtime-sized states"){
        check_lyapunov_exponent<tags::RK45, RuntimeSize, T, Saddle>(+1, lyapunov_accuracy<tags::RK45, T>());
        check_lyapunov_exponent<tags::BDF, RuntimeSize, T, Saddle>(-1, lyapunov_accuracy<tags::BDF, T>());
    }
}

TEMPLATE_TEST_CASE("The Lyapunov exponent of a saddle is exact with the system's Jacobian", "[chaos][lyapunov]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    for_each_stepper([]<typename Stepper>(){
        const double bound = lyapunov_accuracy<Stepper, T>();
        check_lyapunov_exponent<Stepper, SD, T, SaddleWithJac<T, SD>>(+1, bound);
        check_lyapunov_exponent<Stepper, SD, T, SaddleWithJac<T, SD>>(-1, bound);
    });
}

TEMPLATE_TEST_CASE("The Lyapunov exponent of a saddle is exact by finite differences", "[chaos][lyapunov][findiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    for_each_stepper([]<typename Stepper>(){
        const double bound = lyapunov_accuracy<Stepper, T>() + difference_accuracy<T>();
        check_lyapunov_exponent<Stepper, SD, T, PlainSaddle<T, SD>>(+1, bound);
        check_lyapunov_exponent<Stepper, SD, T, PlainSaddle<T, SD>>(-1, bound);
    });
}


TEMPLATE_TEST_CASE("A deviation along the unstable direction stretches by e per unit of time", "[chaos][lyapunov]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const double tol = lyapunov_accuracy<tags::RK45, T>();
    for (int dir : {1, -1}){
        CAPTURE(dir);
        // Forward the deviation grows along (1, 0), backward along (0, 1): |d| = e^|t| either way
        const auto d0 = dir == 1 ? make_state<SD, T>({1, 0}) : make_state<SD, T>({0, 1});
        auto solver = saddle_solver<tags::RK45, SD>(Saddle{}, dir, d0, T(0.5));
        CHECK(solver.period() == T(0.5));
        CHECK(solver.lyapunov_exponent() == T(0));   // nothing to estimate from yet

        SECTION("between renormalizations"){
            REQUIRE(solver.advance_until(T(0.75 * dir)));
            CHECK(solver.elapsed_time() == T(0.75 * dir));
            CHECK_THAT(abs_error(solver.log_ksi(), T(0.75)), IsAtMost(tol));
            CHECK_THAT(abs_error(solver.stretching_number(), T(0.25)), IsAtMost(tol));   // since |t| = 0.5
            // Rates are per unit of t, which runs the other way backward
            CHECK_THAT(abs_error(solver.kick(), T(dir)), IsAtMost(tol));
            CHECK_THAT(abs_error(solver.lyapunov_exponent(), T(dir)), IsAtMost(tol));
            // The deviation was brought back to unit norm at |t| = 0.5
            CHECK_THAT(abs_error(norm_of<SD, T>(current_state(solver).variation), T(exp(T(0.25)))), IsAtMost(tol));
        }
        SECTION("on a renormalization"){
            REQUIRE(solver.advance_until(T(dir)));
            CHECK(solver.stretching_number() == T(0));
            CHECK_THAT(abs_error(solver.log_ksi(), T(1)), IsAtMost(tol));
            CHECK_THAT(abs_error(solver.kick(), T(dir)), IsAtMost(tol));
            CHECK_THAT(abs_error(solver.lyapunov_exponent(), T(dir)), IsAtMost(tol));
        }
    }
}

TEMPLATE_TEST_CASE("Every integration routine gives the same Lyapunov exponent", "[chaos][lyapunov][advance]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    using VarState = var_state_t<SD, T>;
    const double tol = lyapunov_accuracy<tags::RK45, T>();
    const T exact = Saddle::lyapunov_exponent(T(3), T(0.6), T(0.8));
    auto solver = saddle_solver<tags::RK45, SD, SolverPolicy::RichStatic>(Saddle{}, 1, make_state<SD, T>({0.6, 0.8}), T(0.5),
        ode::make_event_list<T, VarState>(ode::make_periodic_event<T, VarState>("tick", T(0.7))));

    SECTION("advance"){
        while (solver.t() < T(3)){
            REQUIRE(solver.advance());
        }
        CHECK_THAT(abs_error(solver.lyapunov_exponent(), Saddle::lyapunov_exponent(solver.t(), T(0.6), T(0.8))), IsAtMost(tol));
    }
    SECTION("advance_by"){
        for (int i = 0; i < 6; i++){
            REQUIRE(solver.advance_by(T(0.5)));
        }
        CHECK_THAT(abs_error(solver.lyapunov_exponent(), exact), IsAtMost(tol));
    }
    SECTION("advance_until, with an observer and checkpoints"){
        size_t calls = 0;
        REQUIRE(solver.advance_until(T(3), [&calls](const T&, const auto&, const T*){ ++calls; return true; }, std::vector<T>{T(1.1), T(2.2)}));
        CHECK(calls > 0);
        CHECK_THAT(abs_error(solver.lyapunov_exponent(), exact), IsAtMost(tol));
    }
    SECTION("interpolate_until"){
        const auto dense = solver.interpolate_until(T(3));
        CHECK_THAT(abs_error(solver.lyapunov_exponent(), exact), IsAtMost(tol));
        auto q = solver.vector_new();
        dense->call(q, T(1.3));
        CHECK_THAT(abs_error(q.main[0], T(exp(T(1.3)))), IsAtMost(tol));
        CHECK_THAT(abs_error(q.main[1], T(exp(T(-1.3)))), IsAtMost(tol));
    }
    SECTION("advance_to_event"){
        for (int i = 1; i <= 4; i++){
            REQUIRE(solver.advance_to_event());
            CHECK(solver.t() == T(i) * T(0.7));
        }
        CHECK_THAT(abs_error(solver.lyapunov_exponent(), Saddle::lyapunov_exponent(solver.t(), T(0.6), T(0.8))), IsAtMost(tol));
    }
}

TEMPLATE_TEST_CASE("A variational solver restarts clones and refuses what it cannot do", "[chaos][lifecycle]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    using VarState = var_state_t<SD, T>;
    const double tol = lyapunov_accuracy<tags::RK45, T>();
    const double roundoff = 4 * epsilon<T>();
    auto solver = saddle_solver<tags::RK45, SD>(Saddle{}, 1, make_state<SD, T>({3, 4}), T(0.5));
    // The deviation is normalized when the solver is built
    CHECK_THAT(max_error<SD>(solver.vector_ics().variation, make_state<SD, T>({T(T(3) / T(5)), T(T(4) / T(5))})), IsAtMost(roundoff));
    REQUIRE(solver.advance_until(T(2)));

    SECTION("Reset starts the estimate over"){
        solver.Reset();
        CHECK(solver.t() == T(0));
        CHECK(solver.lyapunov_exponent() == T(0));
        REQUIRE(solver.advance_until(T(3)));
        CHECK_THAT(abs_error(solver.lyapunov_exponent(), Saddle::lyapunov_exponent(T(3), T(0.6), T(0.8))), IsAtMost(tol));
    }
    SECTION("SetICs normalizes the new deviation, and refuses a zero one"){
        REQUIRE(solver.SetICs(T(1), make_var_state<SD, T>({1, 1}, {0, 5}), T(0)));
        CHECK_THAT(max_error<SD>(solver.vector_ics().variation, make_state<SD, T>({0, 1})), IsAtMost(roundoff));
        REQUIRE(solver.advance_until(T(2)));
        // Along the stable direction, forward: the exponent is -1
        CHECK_THAT(abs_error(solver.lyapunov_exponent(), T(-1)), IsAtMost(tol));
        CHECK_FALSE(solver.SetICs(T(0), make_var_state<SD, T>({1, 1}, {0, 0}), T(0)));
    }
    SECTION("a clone carries the estimate along"){
        auto copy = solver.clone();
        REQUIRE(copy->advance_until(T(3)));
        REQUIRE(solver.advance_until(T(3)));
        CHECK(copy->lyapunov_exponent() == solver.lyapunov_exponent());
    }
    SECTION("a period must be positive, and an event may not move the state"){
        CHECK_THROWS_AS((saddle_solver<tags::RK45, SD>(Saddle{}, 1, make_state<SD, T>({1, 0}), T(0))), std::runtime_error);
        auto masked = ode::make_event_list<T, VarState>(ode::make_periodic_event<T, VarState>("kick", T(1),
            [](VarState& out, const T&, const VarState& q){ out = q; }));
        CHECK_THROWS_AS((saddle_solver<tags::RK45, SD, SolverPolicy::RichStatic>(Saddle{}, 1, make_state<SD, T>({1, 0}), T(0.5), std::move(masked))), std::runtime_error);
    }
}

TEMPLATE_TEST_CASE("A variational solver is reached through the chaotic solver interface", "[chaos][virtual]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const auto q0 = make_var_state<SD, T>({1, 1}, {1, 0});
    const Settings<T> rk45 = settings<tags::RK45, T>(), dop853 = settings<tags::DOP853, T>();

    using Plain = chaos::VariationalSolver<tags::RK45, T, SD, Saddle, SolverPolicy::Virtual>;
    using Rich = chaos::VariationalSolver<tags::RK45, T, SD, Saddle, SolverPolicy::RichVirtual>;
    STATIC_REQUIRE(std::is_same_v<typename Plain::virtual_base, chaos::ChaoticSolver<T, SD, UtilPolicy::Virtual>>);
    STATIC_REQUIRE(std::is_same_v<typename Rich::virtual_base, chaos::ChaoticSolver<T, SD, UtilPolicy::RichVirtual>>);
    STATIC_REQUIRE(std::is_same_v<typename Plain::clone_type, ode::OdeSolver<T, VarSD<SD>>>);
    STATIC_REQUIRE(std::is_void_v<typename chaos::VariationalSolver<tags::RK45, T, SD, Saddle, SolverPolicy::Static>::virtual_base>);

    pbox::Box<chaos::ChaoticSolver<T, SD, UtilPolicy::RichVirtual>> solver = chaos::make_variational_solver<UtilPolicy::RichVirtual, SD>(
        tags::RK45{}, Saddle{}, T(0), q0, T(0.5), var_tolerance<SD>(rk45.rtol), var_tolerance<SD>(rk45.atol));
    CHECK(solver->get_method() == "RK45");
    CHECK(solver->get_period() == T(0.5));
    REQUIRE(solver->do_advance_until(T(0.75)));
    const double bound = lyapunov_accuracy<tags::RK45, T>();
    CHECK(solver->get_elapsed_time() == T(0.75));
    CHECK_THAT(abs_error(solver->get_lyapunov_exponent(), T(1)), IsAtMost(bound));
    CHECK_THAT(abs_error(solver->get_log_ksi(), T(0.75)), IsAtMost(bound));
    CHECK_THAT(abs_error(solver->get_stretching_number(), T(0.25)), IsAtMost(bound));
    CHECK_THAT(abs_error(solver->get_kick(), T(1)), IsAtMost(bound));

    auto plain = chaos::make_variational_solver<UtilPolicy::Virtual, SD>(tags::DOP853{}, Saddle{}, T(0), q0, T(0.5),
        var_tolerance<SD>(dop853.rtol), var_tolerance<SD>(dop853.atol));
    REQUIRE(plain->do_advance_until(T(2)));
    CHECK_THAT(abs_error(plain->get_lyapunov_exponent(), T(1)), IsAtMost(lyapunov_accuracy<tags::DOP853, T>()));
    CHECK(plain->clone()->get_time() == T(2));
}


TEMPLATE_TEST_CASE("The variational driver records the Lyapunov estimate along its integrations", "[chaos][ode-driver]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const double tol = lyapunov_accuracy<tags::RK45, T>();
    const auto exact = [](const T& t) -> T { return Saddle::lyapunov_exponent(t, T(0.6), T(0.8)); };

    for (int dir : {1, -1}){
        CAPTURE(dir);
        auto driver = saddle_driver<SD, T>(dir);
        CHECK(driver.solver()->get_method() == "RK45");

        ode::OdeResult<T, VarSD<SD>> result;
        REQUIRE(driver.integrate_until(&result, T(2 * dir)));
        // An estimate for every point recorded after the first
        CHECK(driver.lyap_values().size() == driver.n_points() - 1);
        CHECK(driver.kick_values().size() == driver.n_points() - 1);
        CHECK(driver.renorm_times().size() == driver.n_points() - 1);
        CHECK(driver.lyap_values().back() == driver.solver()->get_lyapunov_exponent());
        CHECK_THAT(abs_error(driver.lyap_values().back(), exact(T(2 * dir))), IsAtMost(tol));
        CHECK(result.event_data().data("tick").size() == 2);

        // The next integration goes on from there
        REQUIRE(driver.integrate_until(&result, T(3 * dir)));
        CHECK(driver.solver()->get_time() == T(3 * dir));
        CHECK_THAT(abs_error(driver.lyap_values().back(), exact(T(3 * dir))), IsAtMost(tol));

        auto copy = driver.clone();
        CHECK(copy->n_points() == driver.n_points());
        ode::OdeSolution<T, VarSD<SD>> solution;
        REQUIRE(copy->rich_integrate_until(solution, T(4 * dir)));
        auto q = make_var_state<SD, T>({0, 0}, {0, 0});
        solution(q, T(3.5 * dir));
        CHECK_THAT(rel_error(q.main[0], T(exp(T(3.5 * dir)))), IsAtMost(tol));

        driver.reset();
        CHECK(driver.n_points() == 1);
        CHECK(driver.lyap_values().empty());
        CHECK(driver.solver()->get_time() == T(0));
        REQUIRE(driver.integrate_until(nullptr, T(dir)));
        driver.clear();
        CHECK(driver.lyap_values().empty());
        CHECK(driver.renorm_times().empty());
        CHECK(driver.solver()->get_time() == T(dir));
    }
}

TEMPLATE_TEST_CASE("The variational driver integrates by an interval in its direction", "[chaos][ode-driver]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const double tol = lyapunov_accuracy<tags::RK45, T>();
    for (int dir : {1, -1}){
        CAPTURE(dir);
        auto driver = saddle_driver<SD, T>(dir);
        ode::OdeResult<T, VarSD<SD>> result;
        REQUIRE(driver.integrate(&result, T(1.5)));
        CHECK(driver.solver()->get_time() == T(1.5 * dir));
        CHECK_THAT(abs_error(driver.lyap_values().back(), Saddle::lyapunov_exponent(T(1.5 * dir), T(0.6), T(0.8))), IsAtMost(tol));

        ode::OdeSolution<T, VarSD<SD>> solution;
        REQUIRE(driver.rich_integrate(solution, T(0.5)));
        CHECK(driver.solver()->get_time() == T(2 * dir));
    }
}

} // namespace
} // namespace odetest
