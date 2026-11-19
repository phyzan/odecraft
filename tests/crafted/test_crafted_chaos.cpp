// Variational solvers of the compiled interface: the largest Lyapunov exponent of a linear
// saddle, known exactly, with every compiled stepper, forward and backward, through
// make_variational_solver, the chaotic solver interface and the variational driver.

#include <cmath>
#include <memory>
#include <type_traits>

#include "../support/crafted_test.hpp"

namespace odetest {
namespace {

using namespace ode::crafted;

/// @brief The saddle from q = (1, 1) with the deviation d.
template<typename T>
VarState<T> saddle_start(const T& d0, const T& d1){
    return {.main = {T(1), T(1)}, .variation = {d0, d1}};
}

/// @brief A tolerance of a variational state: the same for the state and for its deviation.
template<typename T>
VarState<T> var_tolerance(const T& value){
    return {.main = {value, value}, .variation = {value, value}};
}

/// @brief How close an exponent over |t| <= 3 comes to the exact one with a stepper's settings.
template<typename T>
double lyapunov_accuracy(const Settings<T>& s){
    return 20 * s.error;
}

/// @brief A variational solver of the saddle with the settings the tests give a stepper.
template<typename T>
BoxedChaoticSolver<T> saddle_solver(const std::string& stepper, int dir, const VarState<T>& q0, bool with_jacobian = true){
    const Steppers::Id id = Steppers::getID(stepper);
    const Settings<T> s = settings_of<T>(id);
    return make_variational_solver<T>(id, saddle::system<T>(with_jacobian), T(0), q0, T(0.5), var_tolerance(s.rtol), var_tolerance(s.atol),
        T(0), T(0), s.stepsize, dir);
}


TEMPLATE_TEST_CASE("make_variational_solver gives the exact Lyapunov exponent of a saddle", "[crafted][chaos][lyapunov]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    for (const std::string& name : stepper_names){
        DYNAMIC_SECTION(name){
            const double bound = lyapunov_accuracy(settings_of<T>(Steppers::getID(name)));
            for (int dir : {1, -1}){
                CAPTURE(dir);
                BoxedChaoticSolver<T> solver = saddle_solver<T>(name, dir, saddle_start(T(0.6), T(0.8)));
                CHECK(solver->get_method() == name);
                REQUIRE(solver->do_advance_until(T(3 * dir)));
                CHECK(solver->get_elapsed_time() == T(3 * dir));
                CHECK_THAT(abs_error(solver->get_lyapunov_exponent(), saddle::lyapunov_exponent(T(3 * dir), T(0.6), T(0.8))), IsAtMost(bound));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Without a Jacobian a compiled variational solver differences the system", "[crafted][chaos][lyapunov][findiff]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    // A difference with steps of sqrt(epsilon) is off by about sqrt(epsilon), and so is the exponent
    const double differencing = 10 * std::sqrt(epsilon<T>());
    for (const std::string name : {"RK45", "BDF"}){
        DYNAMIC_SECTION(name){
            const double bound = lyapunov_accuracy(settings_of<T>(Steppers::getID(name))) + differencing;
            BoxedChaoticSolver<T> solver = saddle_solver<T>(name, 1, saddle_start(T(0.6), T(0.8)), false);
            REQUIRE(solver->do_advance_until(T(3)));
            CHECK_THAT(abs_error(solver->get_lyapunov_exponent(), saddle::lyapunov_exponent(T(3), T(0.6), T(0.8))), IsAtMost(bound));
        }
    }
}

TEMPLATE_TEST_CASE("A chaotic solver reports how its deviation grows", "[crafted][chaos][virtual]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    STATIC_REQUIRE(std::is_base_of_v<ode::OdeRichSolver<T, VarSD>, ChaoticSolver<T>>);
    STATIC_REQUIRE(std::is_same_v<BoxedChaoticSolver<T>, pbox::Box<ChaoticSolver<T>>>);
    const double bound = lyapunov_accuracy(settings<tags::RK45, T>());

    // Along the unstable direction the deviation grows by e per unit of time
    BoxedChaoticSolver<T> solver = saddle_solver<T>("RK45", 1, saddle_start(T(1), T(0)));
    CHECK(solver->get_period() == T(0.5));
    CHECK(solver->get_lyapunov_exponent() == T(0));
    REQUIRE(solver->do_advance_until(T(0.75)));
    CHECK(solver->get_elapsed_time() == T(0.75));
    CHECK_THAT(abs_error(solver->get_log_ksi(), T(0.75)), IsAtMost(bound));
    CHECK_THAT(abs_error(solver->get_stretching_number(), T(0.25)), IsAtMost(bound)); // since t = 0.5
    CHECK_THAT(abs_error(solver->get_kick(), T(1)), IsAtMost(bound));
    CHECK_THAT(abs_error(solver->get_lyapunov_exponent(), T(1)), IsAtMost(bound));

    // It is a rich solver of the augmented state too, and a clone of one carries the estimate along
    const VarState<T> q = current_vector(*solver);
    CHECK_THAT(abs_error(T(q.main[0]), T(exp(T(0.75)))), IsAtMost(bound));
    auto copy = solver->clone();
    REQUIRE(solver->do_advance_until(T(2)));
    REQUIRE(copy->do_advance_until(T(2)));
    const auto* chaotic_copy = dynamic_cast<const ChaoticSolver<T>*>(copy.get());
    REQUIRE(chaotic_copy != nullptr);
    CHECK(chaotic_copy->get_lyapunov_exponent() == solver->get_lyapunov_exponent());
}

TEMPLATE_TEST_CASE("The variational driver records the Lyapunov estimate and its events", "[crafted][chaos][ode-driver]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    const Settings<T> s = settings<tags::DOP853, T>();
    const double bound = lyapunov_accuracy(s);
    for (int dir : {1, -1}){
        CAPTURE(dir);
        // y = exp(-t) falls through 1/2 at t = ln 2 going forward, and only grows going backward
        const objfun_t<T, VarState<T>> y_is_half = [](const T& /*t*/, const VarState<T>& q) -> T { return q.main[1] - T(0.5); };
        VariationalODE<T> driver(saddle::system<T>(), T(0), saddle_start(T(0.6), T(0.8)), T(0.5), var_tolerance(s.rtol), var_tolerance(s.atol),
            T(0), T(0), T(0), dir,
            make_variational_event_list<T>(make_variational_periodic_event<T>("tick", T(1)), make_variational_precise_event<T>("y = 1/2", y_is_half, T(0), 0)),
            Steppers::id<tags::DOP853>);
        CHECK(driver.solver()->get_method() == "DOP853");

        OdeResult<T, VarSD> result;
        REQUIRE(driver.integrate_until(&result, T(2 * dir)));
        CHECK(driver.lyap_values().size() == driver.n_points() - 1);
        CHECK(driver.kick_values().size() == driver.n_points() - 1);
        CHECK(driver.renorm_times().size() == driver.n_points() - 1);
        CHECK(driver.lyap_values().back() == driver.solver()->get_lyapunov_exponent());
        CHECK_THAT(abs_error(driver.lyap_values().back(), saddle::lyapunov_exponent(T(2 * dir), T(0.6), T(0.8))), IsAtMost(bound));
        CHECK(result.event_data().data("tick").size() == 2);
        const OrbitData<T, VarSD>& halves = result.event_data().data("y = 1/2");
        REQUIRE(halves.size() == (dir == 1 ? 1 : 0));
        if (dir == 1){
            CHECK_THAT(abs_error(halves.t[0], T(log(T(2)))), IsAtMost(s.error));
        }
    }
}

TEMPLATE_TEST_CASE("A variational driver takes RK45 unless told otherwise and copies whole", "[crafted][chaos][ode-driver][clone]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    const VarState<T> tol = var_tolerance(T(1e-6));
    VariationalODE<T> driver(saddle::system<T>(), T(0), saddle_start(T(0.6), T(0.8)), T(0.5), tol, tol);
    CHECK(driver.solver()->get_method() == "RK45");
    REQUIRE(driver.integrate_until(nullptr, T(1)));

    // Held by the base it shares with every driver of its state, it is copied without slicing
    pbox::owner<OdeDriver<T, VarSD>> original(std::make_unique<VariationalODE<T>>(driver));
    pbox::owner<OdeDriver<T, VarSD>> copy = original;
    const auto* variational = dynamic_cast<const ode::chaos::VariationalODE<T, SD>*>(copy.get_raw_pointer());
    REQUIRE(variational != nullptr);
    CHECK(variational->lyap_values() == driver.lyap_values());
    REQUIRE(copy->integrate_until(nullptr, T(2)));
    CHECK(variational->lyap_values().size() > driver.lyap_values().size());
}

} // namespace
} // namespace odetest
