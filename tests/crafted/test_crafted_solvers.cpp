// Solvers of the compiled interface: built at runtime by make_rich_vsolver for every stepper
// and scalar type the library ships, and driven through the OdeSolver and OdeRichSolver
// interfaces, on states of any rank.

#include "crafted_test.hpp"

#include <cmath>
#include <span>
#include <sstream>
#include <type_traits>

namespace odetest {
namespace {

using namespace ode::crafted;

/// @brief A solver of the oscillator from t = 0, with the settings the tests give its stepper.
template<typename T>
BoxedRichSolver<T> oscillator_solver(const std::string& stepper, int dir = 1, bool with_jacobian = true, EventList<T> events = {}){
    const Steppers::Id id = Steppers::getID(stepper);
    const Settings<T> s = settings_of<T>(id);
    const State<T> q0 = oscillator::at(T(0));
    return make_rich_vsolver<T>(id, oscillator::system<T>(with_jacobian), T(0), q0, filled_like(q0, s.rtol), filled_like(q0, s.atol),
        T(0), T(0), s.stepsize, dir, std::move(events));
}


TEMPLATE_TEST_CASE("make_rich_vsolver builds every compiled stepper", "[crafted][solvers]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    STATIC_REQUIRE(std::is_same_v<BoxedRichSolver<T>, ode::BoxedSolver<T, SD, ode::UtilPolicy::RichVirtual>>);
    for (const std::string& name : stepper_names){
        DYNAMIC_SECTION(name){
            const double bound = settings_of<T>(Steppers::getID(name)).error;
            for (int dir : {1, -1}){
                CAPTURE(dir);
                BoxedRichSolver<T> solver = oscillator_solver<T>(name, dir);
                CHECK(solver->get_method() == name);
                CHECK(solver->get_direction() == dir);
                REQUIRE(solver->do_advance_until(T(5 * dir)));
                CHECK(solver->get_time() == T(5 * dir));
                CHECK_THAT(max_error(current_vector(*solver), oscillator::at(T(5 * dir))), IsAtMost(bound));
            }
        }
    }
}

TEMPLATE_TEST_CASE("Without a Jacobian an implicit compiled stepper differences the system", "[crafted][solvers][findiff]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    BoxedRichSolver<T> solver = oscillator_solver<T>("BDF", 1, false);
    REQUIRE(solver->do_advance_until(T(5)));
    CHECK_THAT(max_error(current_vector(*solver), oscillator::at(T(5))), IsAtMost(settings<tags::BDF, T>().error));

    // And so does the solver asked for the Jacobian: a central difference with steps of
    // sqrt(epsilon) is off by about sqrt(epsilon)
    const double differencing = 50 * std::sqrt(epsilon<T>());
    JacMat<T> J(2, 2);
    solver->get_jac(J, T(0), oscillator::at(T(0)));
    CHECK_THAT(abs_error(T(J(0, 1)), T(1)), IsAtMost(differencing));
    CHECK_THAT(abs_error(T(J(1, 0)), T(-1)), IsAtMost(differencing));
}

TEMPLATE_TEST_CASE("A compiled solver is driven through its interfaces", "[crafted][solvers][virtual]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    const double bound = settings<tags::RK45, T>().error;
    BoxedRichSolver<T> boxed = oscillator_solver<T>("RK45");
    OdeRichSolver<T>& rich = *boxed;
    OdeSolver<T>& solver = rich; // a rich solver is a solver too

    SECTION("accessors"){
        CHECK(solver.get_ics_time() == T(0));
        CHECK_THAT(max_error(solver.get_ics_vector(), oscillator::at(T(0))), IsAtMost(0));
        CHECK(solver.get_is_running());
        CHECK(solver.get_message() == "Running");
        CHECK(solver.get_rtol().size() == 2);
        State<T> out = oscillator::at(T(0));
        solver.get_rhs(out, T(0), State<T>{T(2), T(3)});
        CHECK(out[0] == T(3));
        CHECK(out[1] == T(-2));
        JacMat<T> J(2, 2);
        solver.get_jac(J, T(0), out);
        CHECK(J(0, 1) == T(1)); // the system's own Jacobian
        std::ostringstream text;
        solver.show_state(text, 5);
        CHECK_THAT(text.str(), Catch::Matchers::ContainsSubstring("Running"));
    }
    SECTION("advancing"){
        REQUIRE(solver.do_advance());
        REQUIRE(solver.do_advance_by(T(0.5)));
        size_t calls = 0;
        const std::vector<T> checkpoints = {T(1.5), T(2)};
        std::vector<T> landed;
        const observer_t<T> record = [&](const T& t, const State<T>& q, const T* t_ptr){
            ++calls;
            if (t_ptr){
                landed.push_back(t);
            }
            return q.size() == 2;
        };
        REQUIRE(solver.do_advance_until(T(3), record, std::span<const T>(checkpoints)));
        CHECK(calls > 0);
        CHECK(landed == checkpoints);
        CHECK_THAT(max_error(current_vector(solver), oscillator::at(T(3))), IsAtMost(bound));
    }
    SECTION("dense output"){
        const BoxedInterp<T> dense = solver.do_interpolate_until(T(4));
        CHECK(dense->t_start() == T(0));
        CHECK(dense->t_end() == T(4));
        State<T> q = oscillator::at(T(0));
        for (const T& t : {T(0.3), T(1.7), T(3.9)}){
            dense->call(q, t);
            CHECK_THAT(max_error(q, oscillator::at(t)), IsAtMost(bound));
        }
        const auto step = solver.get_state_interpolator(0, 0);
        CHECK(step->t_end() == solver.get_new_time());
    }
    SECTION("restarting and cloning"){
        REQUIRE(solver.do_advance_until(T(1)));
        std::unique_ptr<OdeSolver<T>> copy = solver.clone();
        CHECK(copy->get_time() == T(1));
        CHECK(copy->get_method() == "RK45");

        solver.do_reset();
        CHECK(solver.get_time() == T(0));
        CHECK(solver.get_step_count() == 0);
        CHECK(copy->get_time() == T(1)); // a solver of its own
        REQUIRE(solver.do_set_ics(T(2), oscillator::at(T(2))));
        REQUIRE(solver.do_advance_until(T(3)));
        CHECK_THAT(max_error(current_vector(solver), oscillator::at(T(3))), IsAtMost(bound));

        // And the clone carries on from where it was made
        REQUIRE(copy->do_advance_until(T(3)));
        CHECK_THAT(max_error(current_vector(*copy), oscillator::at(T(3))), IsAtMost(bound));
    }
}

TEMPLATE_TEST_CASE("A state of rank two is integrated as its flattened components", "[crafted][solvers][shape]", ODECRAFT_CRAFTED_SCALARS){
    using T = TestType;
    // Two oscillators side by side: column j holds (x_j, v_j), the second a quarter turn ahead
    const auto at = [](const T& t){
        State<T> q(2, 2);
        q(0, 0) = cos(t);
        q(1, 0) = -sin(t);
        q(0, 1) = sin(t);
        q(1, 1) = cos(t);
        return q;
    };
    size_t jacobian_rows = 0;
    const ode_t<T> system{
        .Rhs = [](State<T>& out, T /*t*/, const State<T>& q){
            for (size_t j = 0; j < 2; j++){
                out(0, j) = q(1, j);
                out(1, j) = -q(0, j);
            }
        },
        .Jac = [&jacobian_rows](JacMat<T>& J, T /*t*/, const State<T>& /*q*/){
            // Over the flattened state: x_j is component j, and v_j component 2 + j
            jacobian_rows = J.shape(0);
            for (size_t i = 0; i < 4; i++){
                for (size_t j = 0; j < 4; j++){
                    J(i, j) = 0;
                }
            }
            for (size_t j = 0; j < 2; j++){
                J(j, 2 + j) = 1;
                J(2 + j, j) = -1;
            }
        },
    };
    for (const std::string name : {"RK45", "BDF"}){
        DYNAMIC_SECTION(name){
            const Steppers::Id id = Steppers::getID(name);
            const Settings<T> s = settings_of<T>(id);
            const State<T> q0 = at(T(0));
            BoxedRichSolver<T> solver = make_rich_vsolver<T>(id, system, T(0), q0, filled_like(q0, s.rtol), filled_like(q0, s.atol));
            REQUIRE(solver->do_advance_until(T(3)));
            const State<T> q = current_vector(*solver);
            CHECK(q.ndim() == 2);
            CHECK(q.shape(1) == 2);
            CHECK_THAT(max_error(q, at(T(3))), IsAtMost(s.error));
            if (name == "BDF"){
                CHECK(jacobian_rows == 4);
            }
        }
    }
}

} // namespace
} // namespace odetest
