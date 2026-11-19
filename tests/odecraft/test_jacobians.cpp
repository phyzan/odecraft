// Every way a solver computes the Jacobian of its system: by automatic differentiation, from the
// system's own Jacobian, from a Jacobian that may be null, and by central differences, with steps
// chosen from the tolerances or given explicitly.

#include "odecraft_test.hpp"

#include <cmath>

namespace odetest {
namespace {

using ode::SolverPolicy;
using ode::JacPolicy;

template<typename SD, typename T>
state_t<T, SD> sample_point(){
    return make_state<SD, T>({0.7, -0.4});
}

template<typename T>
T sample_time(){
    return T(0.3);
}

/// @brief A solver of `ode` at the sample point, whose Jacobian is what the tests look at.
template<typename SD, typename T, typename OdeType>
auto solver_of(OdeType ode, double atol = 1e-8){
    const auto q = sample_point<SD, T>();
    return ode::getSolver<SD, SolverPolicy::Static>(tags::RK45{}, std::move(ode), sample_time<T>(), q,
        uniform_like<SD, T>(q, T(1e-8)), uniform_like<SD, T>(q, T(atol)));
}

template<typename SD, typename T>
jacmat_t<T, SD> exact_jacobian(){
    auto J = make_jacmat<SD, T>(sample_point<SD, T>());
    Nonlinear::jacobian<SD, T>(J, sample_time<T>(), sample_point<SD, T>());
    return J;
}

/// @brief The accuracy of a central difference with steps of sqrt(epsilon): roundoff over the step.
template<typename T>
double difference_accuracy(){
    return 50 * std::sqrt(epsilon<T>());
}


TEMPLATE_TEST_CASE("An autodiff Jacobian is exact to rounding", "[jacobian][autodiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<StateKinds<2>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            const auto solver = solver_of<SD, T>(Nonlinear{});
            STATIC_REQUIRE(decltype(solver)::JP == JacPolicy::Autodiff);

            auto J = make_jacmat<SD, T>(sample_point<SD, T>());
            solver.jac(J, sample_time<T>(), sample_point<SD, T>());
            CHECK_THAT(max_matrix_error(J, exact_jacobian<SD, T>(), 2), IsAtMost(20 * epsilon<T>()));
        }
    });
}

TEMPLATE_TEST_CASE("An explicit Jacobian is the one the system provides", "[jacobian]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<StateKinds<2>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            const auto solver = solver_of<SD, T>(NonlinearWithJac<T, SD>{});
            STATIC_REQUIRE(decltype(solver)::JP == JacPolicy::Exact);

            auto J = make_jacmat<SD, T>(sample_point<SD, T>());
            solver.jac(J, sample_time<T>(), sample_point<SD, T>());
            CHECK(max_matrix_error(J, exact_jacobian<SD, T>(), 2).value == 0);
        }
    });
}

TEMPLATE_TEST_CASE("A nullable Jacobian is used when it is set", "[jacobian][nullable]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<StateKinds<2>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            const auto solver = solver_of<SD, T>(nullable_nonlinear<T, SD>(true));
            STATIC_REQUIRE(decltype(solver)::JP == JacPolicy::Nullable);

            auto J = make_jacmat<SD, T>(sample_point<SD, T>());
            solver.jac(J, sample_time<T>(), sample_point<SD, T>());
            CHECK(max_matrix_error(J, exact_jacobian<SD, T>(), 2).value == 0);
        }
    });
}

TEMPLATE_TEST_CASE("An explicit step gives central differences of second order in it", "[jacobian][findiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    // Steps where truncation, O(dq^2), dominates the roundoff of T
    constexpr bool single = std::is_same_v<T, float>;
    const double dq = single ? 0.2 : 1e-2;
    const double shrink = single ? 2 : 10;

    for_each_type<StateKinds<2>>([&]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            // The explicit step always means differences, even for a system autodiff would apply to
            const auto solver = solver_of<SD, T>(Nonlinear{});
            const auto q = sample_point<SD, T>();
            const auto error_at = [&](double step){
                auto J = make_jacmat<SD, T>(q);
                solver.jac(J, sample_time<T>(), q, uniform_like<SD, T>(q, T(step)));

                auto J_approx = J;
                solver.jac_approx(J_approx, sample_time<T>(), q, uniform_like<SD, T>(q, T(step)));
                CHECK(max_matrix_error(J, J_approx, 2).value == 0);
                return max_matrix_error(J, exact_jacobian<SD, T>(), 2).value;
            };

            const double coarse = error_at(dq);
            const double fine = error_at(dq / shrink);
            CAPTURE(coarse, fine);
            CHECK(coarse < 0.1 * dq);
            CHECK(std::abs(std::log(coarse / fine) / std::log(shrink) - 2) < 0.2);
        }
    });
}

TEMPLATE_TEST_CASE("The virtual interface computes the same Jacobians", "[jacobian][virtual]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const auto q = sample_point<SD, T>();
    const auto tol = uniform_like<SD, T>(q, T(1e-8));
    ode::BoxedSolver<T, SD, ode::UtilPolicy::Virtual> solver = ode::make_vsolver<SD>(tags::RK45{}, Nonlinear{}, sample_time<T>(), q, tol, tol);

    auto J = make_jacmat<SD, T>(q);
    solver->get_jac(J, sample_time<T>(), q);
    CHECK_THAT(max_matrix_error(J, exact_jacobian<SD, T>(), 2), IsAtMost(20 * epsilon<T>()));

    auto J_explicit = J, J_reference = J;
    const auto dq = uniform_like<SD, T>(q, T(1e-2));
    solver->get_jac(J_explicit, sample_time<T>(), q, dq);
    solver_of<SD, T>(Nonlinear{}).jac(J_reference, sample_time<T>(), q, dq);
    CHECK(max_matrix_error(J_explicit, J_reference, 2).value == 0);
}

TEMPLATE_TEST_CASE("The free finite-difference functions difference any function", "[jacobian][findiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const auto q = sample_point<SD, T>();
    const auto f = [](state_t<T, SD>& out, const T& t, const state_t<T, SD>& x){ Nonlinear::Rhs(out, t, x); };
    auto x = q, y1 = q, y2 = q;
    auto J = make_jacmat<SD, T>(q);

    ode::jac_approx_explicit<T, SD>(f, J, x, y1, y2, sample_time<T>(), q, uniform_like<SD, T>(q, T(1e-3)));
    CHECK_THAT(max_matrix_error(J, exact_jacobian<SD, T>(), 2), IsAtMost(std::is_same_v<T, float> ? 1e-2 : 1e-6));
    CHECK(same_state<SD>(x, q));   // the point it differences around is restored
}


// Differences with steps of their own choosing, from the tolerances

TEMPLATE_TEST_CASE("A null nullable Jacobian falls back to differences", "[jacobian][nullable][findiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<StateKinds<2>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            const auto solver = solver_of<SD, T>(nullable_nonlinear<T, SD>(false));
            auto J = make_jacmat<SD, T>(sample_point<SD, T>());
            solver.jac(J, sample_time<T>(), sample_point<SD, T>());
            CHECK_THAT(max_matrix_error(J, exact_jacobian<SD, T>(), 2), IsAtMost(difference_accuracy<T>()));

            auto J_approx = J;
            solver.jac_approx(J_approx, sample_time<T>(), sample_point<SD, T>());
            CHECK(max_matrix_error(J, J_approx, 2).value == 0);
        }
    });
}

TEMPLATE_TEST_CASE("Without a Jacobian or autodiff the system is differenced with steps that follow atol", "[jacobian][findiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_type<StateKinds<2>>([]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            const auto q = sample_point<SD, T>();
            const auto jacobian_with_atol = [&](double atol){
                const auto solver = solver_of<SD, T>(PlainNonlinear<T, SD>{}, atol);
                STATIC_REQUIRE(decltype(solver)::JP == JacPolicy::Approx);
                auto J = make_jacmat<SD, T>(q);
                solver.jac(J, sample_time<T>(), q);
                auto J_approx = J;
                solver.jac_approx(J_approx, sample_time<T>(), q);
                CHECK(max_matrix_error(J, J_approx, 2).value == 0);
                return J;
            };

            // The step is sqrt(epsilon) * max(atol, |q|): below |q| the tolerance does not matter
            const auto J_small_atol = jacobian_with_atol(1e-8);
            CHECK_THAT(max_matrix_error(J_small_atol, exact_jacobian<SD, T>(), 2), IsAtMost(difference_accuracy<T>()));
            CHECK(max_matrix_error(J_small_atol, jacobian_with_atol(1e-6), 2).value == 0);

            // Above it, the step and with it the result follow atol
            const auto J_large_atol = jacobian_with_atol(10);
            CHECK(max_matrix_error(J_small_atol, J_large_atol, 2).value > 0);
            CHECK_THAT(max_matrix_error(J_large_atol, exact_jacobian<SD, T>(), 2), IsAtMost(10 * difference_accuracy<T>()));
        }
    });
}

TEMPLATE_TEST_CASE("The free autostep difference function follows the tolerances it is given", "[jacobian][findiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const auto q = sample_point<SD, T>();
    const auto f = [](state_t<T, SD>& out, const T& t, const state_t<T, SD>& x){ Nonlinear::Rhs(out, t, x); };
    auto x = q, y1 = q, y2 = q;
    auto J = make_jacmat<SD, T>(q);

    ode::jac_approx_autostep<T, SD>(f, J, x, y1, y2, sample_time<T>(), q, uniform_like<SD, T>(q, T(1e-8)));
    CHECK_THAT(max_matrix_error(J, exact_jacobian<SD, T>(), 2), IsAtMost(difference_accuracy<T>()));
    CHECK(same_state<SD>(x, q));
}

} // namespace
} // namespace odetest
