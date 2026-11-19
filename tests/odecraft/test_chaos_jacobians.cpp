// The derivatives of a variational system: the Jacobian of the main system, the Jacobian of the
// whole system, [[J, 0], [dJ/dq*dq, J]], and the variational right-hand side J*dq, by automatic
// differentiation, from the system's own Jacobian, and by first and second order differences.
// Also the block structure of the variational state descriptor, VarSD.

#include <cmath>

#include "../support/odecraft_test.hpp" // IWYU pragma: keep

namespace odetest {
namespace {

using ode::SolverPolicy;
using ode::JacPolicy;
using ode::contracts::VarSD;

/// @brief The nonlinear system with dual numbers of every order for a Rhs of its own state descriptor
/// only: the second-order duals path, rather than the directional one a fully generic Rhs takes.
template<typename T, typename SD>
struct NonlinearSecondOrderOnly {
    void Rhs(state_t<T, SD>& out, const T& t, const state_t<T, SD>& q) const { Nonlinear::Rhs(out, t, q); }

    template<int Order>
    void Rhs(ode::dual_state_t<T, SD, Order>& out, const T& t, const ode::seed_state_t<T, SD, Order>& q) const { Nonlinear::Rhs(out, t, q); }
};

template<typename SD, typename T>
var_state_t<SD, T> sample_point(){
    return make_var_state<SD, T>({0.7, -0.4}, {0.6, 0.8});
}

template<typename T>
T sample_time(){
    return T(0.3);
}

/// @brief A variational solver of `ode` at the sample point. Only its derivatives are looked at.
template<typename SD, typename T, typename OdeType>
auto variational_solver_of(OdeType ode, double atol = 1e-8){
    const auto q = sample_point<SD, T>();
    const auto tol = var_state_t<SD, T>{.main = uniform_like<SD, T>(q.main, T(atol)), .variation = uniform_like<SD, T>(q.main, T(atol))};
    return ode::chaos::getVariationalSolver<SolverPolicy::Static, SD>(tags::RK45{}, std::move(ode), sample_time<T>(), q, T(1), tol, tol);
}

/// @brief The exact derivatives at the sample point.
template<typename SD, typename T>
struct Exact {
    jacmat_t<T, SD> J, B;
    state_t<T, SD> f, Jdq;

    Exact() : J(make_jacmat<SD, T>(sample_point<SD, T>().main)), B(J), f(sample_point<SD, T>().main), Jdq(f) {
        const auto q = sample_point<SD, T>();
        Nonlinear::jacobian<SD, T>(J, sample_time<T>(), q.main);
        Nonlinear::jacobian_derivative<SD, T>(B, q.main, q.variation);
        Nonlinear::Rhs(f, sample_time<T>(), q.main);
        SD::JD::linalg::dot(Jdq, J, q.variation);
    }
};

template<typename T>
double first_difference_accuracy(){
    return 50 * std::sqrt(epsilon<T>());
}

template<typename T>
double second_difference_accuracy(){
    return 500 * std::sqrt(epsilon<T>());
}

/// @brief The Jacobian of the main system, the full Jacobian and the variational Rhs of a solver.
template<typename SD, typename T, typename Solver>
auto derivatives_of(const Solver& solver){
    const auto q = sample_point<SD, T>();
    auto J_main = make_jacmat<SD, T>(q.main);
    solver.JacMain(J_main, sample_time<T>(), q.main);
    auto J_full = make_jacmat<VarSD<SD>, T>(q);
    solver.jac(J_full, sample_time<T>(), q);
    auto rhs = q;
    solver.rhs(rhs, sample_time<T>(), q);
    return std::tuple{J_main, J_full, rhs};
}

template<typename Body>
void for_each_kind(Body&& body){
    for_each_type<StateKinds<2>>([&]<typename SD>(){
        DYNAMIC_SECTION(kind_name<SD>){
            body.template operator()<SD>();
        }
    });
}


TEMPLATE_TEST_CASE("Autodiff gives every derivative of a variational system exactly", "[chaos][jacobian][autodiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    const double tol = 50 * epsilon<T>();
    for_each_kind([&]<typename SD>(){
        const Exact<SD, T> exact;
        const auto check = [&](const auto& solver){
            const auto [J_main, J_full, rhs] = derivatives_of<SD, T>(solver);
            CHECK_THAT(max_matrix_error(J_main, exact.J, 2), IsAtMost(tol));
            CHECK_THAT(max_matrix_error(J_full[0], exact.J, 2), IsAtMost(tol));
            CHECK_THAT(max_matrix_error(J_full[1], exact.B, 2), IsAtMost(tol));
            CHECK_THAT(max_error<SD>(rhs.main, exact.f), IsAtMost(tol));
            CHECK_THAT(max_error<SD>(rhs.variation, exact.Jdq), IsAtMost(tol));
        };
        SECTION("through directional duals"){
            STATIC_REQUIRE(ode::chaos::VariationalOdeSys<T, SD, Nonlinear>::JP_MAIN == JacPolicy::Autodiff);
            STATIC_REQUIRE(ode::chaos::detail::MainRhsSupportsJacDuals<Nonlinear, T, SD>);
            check(variational_solver_of<SD, T>(Nonlinear{}));
        }
        SECTION("through second order duals"){
            using Ode = NonlinearSecondOrderOnly<T, SD>;
            STATIC_REQUIRE_FALSE(ode::chaos::detail::MainRhsSupportsJacDuals<Ode, T, SD>);
            STATIC_REQUIRE_FALSE(ode::chaos::detail::MainRhsSupportsDirDuals<Ode, T, SD>);
            STATIC_REQUIRE(ode::chaos::detail::MainRhsSupportsDuals<2, Ode, T, SD>);
            check(variational_solver_of<SD, T>(Ode{}));
        }
    });
}

TEMPLATE_TEST_CASE("A variational system uses the Jacobian its main system provides", "[chaos][jacobian]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_kind([]<typename SD>(){
        const Exact<SD, T> exact;
        const auto check = [&](const auto& solver){
            const auto [J_main, J_full, rhs] = derivatives_of<SD, T>(solver);
            CHECK_THAT(max_matrix_error(J_main, exact.J, 2), IsAtMost(own_jacobian_rounding<T>()));
            CHECK_THAT(max_matrix_error(J_full[0], exact.J, 2), IsAtMost(own_jacobian_rounding<T>()));
            CHECK_THAT(max_error<SD>(rhs.variation, exact.Jdq), IsAtMost(10 * epsilon<T>()));
            // Only the derivative of J along dq is left to differences, of second order
            CHECK_THAT(max_matrix_error(J_full[1], exact.B, 2), IsAtMost(second_difference_accuracy<T>()));
        };
        SECTION("an exact one"){
            STATIC_REQUIRE(ode::chaos::VariationalOdeSys<T, SD, NonlinearWithJac<T, SD>>::JP_MAIN == JacPolicy::Exact);
            check(variational_solver_of<SD, T>(NonlinearWithJac<T, SD>{}));
        }
        SECTION("a nullable one that is set"){
            check(variational_solver_of<SD, T>(nullable_nonlinear<T, SD>(true)));
        }
    });
}

TEMPLATE_TEST_CASE("Variational Jacobians by explicit steps converge at second order in them", "[chaos][jacobian][findiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    constexpr bool single = std::is_same_v<T, float>;
    const double dq = single ? 0.2 : 1e-2;
    const double shrink = single ? 2 : 10;
    for_each_kind([&]<typename SD>(){
        // The explicit step means differences of the whole system, whatever the main one provides
        const auto solver = variational_solver_of<SD, T>(Nonlinear{});
        const Exact<SD, T> exact;
        const auto q = sample_point<SD, T>();
        const auto errors_at = [&](double step){
            const var_state_t<SD, T> dq_full{.main = uniform_like<SD, T>(q.main, T(step)), .variation = uniform_like<SD, T>(q.main, T(step))};
            auto J = make_jacmat<VarSD<SD>, T>(q);
            solver.jac(J, sample_time<T>(), q, dq_full);
            return std::pair{max_matrix_error(J[0], exact.J, 2).value, max_matrix_error(J[1], exact.B, 2).value};
        };
        const auto [main_coarse, block_coarse] = errors_at(dq);
        const auto [main_fine, block_fine] = errors_at(dq / shrink);
        CAPTURE(main_coarse, main_fine, block_coarse, block_fine);
        CHECK(main_coarse < 0.1 * dq);
        CHECK(std::abs(std::log(main_coarse / main_fine) / std::log(shrink) - 2) < 0.25);
        CHECK(std::abs(std::log(block_coarse / block_fine) / std::log(shrink) - 2) < 0.25);
    });
}

TEMPLATE_TEST_CASE("The main system of a variational solver is reached on its own", "[chaos][jacobian]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const Exact<SD, T> exact;
    const auto q = sample_point<SD, T>();

    const auto solver = variational_solver_of<SD, T>(NonlinearWithJac<T, SD>{});
    auto f = q.main;
    solver.RhsMain(f, sample_time<T>(), q.main);
    CHECK(same_state<SD>(f, exact.f));

    // and through the virtual interface of a chaotic solver
    const auto tol = var_state_t<SD, T>{.main = uniform_like<SD, T>(q.main, T(1e-8)), .variation = uniform_like<SD, T>(q.main, T(1e-8))};
    auto boxed = ode::chaos::make_variational_solver<ode::UtilPolicy::Virtual, SD>(tags::RK45{}, NonlinearWithJac<T, SD>{}, sample_time<T>(), q, T(1), tol, tol);
    auto J = make_jacmat<SD, T>(q.main);
    boxed->get_jac_main(J, sample_time<T>(), q.main);
    CHECK_THAT(max_matrix_error(J, exact.J, 2), IsAtMost(own_jacobian_rounding<T>()));
    boxed->get_rhs_main(f, sample_time<T>(), q.main);
    CHECK(same_state<SD>(f, exact.f));
}

TEMPLATE_TEST_CASE("Differences of a main system need a non-zero absolute tolerance", "[chaos][jacobian][findiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    // The step of a difference is scaled by atol where the state is small, so a zero one is refused
    CHECK_THROWS_AS((variational_solver_of<SD, T>(PlainNonlinear<T, SD>{}, 0.0)), std::runtime_error);
    CHECK_THROWS_AS((variational_solver_of<SD, T>(nullable_nonlinear<T, SD>(false), 0.0)), std::runtime_error);
    // but what needs no differences does not care
    CHECK_NOTHROW(variational_solver_of<SD, T>(Nonlinear{}, 0.0));
    CHECK_NOTHROW(variational_solver_of<SD, T>(NonlinearWithJac<T, SD>{}, 0.0));
}


// Differences with steps of their own choosing, from the tolerances

TEMPLATE_TEST_CASE("Without a Jacobian or autodiff a variational system is differenced", "[chaos][jacobian][findiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_kind([]<typename SD>(){
        const Exact<SD, T> exact;
        const auto check = [&](const auto& solver){
            const auto [J_main, J_full, rhs] = derivatives_of<SD, T>(solver);
            CHECK_THAT(max_matrix_error(J_main, exact.J, 2), IsAtMost(first_difference_accuracy<T>()));
            CHECK_THAT(max_matrix_error(J_full[0], exact.J, 2), IsAtMost(first_difference_accuracy<T>()));
            CHECK_THAT(max_matrix_error(J_full[1], exact.B, 2), IsAtMost(second_difference_accuracy<T>()));
            CHECK_THAT(max_error<SD>(rhs.variation, exact.Jdq), IsAtMost(first_difference_accuracy<T>()));
            CHECK(same_state<SD>(rhs.main, exact.f));
        };
        SECTION("when it provides nothing"){
            STATIC_REQUIRE(ode::chaos::VariationalOdeSys<T, SD, PlainNonlinear<T, SD>>::JP_MAIN == JacPolicy::Approx);
            check(variational_solver_of<SD, T>(PlainNonlinear<T, SD>{}));
        }
        SECTION("when its nullable Jacobian is null"){
            check(variational_solver_of<SD, T>(nullable_nonlinear<T, SD>(false)));
        }
    });
}

TEMPLATE_TEST_CASE("The differences of a variational system follow its tolerances", "[chaos][jacobian][findiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const Exact<SD, T> exact;
    const auto [J_small, full_small, rhs_small] = derivatives_of<SD, T>(variational_solver_of<SD, T>(PlainNonlinear<T, SD>{}, 1e-8));
    const auto [J_large, full_large, rhs_large] = derivatives_of<SD, T>(variational_solver_of<SD, T>(PlainNonlinear<T, SD>{}, 10));
    // An atol above |q| sets the steps, and with them the results, which stay accurate
    CHECK(max_matrix_error(J_small, J_large, 2).value > 0);
    CHECK(max_matrix_error(full_small[1], full_large[1], 2).value > 0);
    CHECK_THAT(max_matrix_error(J_large, exact.J, 2), IsAtMost(10 * first_difference_accuracy<T>()));
    CHECK_THAT(max_matrix_error(full_large[1], exact.B, 2), IsAtMost(10 * second_difference_accuracy<T>()));
}


TEMPLATE_TEST_CASE("The variational state descriptor works on both halves of its states", "[chaos][contracts]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    for_each_kind([]<typename SD>(){
        using V = VarSD<SD>;
        auto a = make_var_state<SD, T>({1, 2}, {3, 4});
        const auto b = make_var_state<SD, T>({10, 20}, {30, 40});

        V::for_each([](T& a_i, const T& b_i){ a_i += b_i; }, a, b);
        CHECK(same_state<V>(a, make_var_state<SD, T>({11, 22}, {33, 44})));
        CHECK(V::all_true([](const T& x){ return x > 10; }, a));
        CHECK_FALSE(V::all_true([](const T& x){ return x > 30; }, a));
        CHECK(size_of<V>(a) == 4);
        CHECK(to_double(V::template norm_sqr<T>(T(-2))) == 4);

        var_state_t<SD, T> resized{};
        V::resize_like(resized, a);
        CHECK(size_of<V>(resized) == 4);

        std::ostringstream out;
        V::console(out, a);
        CHECK(out.str() == "11 22\n33 44");
    });
}

TEMPLATE_TEST_CASE("The variational Jacobian descriptor solves with its block lower triangular matrices", "[chaos][contracts][linalg]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    const double tol = 100 * epsilon<T>();
    for_each_kind([&]<typename SD>(){
        using JD = typename VarSD<SD>::JD;
        // [[A, 0], [C, A]] with A = [[2, 1], [1, 3]] and C = [[1, 0], [2, 1]]
        const auto like = make_var_state<SD, T>({0, 0}, {0, 0});
        auto M = make_jacmat<VarSD<SD>, T>(like);
        M[0](0, 0) = 2; M[0](0, 1) = 1; M[0](1, 0) = 1; M[0](1, 1) = 3;
        M[1](0, 0) = 1; M[1](0, 1) = 0; M[1](1, 0) = 2; M[1](1, 1) = 1;
        const auto x = make_var_state<SD, T>({1, -1}, {2, 0.5});

        // M x = (A x.main, C x.main + A x.variation) = ((1, -2), (1 + 4.5, 1 + 3.5))
        auto b = like;
        JD::linalg::dot(b, M, x);
        CHECK_THAT(max_error<VarSD<SD>>(b, make_var_state<SD, T>({1, -2}, {5.5, 4.5})), IsAtMost(tol));

        SECTION("lu_solve inverts dot"){
            typename JD::linalg::pivots_type piv{};
            auto LU = M;
            REQUIRE(JD::linalg::lu_factor(LU, piv));
            auto solved = like;
            JD::linalg::lu_solve(solved, LU, piv, b);
            CHECK_THAT(max_error<VarSD<SD>>(solved, x), IsAtMost(tol));
            JD::linalg::lu_solve(b, LU, piv, b);   // in place
            CHECK_THAT(max_error<VarSD<SD>>(b, x), IsAtMost(tol));
        }
        SECTION("identity_minus puts the identity on the diagonal blocks only"){
            auto I_cM = M;
            JD::linalg::identity_minus(I_cM, T(2), M);
            CHECK_THAT(abs_error(I_cM[0](0, 0), T(-3)), IsAtMost(tol));
            CHECK_THAT(abs_error(I_cM[0](0, 1), T(-2)), IsAtMost(tol));
            CHECK_THAT(abs_error(I_cM[1](1, 0), T(-4)), IsAtMost(tol));
            CHECK_THAT(abs_error(I_cM[1](0, 0), T(-2)), IsAtMost(tol));
        }
        SECTION("scale and all_true act on both blocks"){
            auto S = M;
            JD::scale(S, T(-1), M);
            CHECK(JD::all_true([](const T& v){ return v <= 0; }, S));
            CHECK_FALSE(JD::all_true([](const T& v){ return v < 0; }, S));
        }
        SECTION("a singular diagonal block makes it singular"){
            auto S = M;
            S[0](1, 0) = 2; S[0](1, 1) = 1;   // A = [[2, 1], [2, 1]]
            typename JD::linalg::pivots_type piv{};
            CHECK_FALSE(JD::linalg::lu_factor(S, piv));
        }
    });
}

} // namespace
} // namespace odetest
