// Accuracy of every stepper: against exact solutions, at its order of convergence, at its
// tolerance, to the full precision of extended scalar types, and on stiff problems.

#include "../support/odecraft_test.hpp" // IWYU pragma: keep

#include <cmath>

namespace odetest {
namespace {

using ode::SolverPolicy;

/// @brief Integrates the oscillator from t0 over 10 time units in direction `dir`, checking the
/// solution against the exact one at a few times along the way, each landed on exactly.
template<typename Stepper, typename SD, typename T>
void check_follows_exact_solution(int dir){
    CAPTURE(dir);
    const Settings<T> s = settings<Stepper, T>();
    const T t0 = T(0.5);
    auto solver = make_solver<Stepper, SD, SolverPolicy::Static>(Oscillator{}, t0, Oscillator::at<SD>(t0), dir);

    for (double elapsed : {0.25, 1.0, 3.7, 10.0}){
        const T t = t0 + T(elapsed * dir);
        CAPTURE(elapsed);
        REQUIRE(solver.advance_until(t));
        CHECK(solver.t() == t);
        CHECK_THAT(max_error<SD>(current_state(solver), Oscillator::at<SD>(t)), IsAtMost(s.error));
    }
}

TEMPLATE_TEST_CASE("Every stepper follows the exact solution forward and backward", "[steppers]", ODECRAFT_TEST_SCALARS){
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            for_each_type<StateKinds<2>>([]<typename SD>(){
                DYNAMIC_SECTION(kind_name<SD>){
                    check_follows_exact_solution<Stepper, SD, TestType>(+1);
                    check_follows_exact_solution<Stepper, SD, TestType>(-1);
                }
            });
        }
    });
}


/// @brief The error at t_end of a fixed-step integration of the oscillator from t = 0 with step h.
template<typename Stepper, typename T>
double fixed_step_error(const T& h, const T& t_end){
    using SD = StaticSize<2>;
    auto solver = ode::getSolver<SD, SolverPolicy::Static>(Stepper{}, Oscillator{}, T(0), Oscillator::at<SD>(T(0)),
        make_state<SD, T>({0, 0}), make_state<SD, T>({0, 0}), T(0), T(0), h);
    REQUIRE(solver.advance_until(t_end));
    return max_error<SD>(current_state(solver), Oscillator::at<SD>(t_end)).value;
}

/// @brief log2 of how much halving the step reduces the error: the order of the method, as long
/// as h is small enough for the leading error term to dominate, and large enough for roundoff not to.
template<typename Stepper, typename T>
double observed_order(const T& h, const T& t_end){
    return std::log2(fixed_step_error<Stepper>(h, t_end) / fixed_step_error<Stepper>(T(h/2), t_end));
}

TEMPLATE_TEST_CASE("Fixed-step steppers converge at their order", "[steppers][convergence]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    constexpr bool single = std::is_same_v<T, float>;

    SECTION("Euler is first order"){
        CHECK(std::abs(observed_order<tags::Euler>(T(1e-2), T(1)) - 1) < 0.02);
    }
    SECTION("RK4 is fourth order"){
        // Over a longer span the truncation error outgrows single precision roundoff, though
        // only at steps still short of the asymptotic regime
        const T h = single ? T(0.2) : T(0.05);
        CHECK(std::abs(observed_order<tags::RK4>(h, T(10)) - 4) < (single ? 0.3 : 0.05));
    }
}


/// @brief The largest error of an adaptive integration of the oscillator over [0, 5] at tolerance `tol`.
template<typename Stepper, typename T>
Magnitude adaptive_error(double tol){
    using SD = StaticSize<2>;
    const auto tols = make_state<SD, T>({T(tol), T(tol)});
    auto solver = ode::getSolver<SD, SolverPolicy::Static>(Stepper{}, Oscillator{}, T(0), Oscillator::at<SD>(T(0)), tols, tols);
    Magnitude err{0};
    REQUIRE(solver.advance_until(T(5), [&err](const T& t, const auto& q, const T*){
        err = worst(max_error<SD>(q, Oscillator::at<SD>(t)), err);
        return true;
    }));
    return err;
}

TEMPLATE_TEST_CASE("Adaptive steppers are as accurate as their tolerance asks", "[steppers][convergence]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    // Two tolerances a factor 1000 apart, both within what T can resolve
    const double loose = std::is_same_v<T, float> ? 1e-3 : 1e-6;
    const double tight = loose / 1000;

    using Adaptive = std::tuple<tags::RK23, tags::RK45, tags::DOP853, tags::BDF>;
    for_each_type<Adaptive>([&]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            const Magnitude err_loose = adaptive_error<Stepper, T>(loose);
            const Magnitude err_tight = adaptive_error<Stepper, T>(tight);
            CHECK_THAT(err_loose, IsAtMost(100 * loose));
            CHECK_THAT(err_tight, IsAtMost(100 * tight));
            CHECK_THAT(err_tight, IsAtMost(err_loose.value / 20));
        }
    });
}


/**
 * One step of size h from the initial conditions, chosen so that the stepper's true local error,
 * O(h^(order + 1)), is 1000 machine epsilons of T. A coefficient that is only accurate to double
 * precision adds an error of ~1e-17 * h instead, so with an extended type this tells apart a
 * stepper that computes in T from one that merely stores T. The same holds for its dense output.
 */
template<typename Stepper, typename T>
void check_full_precision_step(){
    using SD = StaticSize<2>;
    using O = Orders<Stepper>;
    const T target = T(1000 * epsilon<T>());
    const T h = pow(target, T(1) / T(O::step + 1));

    // The tolerance is loose enough for the first step to be accepted at size h
    const auto loose = make_state<SD, T>({1, 1});
    auto solver = ode::getSolver<SD, SolverPolicy::Static>(Stepper{}, Oscillator{}, T(0), Oscillator::at<SD>(T(0)), loose, loose, T(0), T(0), h);
    REQUIRE(solver.advance());
    REQUIRE(solver.t() == h);

    CAPTURE(to_double(h));
    CHECK_THAT(max_error<SD>(solver.vector_new(), Oscillator::at<SD>(h)), IsAtMost(10 * to_double(target)));

    const T t_mid = h / 2;
    auto q_mid = solver.vector_new();
    solver.light_interp()(q_mid, t_mid);
    CHECK_THAT(max_error<SD>(q_mid, Oscillator::at<SD>(t_mid)), IsAtMost(10 * to_double(T(pow(h, O::dense + 1)))));
}

TEMPLATE_TEST_CASE("Every stepper computes to the full precision of its scalar type", "[steppers][precision]", ODECRAFT_TEST_SCALARS){
    for_each_type<AllSteppers>([]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            check_full_precision_step<Stepper, TestType>();
        }
    });
}

// The stiff problem with its exact Jacobian, and without autodiff
template<typename T, typename SD>
struct PlainStiffDecay {
    void Rhs(state_t<T, SD>& out, const T& t, const state_t<T, SD>& q) const { StiffDecay::Rhs(out, t, q); }
};

template<typename T, typename SD>
struct StiffDecayWithJac : PlainStiffDecay<T, SD> {
    void Jac(jacmat_t<T, SD>& J, const T& /*t*/, const state_t<T, SD>& /*q*/) const { J(0, 0) = StiffDecay::lambda; }
};

template<typename OdeType, typename T>
void check_bdf_on_stiff_problem(ode::JacPolicy expected_policy, size_t max_steps){
    using SD = StaticSize<1>;
    const auto tols = make_state<SD, T>({T(1e-6)});
    auto bdf = ode::getSolver<SD, SolverPolicy::Static>(tags::BDF{}, OdeType{}, T(0), make_state<SD, T>({1}), tols, tols);
    CHECK(decltype(bdf)::JP == expected_policy);

    REQUIRE(bdf.advance_until(T(10)));
    const T exact = cos(T(10));
    CHECK_THAT(abs_error(current_state(bdf)[0], exact), IsAtMost(1e-4));
    CHECK(bdf.step_count() <= max_steps);
}

TEMPLATE_TEST_CASE("BDF solves a stiff problem in far fewer steps than an explicit stepper", "[steppers][bdf][stiff]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<1>;
    const auto tols = make_state<SD, T>({T(1e-6)});

    // Stability alone holds RK45 to |h| < 3.3/|lambda| there, whatever the tolerance
    auto rk45 = ode::getSolver<SD, SolverPolicy::Static>(tags::RK45{}, StiffDecay{}, T(0), make_state<SD, T>({1}), tols, tols);
    REQUIRE(rk45.advance_until(T(10)));
    const size_t explicit_steps = rk45.step_count();
    CHECK(explicit_steps > 3000);

    SECTION("with an autodiff Jacobian"){
        check_bdf_on_stiff_problem<StiffDecay, T>(ode::JacPolicy::Autodiff, explicit_steps / 10);
    }
    SECTION("with an exact Jacobian"){
        check_bdf_on_stiff_problem<StiffDecayWithJac<T, SD>, T>(ode::JacPolicy::Exact, explicit_steps / 10);
    }
    SECTION("with a finite-difference Jacobian"){
        check_bdf_on_stiff_problem<PlainStiffDecay<T, SD>, T>(ode::JacPolicy::Approx, explicit_steps / 10);
    }
}


TEMPLATE_TEST_CASE("Every stepper reports its name and its nature", "[steppers]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    const auto q0 = Oscillator::at<SD>(T(0));

    for_each_type<AllSteppers>([&]<typename Stepper>(){
        DYNAMIC_SECTION(Stepper::name){
            const auto solver = make_solver<Stepper, SD, SolverPolicy::Static>(Oscillator{}, T(0), q0);
            CHECK(solver.method() == Stepper::name);
            CHECK(solver.get_method() == Stepper::name);
            CHECK(ode::Steppers::getName(ode::Steppers::getID(Stepper::name)) == Stepper::name);
        }
    });

    using Ode = PlainOscillator<T, SD>;
    STATIC_REQUIRE(ode::BDF<T, SD, Ode, SolverPolicy::Static>::IS_IMPLICIT);
    STATIC_REQUIRE_FALSE(ode::RK45<T, SD, Ode, SolverPolicy::Static>::IS_IMPLICIT);
    STATIC_REQUIRE(ode::Euler<T, SD, Ode, SolverPolicy::Static>::ERR_EST_ORDER == 1);
    STATIC_REQUIRE(ode::RK4<T, SD, Ode, SolverPolicy::Static>::ERR_EST_ORDER == 4);
    STATIC_REQUIRE(ode::RK23<T, SD, Ode, SolverPolicy::Static>::ERR_EST_ORDER == 2);
    STATIC_REQUIRE(ode::RK45<T, SD, Ode, SolverPolicy::Static>::ERR_EST_ORDER == 4);
    STATIC_REQUIRE(ode::DOP853<T, SD, Ode, SolverPolicy::Static>::ERR_EST_ORDER == 7);
    STATIC_REQUIRE(std::is_same_v<typename ode::RK45<T, SD, Ode, SolverPolicy::Static>::State, state_t<T, SD>>);
}


TEMPLATE_TEST_CASE("Euler can be built from its step alone", "[steppers][euler]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<2>;
    using Euler = ode::Euler<T, SD, Oscillator, SolverPolicy::Static>;
    const auto q0 = Oscillator::at<SD>(T(0));
    const T h = T(1e-3);

    Euler short_form(Oscillator{}, T(0), q0, h);
    Euler full_form(Oscillator{}, T(0), q0, make_state<SD, T>({1, 1}), make_state<SD, T>({1, 1}), T(0), T(0), h);
    CHECK(short_form.stepsize() == h);

    REQUIRE(short_form.advance_until(T(1)));
    REQUIRE(full_form.advance_until(T(1)));
    CHECK(same_state<SD>(short_form.vector_new(), full_form.vector_new()));
    CHECK(short_form.step_count() == full_form.step_count());
}

} // namespace
} // namespace odetest
