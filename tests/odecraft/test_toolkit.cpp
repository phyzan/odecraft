// The toolkit: root finding, norms and linear algebra, the containers and functors the solvers
// are built from, small numeric helpers, and the compile-time concepts that decide how a system
// is differentiated.

#include "odecraft_test.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace odetest {
namespace {

using ode::RootStatus;


TEMPLATE_TEST_CASE("Bisection locates a root to the requested tolerance", "[toolkit][roots]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    const T root = sqrt(T(2));
    const auto f = [](const T& x) -> T { return x*x - 2; };

    SECTION("to a finite tolerance, on the side of the root the bracket closes from"){
        const T ftol = T(std::is_same_v<T, float> ? 1e-4 : 1e-8);
        T out = 0, f_out = 0;
        REQUIRE(ode::bisect_root(out, &f_out, f, T(0), T(2), ftol) == RootStatus::Success);
        CHECK_THAT(abs_error(f_out, T(0)), IsAtMost(to_double(ftol)));
        CHECK(f_out == f(out));
        // out is the converged right bracket: f has the sign there that it has at b = 2
        CHECK(out >= root);
    }
    SECTION("to machine precision with a zero tolerance, and still terminates"){
        T out = 0;
        REQUIRE(ode::bisect_root(out, static_cast<T*>(nullptr), f, T(0), T(2), T(0)) == RootStatus::Success);
        CHECK_THAT(rel_error(out, root), IsAtMost(4 * epsilon<T>()));
    }
    SECTION("from endpoint values given in advance"){
        T out = 0;
        REQUIRE(ode::bisect_root(out, static_cast<T*>(nullptr), f, T(0), T(2), f(T(0)), f(T(2)), T(0)) == RootStatus::Success);
        CHECK_THAT(rel_error(out, root), IsAtMost(4 * epsilon<T>()));
    }
    SECTION("over a reversed bracket, as for a backward integration"){
        T out = 0;
        REQUIRE(ode::bisect_root(out, static_cast<T*>(nullptr), f, T(2), T(0), T(0)) == RootStatus::Success);
        CHECK_THAT(rel_error(out, root), IsAtMost(4 * epsilon<T>()));
        CHECK(out <= root);
    }
    SECTION("and leaves its output alone without a sign change"){
        T out = T(-7);
        const auto positive = [](const T& x) -> T { return x*x + 1; };
        CHECK(ode::bisect_root(out, static_cast<T*>(nullptr), positive, T(-1), T(1), T(0)) == RootStatus::NotBracketed);
        CHECK(out == T(-7));
    }
}


TEMPLATE_TEST_CASE("Root mean square norms weigh each component by its scale", "[toolkit][linalg]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<4>;
    const auto q = make_state<SD, T>({1, -1, 3, -3});
    // sqrt((1 + 1 + 9 + 9)/4) = sqrt(5)
    CHECK_THAT(rel_error(ode::rms_norm<SD, T>(q), T(sqrt(T(5)))), IsAtMost(4 * epsilon<T>()));
    // With scales (1, 1, 3, 3) every component weighs 1
    CHECK_THAT(rel_error(ode::rms_norm<SD, T>(q, make_state<SD, T>({1, 1, 3, 3})), T(1)), IsAtMost(4 * epsilon<T>()));
}

TEMPLATE_TEST_CASE("Dense matrices are inverted multiplied and reduced to their determinant", "[toolkit][linalg]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    const double tol = 100 * epsilon<T>();
    // Needs a row exchange: its leading entry is zero. Its determinant is 3.
    const std::array<T, 9> A = {0, 2, 1,
                                1, 1, 1,
                                2, 1, 0};

    SECTION("inv_mat_row_major inverts it"){
        std::array<T, 9> inv{}, lu{};
        std::array<T, 3> work{};
        std::array<size_t, 3> pivot{};
        ode::inv_mat_row_major(inv.data(), A.data(), 3, work.data(), lu.data(), pivot.data());
        for (size_t i = 0; i < 3; i++){
            for (size_t j = 0; j < 3; j++){
                T product = 0;
                for (size_t k = 0; k < 3; k++){
                    product += A[i*3 + k] * inv[k*3 + j];
                }
                CHECK_THAT(abs_error(product, T(i == j ? 1 : 0)), IsAtMost(tol));
            }
        }
    }
    SECTION("detLU_row_major gives its determinant, zero for a singular one, one for an empty one"){
        auto copy = A;
        CHECK_THAT(abs_error(ode::detLU_row_major(copy.data(), 3), T(3)), IsAtMost(tol));
        std::array<T, 4> singular = {1, 2, 2, 4};
        CHECK(ode::detLU_row_major(singular.data(), 2) == T(0));
        CHECK(ode::detLU_row_major(static_cast<T*>(nullptr), 0) == T(1));
    }
    SECTION("matmul multiplies the leading block of column-major matrices"){
        ndspan::FArray2D<T, 3, 3> M, I, out;
        for (size_t i = 0; i < 3; i++){
            for (size_t j = 0; j < 3; j++){
                M(i, j) = A[i*3 + j];
                I(i, j) = (i == j) ? 1 : 0;
                out(i, j) = 42;
            }
        }
        ode::matmul(out, M, I, 2);
        CHECK(out(0, 1) == T(2));
        CHECK(out(1, 0) == T(1));
        CHECK(out(2, 2) == T(42)); // outside the leading 2x2 block, untouched
    }
}


TEST_CASE("Vectors hold items of fixed or dynamic count", "[toolkit][vector]"){
    SECTION("a dynamic vector grows"){
        ode::Vector<double> v(2, 1.5);
        REQUIRE(v.size() == 2);
        v.push_back(3.0);
        v.emplace_back(4.0);
        CHECK(v.size() == 4);
        CHECK(v[0] == 1.5);
        CHECK(v.data()[3] == 4.0);
        double sum = 0;
        for (double x : v){
            sum += x;
        }
        CHECK(sum == 10.0);
        v.resize(1);
        CHECK(v.size() == 1);
    }
    SECTION("a fixed vector has its size at compile time"){
        ode::Vector<int, 3> v(3, 7);
        STATIC_REQUIRE(sizeof(v) == 3 * sizeof(int));
        CHECK(v.size() == 3);
        CHECK(v[2] == 7);
    }
    SECTION("items are constructed from the arguments given"){
        ode::Vector<std::string> v(2, size_t(3), 'x');
        REQUIRE(v.size() == 2);
        CHECK(v[1] == "xxx");
    }
    SECTION("make_vector moves in items that cannot be copied"){
        auto a = std::make_unique<int>(1);
        auto b = std::make_unique<int>(2);
        ode::Vector<std::unique_ptr<int>> v = ode::make_vector<std::unique_ptr<int>>(std::move(a), std::move(b));
        REQUIRE(v.size() == 2);
        CHECK(*v[1] == 2);
        CHECK(a == nullptr);
    }
}

TEST_CASE("Light functors refer to their data and detach into copies", "[toolkit][functor]"){
    int bound = 10;
    auto f = ode::make_light_functor([](int x, const int& data, const int& constant){ return x + data + constant; }, bound, 100);
    CHECK(f(1) == 111);

    // An lvalue is held by reference, so the functor sees it change. A temporary is copied.
    bound = 20;
    CHECK(f(1) == 121);

    auto detached = f.detach();
    bound = 30;
    CHECK(detached(1) == 121);
    CHECK(f(1) == 131);
}

TEST_CASE("Scratch scopes work on copies and write them back", "[toolkit][scratch]"){
    double scalar = 1;
    std::vector<double> vector = {1, 2};
    {
        auto scratch = ode::make_scratch(scalar, vector);
        auto& [s, v] = scratch;
        auto write_back = ode::make_write_back(scratch, scalar, vector);
        STATIC_REQUIRE(std::is_same_v<decltype(s), double>);          // trivially copyable: a copy
        STATIC_REQUIRE(std::is_same_v<decltype(v), std::vector<double>&>); // otherwise: the item itself
        s = 5;
        v[0] = 9;
        CHECK(scalar == 1);
    }
    CHECK(scalar == 5);
    CHECK(vector[0] == 9);
}

TEST_CASE("Workers lend scratch states", "[toolkit][scratch]"){
    ode::worker_t<std::array<double, 2>> trivial(std::array<double, 2>{1, 2});
    STATIC_REQUIRE(std::is_same_v<decltype(trivial.get()), std::array<double, 2>>);

    ode::worker_t<std::vector<double>> heap(std::vector<double>{1, 2, 3});
    STATIC_REQUIRE(std::is_same_v<decltype(heap.get()), std::vector<double>&>);
    heap.get()[0] = 7;
    CHECK(heap.get()[0] == 7);
    CHECK(heap.get().size() == 3);

    heap = std::vector<double>{4};
    CHECK(heap.get().size() == 1);
}


TEMPLATE_TEST_CASE("Numeric helpers", "[toolkit]", ODECRAFT_TEST_SCALARS){
    using T = TestType;
    using SD = StaticSize<3>;

    SECTION("signs"){
        CHECK(ode::sgn(T(-2)) == -1);
        CHECK(ode::sgn(T(0)) == 0);
        CHECK(ode::sgn(T(3)) == 1);
        CHECK(ode::sgn(T(1), T(2)) == 1);
        CHECK(ode::sgn(T(2), T(1)) == -1);
        CHECK(ode::sgn(T(2), T(2)) == 0);
    }
    SECTION("minima and maxima"){
        const T a = 1, b = 2;
        CHECK(&ode::max_ref(a, b) == &b);
        CHECK(&ode::min_ref(a, b) == &a);
        T out = 0;
        ode::set_max(out, a, b);
        CHECK(out == b);
        ode::set_min(out, a, b);
        CHECK(out == a);
        CHECK(ode::max_elem<T, SD>(make_state<SD, T>({1, 5, 2})) == T(5));
    }
    SECTION("step resizing within bounds"){
        T habs = 1, factor = 3;
        CHECK(ode::resize_step(factor, habs, T(0.1), T(10)));
        CHECK(habs == T(3));
        factor = 10;
        CHECK_FALSE(ode::resize_step(factor, habs, T(0.1), T(10)));
        CHECK(habs == T(10));
        CHECK(factor == T(10) / T(3));
    }
    SECTION("finiteness"){
        CHECK(ode::inf<T>() > T(std::numeric_limits<T>::max()));
        CHECK(ode::isFastFinite(T(1)));
        CHECK_FALSE(ode::isFastFinite(ode::inf<T>()));
        CHECK_FALSE(ode::isFastFinite(T(std::numeric_limits<double>::quiet_NaN())));

        auto q = make_state<SD, T>({1, 2, 3});
        CHECK(ode::all_are_finite<T, SD>(q));
        q[1] = T(std::numeric_limits<double>::quiet_NaN());
        CHECK_FALSE(ode::all_are_finite<T, SD>(q));

        auto J = make_jacmat<SD, T>(q);
        for (size_t i = 0; i < 3; i++){
            for (size_t j = 0; j < 3; j++){
                J(i, j) = 1;
            }
        }
        CHECK(ode::jac_all_are_finite<T, SD>(J));
        J(2, 0) = ode::inf<T>();
        CHECK_FALSE(ode::jac_all_are_finite<T, SD>(J));
    }
    SECTION("copying a state"){
        auto dst = make_state<SD, T>({0, 0, 0});
        const auto src = make_state<SD, T>({4, 5, 6});
        ode::copy_state<SD, T>(dst, src);
        CHECK(same_state<SD>(dst, src));
    }
}

TEST_CASE("Strings and timings", "[toolkit]"){
    CHECK(ode::GetStr("t = ", 1, ", h = ", 0.5) == "t = 1, h = 0.5");

    int calls = 0;
    const double ms = ode::timeit_ms([&calls]{ ++calls; });
    CHECK(calls == 1);
    CHECK(ms >= 0);
}

TEST_CASE("Every stepper status has a message", "[toolkit][status]"){
    using ode::StepperStatus;
    for (StepperStatus status : {StepperStatus::NewStep, StepperStatus::BetweenSteps}){
        CHECK(ode::status_message(status) == "Running");
    }
    for (StepperStatus status : {
        StepperStatus::Uninitialized,
        StepperStatus::BadICS,
        StepperStatus::TinyStep,
        StepperStatus::MinStep,
        StepperStatus::MaxStep,
        StepperStatus::TooSmallStep,
        StepperStatus::Diverges,
        StepperStatus::SingularMatrix,
        StepperStatus::Interrupted,
    }){
        CAPTURE(static_cast<int>(status));
        CHECK_FALSE(ode::status_message(status).empty());
        CHECK(ode::status_message(status) != "Running");
    }
    CHECK(ode::status_message(StepperStatus::Diverges) == "Integrator diverges");
}


TEST_CASE("The Jacobian policy follows from what a system provides", "[toolkit][concepts][jacobian]"){
    using T = double;
    using SD = StaticSize<2>;
    using State = state_t<T, SD>;
    using JacMat = jacmat_t<T, SD>;

    // An Rhs written for any state type is differentiated automatically
    STATIC_REQUIRE(ode::getJacPolicy<T, SD, Oscillator>() == ode::JacPolicy::Autodiff);
    STATIC_REQUIRE(ode::supportsDualRhs<Oscillator, 2, T, SD>);
    // One that only takes its own states is differenced
    STATIC_REQUIRE(ode::getJacPolicy<T, SD, PlainOscillator<T, SD>>() == ode::JacPolicy::Approx);
    STATIC_REQUIRE_FALSE(ode::supportsDualRhs<PlainOscillator<T, SD>, 1, T, SD>);
    // An explicit Jacobian always wins, a nullable one is checked at runtime
    STATIC_REQUIRE(ode::getJacPolicy<T, SD, OscillatorWithJac<T, SD>>() == ode::JacPolicy::Exact);
    STATIC_REQUIRE(ode::getJacPolicy<T, SD, NullableNonlinear<T, SD>>() == ode::JacPolicy::Nullable);

    STATIC_REQUIRE(ode::hasRhsFunc<Oscillator, T, State>);
    STATIC_REQUIRE(ode::hasJacFunc<OscillatorWithJac<T, SD>, T, State, JacMat>);
    STATIC_REQUIRE_FALSE(ode::hasJacFunc<PlainOscillator<T, SD>, T, State, JacMat>);
    STATIC_REQUIRE(ode::hasNullableJac<NullableNonlinear<T, SD>>);
    STATIC_REQUIRE(ode::isNullable<std::function<void()>>);
    STATIC_REQUIRE_FALSE(ode::isNullable<int>);
}

TEST_CASE("Callables are classified by the concepts the solvers constrain on", "[toolkit][concepts]"){
    using T = double;
    using State = std::array<T, 2>;
    const auto rhs = [](State& out, T /*t*/, const State& q){ out = q; };
    const auto objective = [](const T& t, const State& /*q*/) -> T { return t; };
    const auto observer = [](const T& /*t*/, const State& /*q*/, const T* /*checkpoint*/){ return true; };
    const auto interp = [](State& /*out*/, const T& /*t*/){};

    STATIC_REQUIRE(ode::isRhsFunc<decltype(rhs), T, State>);
    STATIC_REQUIRE(ode::isObjFun<decltype(objective), T, State>);
    STATIC_REQUIRE_FALSE(ode::isObjFun<decltype(rhs), T, State>);
    STATIC_REQUIRE(ode::isObserver<decltype(observer), T, State>);
    STATIC_REQUIRE(ode::isStateInterp<decltype(interp), T, State>);
    STATIC_REQUIRE(ode::isArray<std::vector<T>, T>);
    STATIC_REQUIRE_FALSE(ode::isArray<int, T>);
}

TEST_CASE("OdeData deduces the types of its callables", "[toolkit]"){
    using State = std::array<double, 2>;
    const auto rhs = ODECRAFT_LAMBDA(out, t, q){ out[0] = q[1]; out[1] = -q[0]; };

    auto rhs_only = ode::OdeData{rhs};
    STATIC_REQUIRE(std::is_same_v<decltype(rhs_only)::JacType, std::nullptr_t>);
    CHECK(rhs_only.Jac == nullptr);

    const auto jac = [](jacmat_t<double, StaticSize<2>>& /*J*/, double /*t*/, const State& /*q*/){};
    auto both = ode::OdeData{rhs, jac};
    STATIC_REQUIRE(std::is_same_v<decltype(both)::JacType, std::remove_const_t<decltype(jac)>>);

#if !defined(__clang__) || __clang_major__ >= 17
    // The designated-initializer spelling deduces the same types. It needs CTAD for aggregates,
    // which Clang only has from 17
    STATIC_REQUIRE(std::is_same_v<decltype(ode::OdeData{.Rhs = rhs}), decltype(rhs_only)>);
    STATIC_REQUIRE(std::is_same_v<decltype(ode::OdeData{.Rhs = rhs, .Jac = jac}), decltype(both)>);
#endif

    State out{};
    rhs_only.Rhs(out, 0.0, State{1, 2});
    CHECK(out == State{2, -1});
}

TEST_CASE("cache_t and scratch_t copy what is cheap and refer to the rest", "[toolkit]"){
    STATIC_REQUIRE(std::is_same_v<ode::cache_t<double>, const double>);
    STATIC_REQUIRE(std::is_same_v<ode::cache_t<std::vector<double>>, const std::vector<double>&>);
    STATIC_REQUIRE(std::is_same_v<ode::scratch_t<double>, double>);
    STATIC_REQUIRE(std::is_same_v<ode::scratch_t<std::vector<double>>, std::vector<double>&>);
}

} // namespace
} // namespace odetest
