// Precision beyond double: what the extended scalar types buy over a whole integration.
// Compiled once, over the extended types only, rather than once per scalar type.

#include "../support/odecraft_test.hpp" // IWYU pragma: keep

namespace odetest {
namespace {

using ode::SolverPolicy;

/// @brief Integrates the oscillator to t = 0.25 at the tolerance tol, and bounds its error.
template<typename Stepper, typename T>
void check_precision(double tol, double bound){
    using SD = StaticSize<2>;
    REQUIRE(epsilon<T>() * 5 < tol);
    const auto tols = make_state<SD, T>({T(tol), T(tol)});
    auto solver = ode::getSolver<SD, SolverPolicy::Static>(Stepper{}, Oscillator{}, T(0), Oscillator::at<SD>(T(0)), tols, tols);
    REQUIRE(solver.advance_until(T(0.25)));
    CHECK_THAT(max_error<SD>(current_state(solver), Oscillator::at<SD>(T(0.25))), IsAtMost(bound));
}

TEMPLATE_TEST_CASE("Extended precision scalars resolve a solution beyond double precision", "[steppers][precision]",
                   long double, mpfr::mpreal, lazex::LazyType<mpfr::mpreal>){
    using T = TestType;
    constexpr bool multi = is_multiprecision<T>;
    // An error that a double could not even represent the solution to: below its epsilon, 2.2e-16
    SECTION("with RK45"){
        check_precision<tags::RK45, T>(multi ? 1e-24 : 1e-18, multi ? 1e-22 : 2e-16);
    }
    // And for DOP853 past the 30 digits its constants are published to, which only an exact
    // tableau can reach: the published ones would hold the error estimate above about 1e-28
    SECTION("with DOP853"){
        check_precision<tags::DOP853, T>(multi ? 1e-32 : 1e-18, multi ? 1e-30 : 2e-16);
    }
}

} // namespace
} // namespace odetest
