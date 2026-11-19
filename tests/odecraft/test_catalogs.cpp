// The stepper catalogs and their ids: compile-time facts and lookups by name, which do not
// depend on a scalar type, so this source is compiled once.

#include "odecraft_test.hpp"
#include "renamed_stepper.hpp"

#include <string>

namespace odetest {
namespace {

using ode::SolverPolicy;

TEST_CASE("The catalog of built-in steppers names them and gives each an id", "[factories][catalog]"){
    using ode::Steppers;
    STATIC_REQUIRE(Steppers::id<tags::Euler>.index() == 0);
    STATIC_REQUIRE(Steppers::id<tags::RK4>.index() == 1);
    STATIC_REQUIRE(Steppers::id<tags::RK23>.index() == 2);
    STATIC_REQUIRE(Steppers::id<tags::RK45>.index() == 3);
    STATIC_REQUIRE(Steppers::id<tags::DOP853>.index() == 4);
    STATIC_REQUIRE(Steppers::id<tags::BDF>.index() == 5);
    STATIC_REQUIRE(Steppers::id<tags::BDF> != Steppers::id<tags::RK45>);

    for (const std::string name : {"Euler", "RK4", "RK23", "RK45", "DOP853", "BDF"}){
        CAPTURE(name);
        CHECK(Steppers::getName(Steppers::getID(name)) == name);
        CHECK(Steppers::from_stepper<std::string>(Steppers::getID(name), []<typename S>(){ return std::string(S::name); }) == name);
    }
    CHECK_THROWS_AS(Steppers::getID("Midpoint"), std::invalid_argument);
    CHECK_THROWS_AS(Steppers::getID(""), std::invalid_argument);
}

template<typename C, typename Arg>
concept takes_as_id = requires(const Arg& arg) { C::getName(arg); };

TEST_CASE("A catalog's ids belong to it alone", "[factories][catalog]"){
    using Selection = ode::Catalog<tags::RK45, tags::BDF>;
    using Extended = ode::StepperCatalog<tags::RenamedRK45>;

    STATIC_REQUIRE(Selection::id<tags::BDF>.index() == 1);
    STATIC_REQUIRE(Extended::id<tags::RenamedRK45>.index() == 6);
    CHECK(Selection::getName(Selection::id<tags::RK45>) == "RK45");
    CHECK_THROWS_AS(Selection::getID("Euler"), std::invalid_argument);

    // An id is made by its catalog only, so it always names one of its steppers
    STATIC_REQUIRE(takes_as_id<Selection, Selection::Id>);
    STATIC_REQUIRE_FALSE(takes_as_id<Selection, size_t>);
    STATIC_REQUIRE_FALSE(takes_as_id<Selection, ode::Steppers::Id>);
    STATIC_REQUIRE_FALSE(takes_as_id<ode::Steppers, Selection::Id>);
    STATIC_REQUIRE_FALSE(std::is_constructible_v<Selection::Id, size_t>);
    STATIC_REQUIRE_FALSE(std::is_default_constructible_v<Selection::Id>);
    STATIC_REQUIRE_FALSE(std::is_convertible_v<Selection::Id, size_t>);
    STATIC_REQUIRE(std::is_trivially_copyable_v<Selection::Id>);

    STATIC_REQUIRE(ode::detail::isStepper<tags::RK45>);
    STATIC_REQUIRE_FALSE(ode::detail::isStepper<ode::Steppers::Id>);
    STATIC_REQUIRE(tags::DOP853::name == "DOP853");
    STATIC_REQUIRE(std::is_same_v<tags::RK45::type<double, StaticSize<2>, Oscillator, SolverPolicy::Static>,
                                  ode::RK45<double, StaticSize<2>, Oscillator, SolverPolicy::Static>>);
}

} // namespace
} // namespace odetest
